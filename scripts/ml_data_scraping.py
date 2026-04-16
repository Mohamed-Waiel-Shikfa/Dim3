import objaverse
import os
import random
import json
import shutil
from tqdm import tqdm

os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/hf_cache"
os.environ["OBJAVERSE_CACHE_DIR"] = "/tmp/objaverse_cache"

# ----------------------------
# 0. Config
# ----------------------------
SAVE_DIR = "medical_objs"
METADATA_FILE = "medical_metadata.json"
MAX_OBJECTS = 500
TEMP_LIMIT = 2000
BATCH_SIZE = 50
RANDOM_SEED = 42

os.makedirs(SAVE_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

# ----------------------------
# 0.1 Allowed file formats
# ----------------------------
ALLOWED_EXTENSIONS = {
    ".abc", ".usd", ".usda", ".usdc", ".usdz", ".svg",
    ".obj", ".ply", ".stl", ".fbx", ".bvh", ".glb", ".gltf"
}

# ----------------------------
# 1. Load metadata
# ----------------------------
def load_annotations_safe():
    """Load annotations, wiping corrupted cache and retrying once if needed."""
    import glob
    for attempt in range(2):
        try:
            return objaverse.load_annotations()
        except EOFError:
            if attempt == 0:
                print("⚠️  Corrupted cache detected — clearing and retrying...")
                cache_dirs = [
                    os.path.expanduser("~/.objaverse"),
                    "/tmp/objaverse_cache",
                    os.environ.get("OBJAVERSE_CACHE_DIR", ""),
                ]
                for d in cache_dirs:
                    if d and os.path.exists(d):
                        for f in glob.glob(os.path.join(d, "**/*.json.gz"), recursive=True):
                            os.remove(f)
                            print(f"   Deleted: {f}")
            else:
                raise RuntimeError("Failed to load annotations after cache reset.")

print("Loading annotations...")
annotations = load_annotations_safe()
print(f"Total annotations loaded: {len(annotations)}")

# ----------------------------
# 2. Keywords
# ----------------------------

# Must match at least one of these (human medical context only)
MEDICAL_KEYWORDS = [
    # Human bones
    "human skull", "human vertebra", "human femur", "human tibia",
    "human humerus", "human pelvis", "human skeleton", "human bone",
    "human rib", "cranium", "mandible", "scapula", "clavicle",
    "sternum", "patella", "fibula", "ulna", "radius bone", "sacrum",
    "human spine", "spinal cord", "intervertebral",
    # Human organs
    "human heart", "human lung", "human brain", "human kidney",
    "human liver", "human anatomy", "human muscle", "human organ",
    "cardiac", "aorta", "trachea", "bronchi", "alveoli",
    "cerebral", "cerebellum", "hippocampus", "cortex",
    "renal", "hepatic", "pancreas", "gallbladder", "bladder",
    "uterus", "ovary", "prostate",
    # Medical / surgical
    "surgical instrument", "medical instrument", "scalpel",
    "forceps", "retractor", "endoscope", "stethoscope",
    "syringe", "medical device", "hospital equipment",
    # Implants / prosthetics
    "hip implant", "hip replacement", "knee implant", "knee replacement",
    "dental implant", "bone implant", "orthopedic implant",
    "prosthetic limb", "prosthetic arm", "prosthetic leg",
    # Medical imaging / models
    "ct scan", "mri scan", "medical scan", "dicom",
    "anatomical model", "medical model",
]

# Reject if ANY of these appear anywhere in the text
BLACKLIST = [
    # Animals
    "animal", "bird", "fish", "reptile", "insect", "dinosaur",
    "dog", "cat", "horse", "cow", "pig", "sheep", "deer",
    "eagle", "owl", "crow", "parrot", "turtle", "frog",
    "shark", "whale", "dolphin", "bear", "wolf", "lion",
    "fossil", "specimen", "taxidermy", "wildlife", "zoo",
    # Architecture / ruins / objects
    "castle", "ruins", "building", "church", "temple", "tower",
    "bridge", "wall", "stone", "inscription", "monument",
    "statue", "sculpture", "artifact", "archaeological",
    "medieval", "ancient", "roman", "greek", "viking",
    # Entertainment / games
    "game", "gaming", "cartoon", "anime", "manga",
    "fantasy", "sci-fi", "scifi", "fictional", "character",
    "weapon", "sword", "gun", "armor", "shield",
    "lowpoly", "low poly", "pixel", "voxel",
    "robot", "mech", "alien", "monster", "zombie",
    # Nature / environment
    "plant", "tree", "rock", "terrain", "landscape",
    "mineral", "crystal", "gem", "shell",
    # Vehicles / objects
    "car", "vehicle", "plane", "ship", "boat",
    "furniture", "food", "cloth",
]

# ----------------------------
# 3. Filter objects
# ----------------------------
print("Filtering objects...")
uids_to_download = []

for uid, meta in tqdm(annotations.items()):
    # tags can be a list of dicts like {"name": "skull"}
    tags = meta.get("tags", [])
    tag_text = " ".join(
        t.get("name", "") if isinstance(t, dict) else str(t)
        for t in tags
    )

    text = " ".join([
        str(meta.get("name", "")),
        tag_text,
        str(meta.get("description", ""))
    ]).lower()

    if not any(k in text for k in MEDICAL_KEYWORDS):
        continue
    if any(b in text for b in BLACKLIST):
        continue

    uids_to_download.append(uid)
    if len(uids_to_download) >= TEMP_LIMIT:
        break

print(f"Collected {len(uids_to_download)} keyword candidates")

# ----------------------------
# 3.5 Pre-filter by file format (avoids downloading unsupported files)
# ----------------------------
print("Loading object paths for format pre-filter...")
# Load object paths directly from the cached file
import gzip
_paths_file = os.path.expanduser("~/.objaverse/hf-objaverse-v1/object-paths.json.gz")
with gzip.open(_paths_file, "rt") as f:
    object_paths = json.load(f)  # uid -> e.g. "glbs/000/uid.glb"

uids_to_download = [
    uid for uid in uids_to_download
    if os.path.splitext(object_paths.get(uid, ""))[1].lower() in ALLOWED_EXTENSIONS
]
print(f"After format pre-filter: {len(uids_to_download)} candidates")

# ----------------------------
# 4. Sample to final size
# ----------------------------
uids_to_download = random.sample(
    uids_to_download,
    min(MAX_OBJECTS, len(uids_to_download))
)
print(f"Final dataset size: {len(uids_to_download)}")

# ----------------------------
# 5. Download + filter formats
# ----------------------------
print("Downloading objects...")
saved_objects = {}  # uid -> saved path

for i in range(0, len(uids_to_download), BATCH_SIZE):
    batch = uids_to_download[i:i + BATCH_SIZE]
    print(f"Downloading batch {i} → {i + len(batch)}")

    try:
        objects = objaverse.load_objects(uids=batch)
    except Exception as e:
        print(f"  ⚠️  Batch failed: {e}")
        continue

    for uid, obj_path in objects.items():
        ext = os.path.splitext(obj_path)[1].lower()

        if ext not in ALLOWED_EXTENSIONS:
            print(f"  ⏭  Skipping unsupported format {ext} for {uid}")
            continue

        # FIX: prefix filename with uid to avoid collisions
        basename = os.path.basename(obj_path)
        filename = f"{uid[:8]}_{basename}"
        new_path = os.path.join(SAVE_DIR, filename)

        if not os.path.exists(new_path):
            try:
                # FIX: use shutil.copy2 instead of os.rename (cross-filesystem safe)
                shutil.copy2(obj_path, new_path)
                saved_objects[uid] = new_path
            except Exception as e:
                print(f"  ⚠️  Failed to copy {uid}: {e}")
        else:
            saved_objects[uid] = new_path

# ----------------------------
# 6. Save metadata (post-download, only saved objects)
# ----------------------------
print("Saving metadata...")
metadata_subset = {
    uid: {
        **annotations[uid],
        "_saved_path": saved_objects[uid],
        "_format": os.path.splitext(saved_objects[uid])[1].lower(),
    }
    for uid in saved_objects
}

with open(METADATA_FILE, "w") as f:
    json.dump(metadata_subset, f, indent=2)

print(f"\n✅ Done!")
print(f"   Attempted : {len(uids_to_download)}")
print(f"   Saved     : {len(saved_objects)}")
print(f"   Skipped   : {len(uids_to_download) - len(saved_objects)}")
print(f"   Output dir: {SAVE_DIR}/")
print(f"   Metadata  : {METADATA_FILE}")