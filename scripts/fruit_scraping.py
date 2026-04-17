import objaverse
import os
import random
import json
import shutil
from tqdm import tqdm

#config
SAVE_DIR = "fruit_objs"
METADATA_FILE = "fruit_metadata.json"
MAX_OBJECTS = 1000
BATCH_SIZE = 50
RANDOM_SEED = 42

os.makedirs(SAVE_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

#allowed file formats
ALLOWED_EXTENSIONS = {
    ".abc", ".usd", ".usda", ".usdc", ".usdz", ".svg",
    ".obj", ".ply", ".stl", ".fbx", ".bvh", ".glb", ".gltf"
}

#load metadata
print("loading annotations")
annotations = objaverse.load_annotations()
print(f"total annotations loaded: {len(annotations)}")

#fruit targets
FRUIT_TARGETS = {
    "cherry": ["cherry fruit", "cherry", "cherries", "prunus avium", "sour cherry"],
    "pomegranate": ["pomegranate fruit", "pomegranate", "punica granatum", "punica"],
    "blackberry": ["blackberry fruit", "blackberry", "blackberries", "rubus fruticosus", "bramble", "dewberry"],
    "plum": ["plum fruit", "plum", "plums", "prunus domestica", "prunus", "damson"],
    "peach": ["peach fruit", "peach", "peaches", "prunus persica", "nectarine"],
    "dragonfruit": ["dragon fruit", "pitaya", "pitahaya", "hylocereus", "hylocereus undatus", "dragonfruit"],
    "lemon": ["lemon fruit", "lemon", "lemons", "citrus limon"],
    "lime": ["lime fruit", "lime", "limes", "citrus aurantiifolia", "citrus latifolia"],
    "orange": ["orange fruit", "orange", "citrus sinensis", "mandarin", "tangerine", "clementine", "satsuma", "blood orange"],
    "apple": ["apple fruit", "apple", "apples", "malus domestica", "red apple", "green apple", "gala apple", "fuji apple", "braeburn apple"],
    "banana": ["banana", "bananas", "musa acuminata", "musa balbisiana", "plantain"],
    "strawberry": ["strawberry", "strawberries", "fragaria", "garden strawberry"],
    "pineapple": ["pineapple", "pineapples", "ananas comosus", "ananas"],
    "watermelon": ["watermelon", "watermelons", "citrullus lanatus", "citrullus"],
    "grapes": ["grape", "grapes", "vitis vinifera", "vitis", "bunch of grapes", "wine grape"],
    "pear": ["pear fruit", "pear", "pears", "pyrus", "pyrus communis"],
    "mango": ["mango", "mangos", "mangoes", "mangifera indica", "mangifera"],
    "blueberry": ["blueberry", "blueberries", "vaccinium corymbosum", "vaccinium"],
    "raspberry": ["raspberry", "raspberries", "rubus idaeus", "rubus"],
    "kiwi": ["kiwi fruit", "kiwi", "kiwis", "actinidia", "kiwi apple"],
    "avocado": ["avocado", "avocados", "persea americana", "persea"],
    "coconut": ["coconut", "coconuts", "cocos nucifera", "cocos"]
}

#blacklist
BLACKLIST = [
    "juice", "soda", "drink", "can", "bottle", "box", "package", "advertisement",
    "logo", "icon", "vector", "illustration", "cartoon", "character", "game",
    "toy", "computer", "iphone", "ipad", "macbook", "apple store", "watch",
    "furniture", "room", "house", "car", "vehicle", "clothing", "fashion",
    "lowpoly", "low poly", "pixel", "voxel", "abstract", "geometric",
    "princess peach", "mario", "nintendo", "blackberry phone", "blackberry device",
    "cherry keyboard", "cherry mx", "logo", "brand", "company", "technology",
    "electronic", "monitor", "screen", "keyboard", "mouse", "laptop", "pc",
    "architecture", "building", "city", "street", "landscape", "interior",
    "decoration", "art", "sculpture", "statue", "character", "human", "person"
]

#confirmation tags
FRUIT_CONFIRMATION_TAGS = [
    "fruit", "food", "nature", "plant", "vegetable", "produce", "organic", 
    "botany", "botanical", "healthy", "garden", "agriculture", "orchard"
]

#filter candidates
uids_to_download = []
seen_uids = set()
uid_to_label = {}
TARGET_MIN_COUNT = 40

#preprocess metadata
processed_meta = {}
for uid, meta in tqdm(annotations.items(), desc="Preprocessing"):
    tags = " ".join(t.get("name", "") if isinstance(t, dict) else str(t) for t in meta.get("tags", [])).lower()
    processed_meta[uid] = {
        "name": str(meta.get("name", "")).lower(),
        "tags": tags,
        "description": str(meta.get("description", "")).lower()
    }

#search loop
for label, synonyms in tqdm(FRUIT_TARGETS.items(), desc="Targets"):
    found_for_target = 0
    for syn in synonyms:
        if found_for_target >= TARGET_MIN_COUNT: break
        for uid, meta in processed_meta.items():
            if uid in seen_uids: continue
            if found_for_target >= TARGET_MIN_COUNT: break
            if syn in meta["name"]:
                text = f"{meta['name']} {meta['tags']} {meta['description']}"
                if not any(b in text for b in BLACKLIST):
                    if any(c in text for c in FRUIT_CONFIRMATION_TAGS):
                        uids_to_download.append(uid)
                        seen_uids.add(uid)
                        uid_to_label[uid] = label
                        found_for_target += 1
        if found_for_target < TARGET_MIN_COUNT:
            for uid, meta in processed_meta.items():
                if uid in seen_uids: continue
                if found_for_target >= TARGET_MIN_COUNT: break
                if syn in meta["tags"] or syn in meta["description"]:
                    text = f"{meta['name']} {meta['tags']} {meta['description']}"
                    if not any(b in text for b in BLACKLIST):
                        if any(c in text for c in FRUIT_CONFIRMATION_TAGS):
                            uids_to_download.append(uid)
                            seen_uids.add(uid)
                            uid_to_label[uid] = label
                            found_for_target += 1

#prefilter by file format
import gzip
_paths_file = os.path.expanduser("~/.objaverse/hf-objaverse-v1/object-paths.json.gz")
with gzip.open(_paths_file, "rt") as f:
    object_paths = json.load(f)

uids_to_download = [
    uid for uid in uids_to_download
    if os.path.splitext(object_paths.get(uid, ""))[1].lower() in ALLOWED_EXTENSIONS
]

#sampling
uids_to_download = random.sample(uids_to_download, min(MAX_OBJECTS, len(uids_to_download)))
print(f"final dataset size: {len(uids_to_download)}")

#download and rename
print(f"downloading {len(uids_to_download)} fruits")
keyword_counters = {}
for i in range(0, len(uids_to_download), BATCH_SIZE):
    batch = uids_to_download[i:i + BATCH_SIZE]
    try:
        objects = objaverse.load_objects(uids=batch)
        for uid, obj_path in objects.items():
            ext = os.path.splitext(obj_path)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                label = uid_to_label.get(uid, "fruit").replace(" ", "_")
                count = keyword_counters.get(label, 0) + 1
                keyword_counters[label] = count
                new_filename = f"{label}{count}{ext}"
                new_path = os.path.join(SAVE_DIR, new_filename)
                shutil.copy2(obj_path, new_path)
    except Exception as e:
        print(f"batch failed: {e}")

print(f"scraping done saved in {SAVE_DIR}")
