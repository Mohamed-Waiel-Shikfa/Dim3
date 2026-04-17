# Dim3: A 3D Object Understanding Machine

**Course:** 15-288 Spring 2026

**Project:** Query Answering Machine (QuAM) for 3D Meshes

---

## Overview
Dim3 is a machine learning project designed to translate raw 3D geometry into meaningful semantic and geometric insights. By serving as a preprocessing layer, Dim3 aims to bridge the gap between 3D spatial data and natural language understanding.

---

## Architecture & Roadmap

The project is structured as **five progressive model iterations**, each building on lessons learned from the previous one.

| # | Iteration | Representation | Approach | Status |
|---|-----------|---------------|----------|--------|
| 1 | **3D Voxel CNN** | Volumetric voxel grid | 3-D Convolutional Neural Network | 🔲 Planned |
| 2 | **PointNet** | Raw point cloud | TBD, candidates: PointNet / PointNet++ | 🔲 Planned |
| 3 | **Mesh GNN** | Graph (vertices + edges, maybe somehow faces as well) | Graph Neural Network | 🔲 Planned |

---

## Repository Structure

```text
.
├── data/
│   └── fruits/                  # Sample 3D meshes (.obj, .stl) for pipeline testing
├── models/
│   └── placeholder.py           # ML models and architectures (Iterative updates)
├── notebooks/
│   ├── DataPreparation.ipynb    # Deliverable D2: Data wrangling, EDA, and feature extraction
│   └── QuAM_report.ipynb        # Deliverable D3: Model iterations and QuAM interface demo
├── other/
│   ├── feedback_on_D1.txt       # D1 proposal Feedback
│   ├── project_outline.pdf      # Project rubrics and guidelines
│   └── project_Proposal_D1.pdf  # D1 proposal
├── scripts/
│   └── blender_pipeline.py      # Headless Blender Python script (bpy) for data wrangling
├── web_interface/
│   └── index.html               # Frontend for the Query Answering Machine
├── README.md
├── requirements.txt
├── LISCENCE
└── .gitignore
```

---

## Contributing

This is a university course project. External contributions are not expected, but feel free to open an issue if you spot something.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
