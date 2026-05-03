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
├── data
│   ├── fruit
│   ├── fruit_dataset.pdf
│   ├── fruit_graph
│   ├── fruit_low_poly
│   ├── fruit_objs
│   ├── fruit_points
│   ├── fruit_remeshed
│   └── fruit_voxels
├── LICENSE
├── models
│   └── placeholder.py
├── notebooks
│   ├── DataPreparation.ipynb
│   └── QuAM_report.ipynb
├── other
│   ├── feedback_on_D1.txt
│   ├── project_outline.pdf
│   └── project_Proposal_D1.pdf
├── README.md
├── requirements.txt
├── scripts
│   ├── 3d_file_to_obj.py
│   ├── fruit_scraping.py
│   ├── mesh_cleanup.py
│   ├── mesh_to_3D_cnn_input_feature.py
│   ├── mesh_to_gnn_input_feature.py
│   ├── mesh_to_pointnet_input_feature.py
│   ├── model_sorter.html
│   └── verify_model_input.py
└── web_interface
    ├── 3d_object_viewer.html
    └── dim3_engine
        ├── main.py
        └── templates
            ├── base.html
            ├── data_processing.html
            ├── model_evaluation.html
            └── model_training.html
```

---

## Running the Web Interface

### Setting Up the Environment

1. **Create the virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   * **Windows:**
       ```cmd
       venv\Scripts\activate
       ```
   * **Linux / macOS:**
       ```bash
       source venv/bin/activate
       ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Launching the Application

4. **Navigate to the project directory:**
   ```bash
   cd web_interface/dim3_engine
   ```

5. **Start the server:**
   ```bash
   uvicorn main:app --reload
   ```

---

## Contributing

This is a university course project. External contributions are not expected, but feel free to open an issue if you spot something.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
