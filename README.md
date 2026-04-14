# Dim3: A 3D Object Understanding Machine

**Course:** 15-288 Spring 2026
**Project:** Query Answering Machine (QuAM) for 3D Meshes

## Overview
Dim3 is a machine learning project designed to translate raw 3D geometry into meaningful semantic and geometric insights. By serving as a preprocessing layer, Dim3 aims to bridge the gap between 3D spatial data and natural language understanding.

This project is being developed in multiple iterations, starting from classical machine learning on handcrafted geometric features extracted via Blender, and scaling up to deep learning architectures.

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
