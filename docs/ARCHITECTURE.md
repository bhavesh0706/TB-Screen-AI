# System Architecture

## Components

1. Streamlit Patient Interface
2. FastAPI Backend
3. AI Inference Engine
4. Report Generator
5. Future Admin Dashboard

## High-Level Workflow

Patient
   │
   ▼
Chest X-ray Upload
   │
   ▼
Streamlit Interface
   │
   ▼
FastAPI Backend
   │
   ▼
AI Pipeline
 ├── U-Net Segmentation
 ├── DenseNet121 Classification
 ├── YOLOv8 Localization
 └── Grad-CAM
   │
   ▼
Report Generator
   │
   ▼
Final Screening Report