# TB-Screen AI

AI-assisted Tuberculosis Screening and Report Generation System.

## Overview

TB-Screen AI is a deep learning-based screening platform designed to assist healthcare workers in identifying potential tuberculosis cases from chest X-ray images. The system combines multiple AI models with explainability and automated report generation for district-level deployment.

## Phase 1

- Public chest X-ray datasets
- DenseNet121 classification model
- U-Net lung segmentation
- YOLOv8 abnormality localization
- Grad-CAM explainability
- Streamlit screening prototype
- FastAPI backend

## Phase 2

- DTO dataset fine-tuning
- District-specific deployment
- Multi-center validation
- Offline-first deployment
- State-level screening dashboard

## Technology Stack

| Component | Technology |
|-----------|------------|
| AI | TensorFlow |
| Backend | FastAPI |
| Frontend | Streamlit |
| Language | Python 3.11 |
| Environment | Conda |

## Repository Structure

```
TB-Screen-AI/
├── ai-model/
├── backend/
├── streamlit-app/
├── admin-frontend/
├── docs/
├── scripts/
├── requirements.txt
└── README.md
```