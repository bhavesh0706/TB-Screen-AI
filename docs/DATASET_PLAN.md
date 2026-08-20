# Dataset Plan

## Phase 1 Datasets

| Dataset | Purpose |
|----------|---------|
| NIH ChestX-ray14 | Classification |
| Shenzhen TB | Tuberculosis |
| Montgomery TB | Tuberculosis |
| VinDr-CXR | Localization |

## Planned Split

- Train: 70%
- Validation: 15%
- Test: 15%

## Preprocessing

- Resize to 224×224
- CLAHE enhancement
- Normalization
- Data augmentation