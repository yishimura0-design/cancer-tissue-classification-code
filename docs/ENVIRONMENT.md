# Software environment and reproducibility

The original July 2026 BF/ED and Angle Classical audit records reported the following reproducibility environment:

| Package | Recorded version |
| --- | --- |
| Python | 3.9.21 |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| scikit-learn | 1.6.1 |
| scipy | 1.13.1 |
| scikit-image | 0.24.0 |
| Pillow | 11.3.0 |
| LightGBM | 4.6.0 |
| joblib | 1.5.2 |
| tifffile | 2024.8.30 |
| imageio | 2.37.2 |
| matplotlib | 3.9.4 |
| PyTorch (from shared environment) | 2.8.0+cu126 |
| torchvision (from shared environment) | 0.23.0+cu126 |

For **classical** CPU runs, use `requirements-classical.txt` and reproduce the available CPU library versions. CUDA is not required.

For **CNN** runs, set up a version-compatible PyTorch / torchvision installation matching the host GPU driver and CUDA. The environment listed above is an audit snapshot, **not evidence** that every later B0/A1 or full-frame CNN job used an identical set of libraries. Confirm the runtime metadata of those experiments before claiming an exact environment lock.

Record `python --version`, `pip freeze`, Git commit SHA, deterministic seeds and actual GPU name with any manuscript reproduction attempt. Bitwise identical results are not guaranteed across hardware or execution frameworks.

No datasets, credentials, absolute user home directories, or full institutional system logs are bundled.
