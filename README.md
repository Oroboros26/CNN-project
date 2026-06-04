# CNN Project

A polished pneumonia detection project using chest X-ray images, deep learning, and a deployable inference interface.

This repository combines exploratory model development, transfer learning, and a demo app to showcase practical medical imaging classification.

## Table of Contents

- [Highlights](#highlights)
- [Repository Structure](#repository-structure)
- [Data Sources](#data-sources)
- [Model Workflow](#model-workflow)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Running the Notebook](#running-the-notebook)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Running CLI Inference](#running-cli-inference)
- [Local Dataset Support](#local-dataset-support)
- [Output Artifacts](#output-artifacts)
- [Notes](#notes)
- [Future Enhancements](#future-enhancements)

## Highlights

- End-to-end pneumonia detection pipeline for chest X-rays
- Dual model experiments: custom CNN and MobileNetV2 transfer learning
- Dataset download and preprocessing with `kagglehub`
- Streamlit demo app for image upload and prediction
- Lightweight CLI tool for single-image inference
- Local dataset validation helper in `data/`

## Repository Structure

- `CNN_Project.ipynb` — core notebook for data loading, model building, training, evaluation, and saving.
- `colab_kernel_launcher.py` — Streamlit app for interactive inference.
- `requirements.txt` — reproducible Python dependency list.
- `predict_cli.py` — CLI utility for image-based predictions.
- `data/` — local dataset support folder with documentation and validation.
- `pneumonia_detection_model_transfer.h5` — saved transfer learning model for deployment.
- `README.md` — project documentation.

## Data Sources

The notebook uses two Kaggle datasets:

- `mitgandhi10/dataset-for-cnn` — initial small dataset experiment.
- `paultimothymooney/chest-xray-pneumonia` — larger dataset used for transfer learning and improved generalization.

## Model Workflow

The main notebook follows a clear pipeline:

1. Download and prepare the dataset using `kagglehub`.
2. Create labeled image metadata for `NORMAL` and `PNEUMONIA` categories.
3. Build image generators with augmentation for training.
4. Train a custom CNN architecture from scratch.
5. Evaluate the first model on test data.
6. Construct a transfer learning model using `MobileNetV2`.
7. Train and validate the transfer learning model.
8. Evaluate the transfer model on unseen test samples.
9. Save the final model for inference.

## Model Architecture

### Custom CNN

- Input size: `224x224x3`
- `Conv2D` + `MaxPooling2D` feature extraction blocks
- `Dropout` for regularization
- Dense classification head with `sigmoid` activation

### Transfer Learning

- Base network: `MobileNetV2` pretrained on ImageNet
- Frozen base layers for efficient transfer learning
- Custom head with `GlobalAveragePooling2D`, `Dense`, and `Dropout`
- Binary pneumonia classification output

## Installation

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Or install packages manually:

```bash
pip install tensorflow pandas numpy matplotlib kagglehub streamlit pillow
```

## Running the Notebook

1. Open `CNN_Project.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
2. Execute the cells sequentially.
3. Confirm dataset download and preprocessing succeed.
4. Review the training, evaluation, and visualization results.

## Running the Streamlit App

Launch the demo app from the repository root:

```bash
streamlit run colab_kernel_launcher.py
```

The Streamlit app allows image upload and returns a pneumonia prediction using the saved transfer model.

## Running CLI Inference

Use the CLI utility to predict a single chest X-ray image:

```bash
python predict_cli.py --image path/to/xray.jpg
```

By default, the script loads `pneumonia_detection_model_transfer.h5`.

## Local Dataset Support

A `data/` directory is included to support local dataset management.

Recommended layout:

```text
data/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

Use `data/dataset_setup.py` to verify dataset structure.

## Output Artifacts

- `pneumonia_detection_model_transfer.h5` — final transfer learning model for inference.
- `predict_cli.py` — CLI inference script.
- `colab_kernel_launcher.py` — Streamlit demo app.

## Notes

- This project is for demonstration and research purposes only.
- The Streamlit app should not be used as a medical diagnostic tool.
- Kaggle dataset downloads require internet access and valid credentials.
- Transfer learning training may require significant compute resources.

## Future Enhancements

- Add evaluation metrics: confusion matrix, precision, recall, and F1-score.
- Add a dedicated `requirements.txt` for reproducible setup.
- Add local dataset setup scripts to streamline workflow.
- Add a lightweight CLI or web deployment script for production use.
