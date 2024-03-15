# PROJECT_NAME

## Overview

The PROJECT_NAME Project is a machine learning project designed to PURPOSE_OF_PROJECT.
In this project, I leverage the power of machine learning to assist PROBLEM_DOMAIN/DEFINITON

## Project Goals

The primary goals of this project are as follows:

```text
1. Base goal of the ML Model: 

2. How this ML model solved problem: Enable accurate and efficient diagnosis by automating the classification of histological samples. 
This can assist pathologists and medical professionals in identifying abnormal or cancerous tissues.

3. Data Augmentation and Preprocessing: Implement data augmentation techniques and preprocessing methods to improve model generalization and robustness. 
This involves techniques like random rotations, flips, and resizing of images.

4. Transfer Learning: Utilize transfer learning with pre-trained deep learning models, such as ResNet, to leverage their feature extraction capabilities and fine-tune them for histology classification.

5. Performance Metrics: Employ various performance metrics, including accuracy, precision, recall, F1 score, ROC curves, and precision-recall curves, to evaluate and measure the model's effectiveness.

6. Experiment Tracking: Monitor and visualize model training and performance using tools like Weights and Biases (wandb) to gain insights into the training process and make informed decisions.
```

## Installation

`Prerequisites`: `Conda` (Miniconda or Anaconda) installed.

Follow these steps to set up your project environment using Conda:

1. Clone this repository:

```bash
git clone https://github.com/frknayk/REPOSITORY_NAME.git
cd REPOSITORY_NAME
```

2. Install project as python-package, necessary for path management

```bash
pip install -e .
```

3. Create a Conda environment and install the required dependencies:

```bash
conda env create -f environment.yml
conda activate env_PROJECT
```

## Weights and Biases (wandb) Setup

1. Install wandb:

```bash
pip install wandb
```

2. Log in to wandb:

```bash
wandb login
```

Follow the prompts to authorize wandb.

## Simple Usage Guide

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

1. Prepare your dataset and organize it as needed.

2. Modify the configuration parameters in config.yaml to customize your experiment settings.

3. Run the training script:

```bash
python train.py --config config.yaml
```

4. Monitor and visualize your experiment in the wandb dashboard:

```text
Access the wandb dashboard at https://wandb.ai/.
Explore training metrics, logs, and visualizations in real-time.
```

## Directory Structure

- data/: Dataset files or links to datasets.
- models/: Model architectures and related utilities.
- configs/: Configuration YAML files for datasets.
- logs/ : Consists wandb/ and checkpoints/ folder belonging to experiment runs.
- environment.yaml: Conda environment specification.
- requirements.txt: Additional Python package requirements.
- README.md: Project overview and setup instructions.


## TODO-List

- TODO_1
- TODO_2