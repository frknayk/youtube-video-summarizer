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

## Training Guide

1. Data Preparation:

- Organize your dataset into appropriate directories.
- Implement data augmentation and transformation if needed in `dataset_loader.py`

2. Configuration:

- Modify the parameters in config.yaml to specify your model, data paths, batch size, and other hyperparameters.

3. Training:

- Use the `train.py` script to start training:

```bash
python package_name/train.py --config base_config.yaml
```

- Running all experiment configs together: Put all configuration files under `configs/` folder then call `run_all_experiments()` function from `train.py` script.

```bash
python package_name/train.py --run_all True
```

- Or you can directly run the training script in the root path after modifying the config files inside the script:

```bash
python train_all.py
```

4. Explore W&B UI to track and compare experiments

```bash
https://wandb.ai/YourWandbUserName/PROJECT_NAME
```

5. Evaluation:

- Evaluate your trained model on a separate test dataset using the `evaluate.py` script.

## Dataset

The project relies on a dataset containing histology images. These images are organized into four main classes:

```text
IF classification insert classes here
```

The dataset is divided into `training`, `validation`, and `test` sets to facilitate model training and evaluation.
Each class is represented by a dedicated folder containing DATA_SOURCE


## Features

- Organized directory structure for datasets, models, scripts, and more.
- Configuration using YAML files for datasets and hyperparameters.
- W&B integration for experiment tracking.
- Seamless model architecture selection based on YAML configuration.
- Conda environment setup for consistent dependencies.

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