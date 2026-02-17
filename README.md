# ML Frameworks Lab 2 – Deep Learning Pipeline with PyTorch and DVC

## Overview

This project implements a modular Deep Learning pipeline using PyTorch.
It includes:

- Custom Dataset implementation
- Neural network defined with `nn.Module`
- Modular training and evaluation logic
- GPU acceleration (CUDA)
- Dataset versioning with DVC
- Multiple controlled experiments

The pipeline is fully reproducible and runs from `main.py`.

---

## Project Structure

```
ML-LAB2/
│
├── data/                # Dataset (tracked by DVC, not Git)
├── experiments/         # Experiment results
├── src/
│   ├── dataset.py       # Custom Dataset class
│   ├── model.py         # Neural network definition
│   ├── train.py         # Training loop
│   ├── evaluate.py      # Evaluation logic
│   └── main.py          # Entry point
│
├── data.dvc             # DVC pointer file
├── pyproject.toml       # Project dependencies
├── uv.lock              # Locked dependency versions
└── README.md
```

---

## Environment Setup

This project uses **uv** for dependency management.

Install dependencies:

```
uv sync
```

---

## Running the Pipeline

To train and evaluate the model with default hyperparameters:

```
uv run python src/main.py
```

The script automatically:

- Detects GPU (if available)
- Trains the model
- Evaluates test accuracy
- Logs results to `experiments/results.csv`

---

## Running with Custom Hyperparameters

The training script supports configurable hyperparameters via command-line arguments.

Example:

```
uv run python src/main.py --experiment exp1 --lr 0.001 --batch_size 64 --epochs 5
```

Available arguments:

- `--experiment` → Name of the experiment (used for logging)
- `--lr` → Learning rate
- `--batch_size` → Batch size
- `--epochs` → Number of training epochs

### Example Experiments

Baseline configuration:

```
uv run python src/main.py --experiment exp1 --lr 0.001 --batch_size 64 --epochs 5
```

Lower learning rate:

```
uv run python src/main.py --experiment exp2 --lr 0.0005 --batch_size 64 --epochs 5
```

Larger batch size and more epochs:

```
uv run python src/main.py --experiment exp3 --lr 0.001 --batch_size 128 --epochs 10
```

Results are automatically appended to:

```
experiments/results.csv
```

---

## GPU Support

If CUDA is available, training runs on GPU.
Otherwise, the script automatically falls back to CPU.

Example output:

```
Using GPU: NVIDIA GeForce RTX 3070
```

---

## Dataset Management (DVC)

The CIFAR-10 dataset is tracked using **DVC**.

The raw dataset is NOT committed to Git.
Instead, Git tracks the small `data.dvc` file, which references the dataset.

To reproduce data:

```
uv run dvc pull
```

If no DVC remote is configured, the dataset will automatically download when running the training script.

---

## Experiments

Three experiments were conducted using different hyperparameters.

| Experiment | Learning Rate | Batch Size | Epochs | Accuracy |
|------------|--------------|------------|--------|----------|
| exp1       | 0.001        | 64         | 5      | 0.7228   |
| exp2       | 0.0005       | 64         | 5      | 0.6940   |
| exp3       | 0.001        | 128        | 10     | 0.7281   |

### Observations

- Lowering the learning rate to 0.0005 reduced performance.
- Increasing batch size and training for more epochs slightly improved accuracy.
- The baseline configuration already performed competitively.

---

## Reproducibility

The entire training and evaluation workflow starts from:

```
src/main.py
```

All dependencies are locked via `uv.lock`, and the dataset is versioned with DVC, ensuring reproducible results.

---

## How to Clone and Run from Scratch

Follow these steps to fully reproduce the project on a new machine.

### 1. Clone the repository

```
git clone https://github.com/EdvinGlawing/ML-LAB2.git
cd ML-LAB2
```

---

### 2. Install dependencies using uv

Make sure `uv` is installed.

Then run:

```
uv sync
```

This installs all dependencies exactly as specified in `uv.lock`.

---

### 3. Pull the dataset using DVC

Since the dataset is tracked by DVC (not Git), you must pull it separately:

```
uv run dvc pull
```

If the dataset is not stored in a DVC remote, it will automatically download when you run the training script.

---

### 4. Run the training pipeline

```
uv run python src/main.py
```

The script will:

- Detect GPU (if available)
- Train the model
- Evaluate accuracy
- Log results to `experiments/results.csv`

---

### Expected Output

Example:

```
Using GPU: NVIDIA GeForce RTX 3070
Epoch 1/5, Loss: ...
...
Accuracy: 0.72
Final Accuracy: 0.72
```

The experiment results will be stored in:

```
experiments/results.csv
```

---

The project is fully reproducible from scratch using uv and DVC.