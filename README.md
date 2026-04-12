# aigul_CS515_res

Repository for CS515 Deep Learning coursework experiments, implementations, generated results, and report sources.

This repository currently contains completed work for:

- `HW1`: MNIST classification with MLP ablations
- `HW2`: CIFAR-10 transfer learning and knowledge distillation
- `HW3`: CIFAR-10 robustness with AugMix, PGD evaluation, and distillation from a robust teacher

## Repository Structure

```text
aigul_CS515_res/
├── corecode/
│   ├── HW1/                  # Homework 1 code and analysis scripts
│   ├── HW2/                  # Homework 2 code and experiment runners
│   ├── HW3/                  # Homework 3 code and experiment runners
│   └── models/               # Shared model definitions
├── data/
│   ├── MNIST/                # MNIST dataset files
│   └── cifar-10-batches-py/  # CIFAR-10 python-format dataset files
├── results/
│   ├── checkpoints/          # Saved models, histories, and JSON test outputs
│   ├── logs/                 # Per-run logs and config snapshots
│   ├── plots/                # Generated visualizations
│   └── summaries/            # Aggregated summaries where available
├── requirements.txt
├── README.md
└── overview.ipynb
```

## Homework Overview

### HW1

HW1 studies MNIST classification with fully connected networks. The code includes:

- baseline training
- ablation suites over architecture, activation, dropout, regularization, and training hyperparameters
- comparison scripts
- plotting and t-SNE analysis

Main entry points:

- [corecode/HW1/main.py](C:/Users/İhsan/Desktop/Development/cs515/aigul_CS515_res/corecode/HW1/main.py)
- [corecode/HW1/run_ablation_suite.py](C:/Users/İhsan/Desktop/Development/cs515/aigul_CS515_res/corecode/HW1/run_ablation_suite.py)
- [corecode/HW1/compare_experiments.py](C:/Users/İhsan/Desktop/Development/cs515/aigul_CS515_res/corecode/HW1/compare_experiments.py)

### HW2

HW2 studies CIFAR-10 classification with:

- transfer learning using ResNet-18
- scratch-trained ResNet-18 teachers
- knowledge distillation into SimpleCNN and MobileNetV2 students
- result comparison and complexity analysis

Main entry points:

- [corecode/HW2/main.py](C:/Users/İhsan/Desktop/Development/cs515/aigul_CS515_res/corecode/HW2/main.py)
- [corecode/HW2/run_experiments.py](C:/Users/İhsan/Desktop/Development/cs515/aigul_CS515_res/corecode/HW2/run_experiments.py)
- [corecode/HW2/compare_results.py](C:/Users/İhsan/Desktop/Development/cs515/aigul_CS515_res/corecode/HW2/compare_results.py)

### HW3

HW3 extends CIFAR-10 experiments into robustness. It includes:

- AugMix teacher training with JSD consistency loss
- CIFAR-10-C corruption robustness evaluation
- PGD adversarial evaluation under `L_inf` and `L2`
- Grad-CAM and adversarial t-SNE analysis
- distillation from the AugMix teacher into SimpleCNN and MobileNetV2 students
- black-box transferability evaluation

Main entry points:

- [corecode/HW3/main.py](C:/Users/İhsan/Desktop/Development/cs515/aigul_CS515_res/corecode/HW3/main.py)
- [corecode/HW3/run_experiments.py](C:/Users/İhsan/Desktop/Development/cs515/aigul_CS515_res/corecode/HW3/run_experiments.py)
- [corecode/HW3/compare_results.py](C:/Users/İhsan/Desktop/Development/cs515/aigul_CS515_res/corecode/HW3/compare_results.py)

## Results Layout

Results are organized by homework under `results/`:

- `results/checkpoints/HW1`
- `results/checkpoints/HW2`
- `results/checkpoints/HW3`
- `results/logs/HW1`, `results/logs/HW2`, `results/logs/HW3`
- `results/plots/HW1`, `results/plots/HW2`, `results/plots/HW3`

Typical contents of an experiment directory include:

- `*_best_model.pt`
- `*_history.json`
- `*_test_results.json`
- task-specific robustness JSON files for HW3, such as:
  - `*_test_results_corrupted.json`
  - `*_test_results_adversarial_linf.json`
  - `*_test_results_adversarial_l2.json`
  - `*_test_results_transferability.json`

## Data Requirements

The repository already contains:

- MNIST
- CIFAR-10 in python-batch format

For HW3 corruption evaluation, `CIFAR-10-C` is also required, but it is not currently bundled under [data](C:/Users/İhsan/Desktop/Development/cs515/aigul_CS515_res/data). To run the `corrupted` evaluation mode, place the extracted dataset at:

`data/CIFAR-10-C`

The folder should contain files such as:

- `labels.npy`
- `gaussian_noise.npy`
- `motion_blur.npy`
- `jpeg_compression.npy`

Official source:

- [CIFAR-10-C on Zenodo](https://zenodo.org/records/2535967)

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

Run commands from the repository root or by changing into the corresponding homework folder.

