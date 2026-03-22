# aigul_CS515
Repository for CS515 - Deep Learning Course

## Repository Structure

```
aigul_CS515/
├── corecode/
│   ├── HW1/          # Homework 1: MNIST classification with MLP
│   ├── HW2/          # Homework 2: 
│   └── ...
├── results/
│   ├── checkpoints/  # Saved model weights and training history
│   ├── logs/         # Experiment logs and config snapshots
│   ├── plots/        # Generated visualizations
│   └── summaries/    # Aggregated experiment summaries
├── requirements.txt
└── README.md
```

Each homework lives in its own self-contained folder under `corecode/`.
Results are stored at the repository root under `results/`, organized by homework (e.g., `results/checkpoints/HW1/`).

## Usage

Run scripts from within the homework folder:

```bash
cd corecode/HW1
python main.py --mode both --experiment_name baseline_relu --hidden_sizes 256 128
python run_ablation_suite.py --device cpu
```

## Requirements

```bash
pip install -r requirements.txt
```
