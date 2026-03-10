# Prism Server Experiments

This directory contains the Python-based server-side framework for the Prism project. It includes all the necessary scripts, data loaders, and training logic to simulate and evaluate various fine-tuning configurations (including First-Order and Zeroth-Order approaches) prior to on-device deployment.

## Environment Setup

To ensure reproducibility and avoid dependency conflicts, we recommend creating an isolated Conda environment.

1. **Create and activate the Conda environment:**
   ```bash
   conda create -n prism python=3.10
   conda activate prism
   ```

2. **Install the required dependencies:**
   Navigate to the `server/` directory and install the packages listed in the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

## Reproducing Experiments

We have prepared comprehensive bash scripts to automate the evaluation process. You can reproduce all the server-side experiments presented in our paper with a single command.

Make sure you are in the `server/` directory, and simply run:

```bash
sh ./experiments/run.sh
```

This script will sequentially execute the configured experimental pipelines across different models and evaluation benchmarks.
