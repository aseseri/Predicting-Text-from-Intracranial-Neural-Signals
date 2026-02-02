# Predicting Text from Intracranial Neural Signals

## Project Overview

This project explores the decoding of speech from intracranial neural recordings using data from the Brain-to-Text 2024 benchmark. The primary objective was to improve the Phoneme Error Rate (PER) of a baseline Gated Recurrent Unit (GRU) model.

By methodically optimizing the baseline architecture through optimizer refinements, regularization techniques, and data augmentation, I achieved a **Validation PER of 18.37%**, representing a **16.4% relative improvement** over the baseline model (21.98%).

A full detailed report of the methods, experiments, and results is available in the `docs/` folder:
**[Read the Final Report (PDF)](https://github.com/aseseri/Predicting-Text-from-Intracranial-Neural-Signals/blob/main/docs/Seseri_Final_Report.pdf)**

## Key Improvements

This repository builds upon the baseline `neural_seq_decoder` by implementing the following improvements:

* **Optimizer & Scheduling:** Replaced the standard Adam optimizer with **AdamW** and implemented **OneCycleLR** scheduling to decouple weight decay and improve convergence stability.
* **Architecture Modifications:** Integrated **Layer Normalization** into the GRU decoder to stabilize hidden states during training.
* **Data Augmentation:** Implemented **Time Masking** (randomly masking contiguous temporal chunks of the neural signal) to improve model robustness against signal gaps.
* **Regularization Tuning:** Optimized hyperparameters including increased dropout (0.5), white noise injection (1.0 SD), and batch size (128) to mitigate overfitting.
* **Transformer Experimentation:** Implemented a Transformer architecture with positional encodings for comparison.

## Results

The table below summarizes the performance of the baseline model versus the optimized models developed in this project.

| Architecture | Validation PER | Improvement vs Baseline | Training Time |
| --- | --- | --- | --- |
| **Baseline GRU** | 21.98% | - | 67 min |
| **Optimized GRU (Final Model)** | **18.37%** | **+16.4%** | 133 min |
| Transformer (8 Layers) | 21.22% | +3.5% | 17 min |

## Repository Structure

The codebase is organized as follows:

* **`src/neural_decoder/`**
* `model.py`: Defines the `GRUDecoder` (updated with LayerNorm) and the experimental `TransformerDecoder` (with Positional Encodings).
* `augmentations.py`: Custom PyTorch modules for `TimeMasking`, `FeatureMasking`, and `GaussianSmoothing`.
* `neural_decoder_trainer.py`: The main training loop, updated to support AdamW, OneCycleLR, and checkpoint resumption.


* **`scripts/`**
* `train_model.py`: The entry point for running experiments. Contains the hyperparameter dictionary configuration.


* **`notebooks/`**
* `formatCompetitionData.ipynb`: Utility to format the raw Dryad dataset for training.


* **`docs/`**
* `Seseri_Final_Report.pdf`: Detailed project write-up.



## Installation

This project utilizes `uv` for dependency management and Python 3.9.

1. **Clone the repository:**
```bash
git clone https://github.com/aseseri/Predicting-Text-from-Intracranial-Neural-Signals.git
cd neural_seq_decoder

```


2. **Create virtual environment and install dependencies:**
```bash
uv venv -p 3.9
source .venv/bin/activate
uv pip install -e .

```



## Data Setup

The dataset is not included in this repository due to size constraints.

1. Download the Brain-to-Text '24 benchmark dataset from [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq).
2. Place the downloaded files into a folder named `data/` in the project root.
3. Format the data using the provided notebook:
```bash
uv pip install ipykernel
# Open and run notebooks/formatCompetitionData.ipynb

```



## Usage

To train the model using the optimized configuration (Experiment 14 settings):

```bash
python ./scripts/train_model.py

```

*Note: You can modify hyperparameters directly in `scripts/train_model.py`.*

## Credits

This project is based on the [Neural Sequence Decoder](https://github.com/cffan/neural_seq_decoder) baseline provided by the Brain-to-Text Benchmark '24.
