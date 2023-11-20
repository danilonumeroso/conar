# Neural Algorithmic Reasoning for Combinatorial Optimisation
Official code repository for the paper [Neural Algorithmic Reasoning for Combinatorial Optimisation](https://arxiv.org/pdf/2306.06064.pdf).

## Key files/locations

- `baselines/`: Deterministic baseline algorithms for comparison.
- `datasets/`: Code responsible for handling the datasets
- `layers/`+`models/`: Our model's implementations are in those two locations. We have aimed to keep to the following "rule": If a class is NOT a [pytorch lightning](https://lightning.ai/docs/pytorch/stable/) class, it is responsible for processing a datapoint, but NOT for loss computation, dataloaders, etc. If it is a [pytorch lightning](https://lightning.ai/docs/pytorch/stable/) class, then the code related to loss computation/dataloaders/logging/etc. is likely to be there.
- `prepare_datasets.py`: Script to generate and preprocess the data used for training and testing.
- `serialised_models/`: Pre-trained models for reproducing paper results. These models offer a starting point for further experimentation and research. Due to repository space constraints we provide only the key pre-trained models. We are happy to provide other models on a per-case basis.
  
  Any models that you train will be saved in the directory in the format `best_`+given name. If you do not explicitly provide a name (using `--model-name`), the date+time at the time of the **starting** of the training script is used.

  To avoid mixing provided models with those that you may wish to train, our pre-trained checkpoints are in individual subdirectories.
- `train_reasoner.py`: Main script for training neural algorithmic reasoners.
- `train_tsp.py`: Script for fine-tuning algorithmic reasoners on TSP problems.
- `train_vkc.py`: Script for fine-tuning algorithmic reasoners on VKC problems.
- `test_*.py`: Scripts for testing models. (`*` denotes a wildmark, there is no file `test_*.py`)

### Data Preparation

**IMPORTANT**: Use **only** `prepare_datasets.py`  to generate the datasets required for training and evaluating the models. The data may end up different from ours otherwise!

## Getting Started

1. **Install Dependencies**: 

   To start with, make sure your GPU drivers are up-to date.

   We provide two ways to install our dependencies. 

   * Option 1:
     ```bash
     pip install -r requirements.txt
     ```
     
     This will install *most* (see below) of the necessary libraries in the **current** python environment.
     The CLRS-30 repository will be installed from [here](https://pypi.org/project/dm-clrs/). The JAX library may be installed **without** GPU support. PyTorch/PyG will still be GPU-enabled.



   * Option 2:
     ```bash
     conda env create -f gpuenv.yml
     ```

     This will install *most* (see below) of the necessary libraries in a **new environment named `conar`** and JAX will also be GPU-enabled. However, you have to clone CLRS-30 from [its GitHub repository](https://github.com/google-deepmind/clrs) and edit `gpuenv.yml` (line 33).

   Regardless of which option you choose, you have to *manually*:
   * Install `pyconcorde`. See [here](https://github.com/jvkersch/pyconcorde).
   * Install Gurobi. See [here](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer-).
   * Install LKH. See [here](http://webhotel4.ruc.dk/~keld/research/LKH-3/). Copy the executable to `/usr/local/bin/`.
   

2. **Prepare Data**:
   ```bash
   python prepare_datasets.py
   ```
   To ensure that generated data are the same as in our paper, compare with the
   following checksum and datapoint:
   ```bash
   5586df9e7f8bcdf36d7c08f2bce3f23e1cdf45fc  data/tsp_large/num_nodes_16/processed/train/data_1001.pt
   ```
   If this matches, almost certainly the rest will match too.

   **NOTE**: If you use different PyTorch version, the sha may differ, but the data may still be the same.


3. **Training Models**:
   - For the neural algorithmic reasoner:
     ```bash
     python train_reasoner.py [OPTIONS]
     ```
   - For TSP solver:
     ```bash
     python train_tsp.py [OPTIONS]
     ```
   - For VKC solver:
     ```bash
     python train_vkc.py [OPTIONS]
     ```

4. **Evaluating Models**:
   - `test_co.py` can be used for evaluating model performance on both TSP and VKC tasks.
     
---

For more detailed instructions and documentation, refer to the individual script files and comments within the code. `[OPTIONS]` for each script can be viewed in the beginning of each file or by calling the script with the `--help` command.
