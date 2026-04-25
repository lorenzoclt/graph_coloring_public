# Graph Coloring via Diffusion-Based Message Passing

A graph neural network that solves the q-coloring problem using a diffusion-denoising approach. Given a graph, the model iteratively denoises random node features into a valid coloring — an assignment of q colors such that no two adjacent nodes share the same color.

## Approach

**Training.** Graphs with known (planted) colorings are corrupted with Gaussian noise at a random strength α. The network learns to recover the original coloring by minimizing:

$$\mathcal{L} = E_{\text{Potts}} - \gamma \cdot \text{overlap} + \lambda \cdot H$$

where E_Potts is the continuous Potts energy (fraction of same-color neighbors), overlap measures agreement with the planted coloring, and H is an entropy term that encourages exploration in early epochs.

**Inference.** Starting from random node features, the model alternates between adding noise (forward diffusion) and denoising (reverse step) over a schedule of increasing α values. The process stops early if a zero-energy (valid) coloring is found.

**Architecture.** A Multiscale Message Passing Network (MMPN) with 5 layers. Each layer computes edge messages (φ), aggregates them via mean pooling, and updates node features (γ). Intermediate features from all layers are concatenated and passed through a final MLP, followed by temperature-scaled softmax.

## Repository Structure

| File | Description |
|---|---|
| `GNN.py` | Model definition (MMPN_denoiser for training, MMPN_torch for inference) |
| `training.py` | Training loop using PyTorch Lightning |
| `hparams.py` | Hyperparameter grid |
| `dataset.py` | Dataset and dataloader classes |
| `create_dataset.py` | Graph generator in Python (planted and random graphs, 1k–100k nodes) |
| `create_graphs.c` | Graph generator in C (planted graphs, N=10000, faster) |
| `create_python_dataset.py` | Converts C-generated graphs to PyTorch Geometric format |
| `computeoverlap.c` | C library for computing overlap with planted coloring (via all q! permutations) |
| `denoising.py` | Inference: iterative denoising with energy tracking |
| `checking_training.py` | Evaluation: benchmarks trained models across α values |
| `functions.py` | Utilities (energy computation, autocorrelation, checkpoint loading) |
| `plot.py` | Results parsing and visualization |

## Usage

### Generate data

```bash
# Option A: Python (flexible, supports multiple sizes)
python create_dataset.py

# Option B: C (fast, N=10000 only)
gcc -O2 -o create_graphs create_graphs.c -lm
./create_graphs
python create_python_dataset.py
```

### Compile overlap library

```bash
gcc -shared -O2 -fPIC -o computeoverlap.so computeoverlap.c -lm
```

### Train

```bash
python training.py
```

Training logs are saved to TensorBoard. Checkpoints are saved to `training_log/`.

### Run inference

```bash
python denoising.py
```

Results are logged to `Logs/`.

### Run demo

```bash
python demo.py

```


## Dependencies

- Python 3.8+
- PyTorch
- PyTorch Geometric
- PyTorch Lightning
- torch-scatter
- OR-Tools (for `create_dataset.py` only)
