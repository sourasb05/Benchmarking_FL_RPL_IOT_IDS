# Benchmarking Federated Learning for RPL-based IoT Intrusion Detection

This repository benchmarks **Federated Learning (FL) algorithms** for binary intrusion detection on RPL-based IoT network traffic. A two-layer **LSTM classifier** is trained in a federated setting across multiple clients, each holding a distinct attack-domain dataset. The benchmark currently implements **FedAvg** and **SCAFFOLD**, with a shared evaluation protocol that reports per-client local and global model performance after every communication round.

---

## Algorithms

| Algorithm | File | Description |
|-----------|------|-------------|
| **FedAvg** | `server.py` / `client.py` | McMahan et al. (2017). Clients train locally with Adam; server averages model weights. |
| **SCAFFOLD** | `scaffold_server.py` / `scaffold_client.py` | Karimireddy et al. (2020). Adds per-client control variates to correct client drift. Correction applied as a post-step parameter nudge (compatible with Adam). |

---

## Repository Structure

```
Benchmarking_FL_RPL_IOT_IDS/
├── src/
│   ├── main.py               # Entry point — parses args, builds model, dispatches algorithm
│   ├── client.py             # FedAvg client (train + evaluate)
│   ├── server.py             # FedAvg server loop
│   ├── scaffold_client.py    # SCAFFOLD client (train with control variate correction)
│   ├── scaffold_server.py    # SCAFFOLD server loop
│   ├── models.py             # LSTMClassifier, LSTMModelWithAttention, CTVAE
│   ├── utils.py              # Data loading, windowing, normalisation, arg parsing
│   └── evaluate_model.py     # Evaluation utilities
├── attack_data/              # (not tracked) RPL IoT domain folders of CSVs
│   ├── <domain_name>/
│   │   ├── ..._1_60_sec.csv
│   │   ├── ..._2_60_sec.csv
│   │   └── ...
│   └── ...
├── saved_models/             # (not tracked) Best model checkpoints
│   ├── fedavg/
│   │   ├── best_global_model.pth
│   │   └── best_local_model_client_<id>.pth
│   └── scaffold/
│       ├── best_global_model.pth
│       └── best_local_model_client_<id>.pth
├── results/                  # (not tracked) Metrics JSON + convergence plots
│   ├── fedavg_metrics.json
│   ├── scaffold_metrics.json
│   └── plots/
│       └── <algo>_client_<id>_convergence.png
├── .gitignore
└── README.md
```

---

## Data Format

- Place all data under `attack_data/<domain_name>/` — each domain folder corresponds to one attack scenario.
- Each CSV must contain a `label` column (`0` = benign, `1` = attack) and numeric feature columns.
- Filenames follow the pattern `..._<index>_60_sec.csv`. The code uses the index to sort files consistently.
- Per domain: first **16 files → train**, last **4 files → test** (file-level split).

**Sliding window preprocessing** (applied per CSV):
- `window_size` consecutive rows → one sample; label = last row's label.
- Per-domain min–max normalisation computed from train files only, then applied to test.
- Windows are reshaped to `(B, window_size, n_raw_features)` — true temporal sequences for the LSTM.

---

## Model

**LSTMClassifier** (`models.py`)
- 2-layer stacked LSTM → ReLU FC head → linear output
- Input: `(batch, window_size, n_raw_features)` — `n_raw_features` is auto-detected from the first CSV at runtime
- Output: raw logits `(batch, num_classes)`

---

## Federated Training Protocol

Each communication round:

1. **Local training** — each client initialises from the current global model and trains for `local_epochs` on its domain data.
2. **Local evaluation** — the trained local model is evaluated immediately on the client's own test data (before aggregation). Best local model per client is saved.
3. **Aggregation** — FedAvg averages model weights; SCAFFOLD additionally aggregates control variate deltas.
4. **Global evaluation** — the aggregated global model is evaluated on every client's test data. Best global model is saved.

**Metrics reported per client per round:** Accuracy, F1 (macro), Precision (macro), Recall (macro), Loss.

**After all rounds:**
- Per-client convergence plots: local vs global Accuracy and F1 across rounds.
- Final evaluation of the best saved global model across all clients.
- Final evaluation of each client's best saved local model.
- Side-by-side comparison table.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install torch numpy pandas scikit-learn matplotlib
```

For GPU support install PyTorch with the appropriate CUDA build from [pytorch.org](https://pytorch.org).

---

## Quick Start

```bash
cd src

# Run FedAvg
python main.py --algorithm fedavg --global_iters 20 --local_epochs 5 --lr 0.001

# Run SCAFFOLD
python main.py --algorithm scaffold --global_iters 20 --local_epochs 5 --lr 0.001
```

---

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--algorithm` | `fedavg` | FL algorithm: `fedavg` or `scaffold` |
| `--hidden_size` | `64` | LSTM hidden dimension |
| `--output_size` | `2` | Number of classes |
| `--window_size` | `10` | Sliding window length |
| `--step_size` | `2` | Stride between windows |
| `--batch_size` | `64` | Mini-batch size |
| `--global_iters` | `20` | Number of FL communication rounds |
| `--local_epochs` | `5` | Local training epochs per round |
| `--lr` | `0.001` | Learning rate (Adam) |
| `--seed` | `42` | Random seed |

> `--input_size` and `--n_raw_features` are **auto-computed** from the data at runtime; do not set them manually.

---

## Outputs

| Output | Location | Description |
|--------|----------|-------------|
| Best global model | `saved_models/<algo>/best_global_model.pth` | Saved when avg accuracy across clients improves |
| Best local models | `saved_models/<algo>/best_local_model_client_<id>.pth` | Per-client, saved when that client's local accuracy improves |
| Metrics JSON | `results/<algo>_metrics.json` | Per-iteration local and global metrics for all clients, plus final comparison |
| Convergence plots | `results/plots/<algo>_client_<id>_convergence.png` | Accuracy and F1 curves: local vs global per client |

---

## SCAFFOLD Implementation Notes

The standard SCAFFOLD correction adds `(c - c_i)` to the gradient before the optimiser step. With Adam this is distorted by the adaptive variance scaling, so the correction is applied as a **post-step parameter nudge** instead:

```
x ← x - lr * (c_i - c_server)
```

Control variate update follows **Option II** from the paper:

```
c_i_new = c_i - c_server + (x_global - x_local) / (K × lr)
```

where `K` is the total number of local steps. The server accumulates the average control variate delta each round.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `FileNotFoundError: .../attack_data` | Create `attack_data/` at the project root and add domain subfolders |
| `'label' column missing` | Each CSV must have a `label` column with integer `0`/`1` values |
| `CUDA not used` | Check PyTorch install and GPU drivers; CPU fallback is automatic |
| Shape mismatch in LSTM | Do not set `--input_size` manually — it is auto-computed from the data |
