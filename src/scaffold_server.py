import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scaffold_client import ScaffoldClient
from utils import save_results_as_json
import time

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _plot_client_convergence(client_id, global_hist, local_hist, plots_dir):
    """2-panel convergence plot: Accuracy and F1 (local vs global)."""
    iters = list(range(1, len(global_hist) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Client {client_id} — SCAFFOLD Convergence (Local vs Global)", fontsize=13)

    for ax, metric, ylabel in zip(
        axes,
        ["accuracy", "f1"],
        ["Accuracy", "F1 Score (macro)"],
    ):
        ax.plot(iters, [m[metric] for m in global_hist], marker="o", label="Global model")
        ax.plot(iters, [m[metric] for m in local_hist],  marker="s", linestyle="--", label="Local model")
        ax.set_xlabel("Global Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, f"scaffold_client_{client_id}_convergence.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _print_comparison_table(client_list, best_global_results, best_local_results):
    """Print a side-by-side table: best global model vs best local model per client."""
    header = (
        f"{'Client':>8} | "
        f"{'G-Acc':>7} {'G-F1':>7} {'G-Prec':>8} {'G-Rec':>7} {'G-AUC':>7} | "
        f"{'L-Acc':>7} {'L-F1':>7} {'L-Prec':>8} {'L-Rec':>7} {'L-AUC':>7}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print("  FINAL: Best Global Model vs Best Local Model per Client")
    print(sep)
    print(header)
    print(sep)

    for c in client_list:
        cid = c.client_id
        g = best_global_results[cid]
        l = best_local_results[cid]
        print(
            f"{cid:>8} | "
            f"{g['accuracy']:>7.4f} {g['f1']:>7.4f} {g['precision']:>8.4f} {g['recall']:>7.4f} {g['auc_roc']:>7.4f} | "
            f"{l['accuracy']:>7.4f} {l['f1']:>7.4f} {l['precision']:>8.4f} {l['recall']:>7.4f} {l['auc_roc']:>7.4f}"
        )

    def _mean(d, k):
        return sum(v[k] for v in d.values()) / len(d)

    print(sep)
    print(
        f"{'AVG':>8} | "
        f"{_mean(best_global_results,'accuracy'):>7.4f} {_mean(best_global_results,'f1'):>7.4f} "
        f"{_mean(best_global_results,'precision'):>8.4f} {_mean(best_global_results,'recall'):>7.4f} {_mean(best_global_results,'auc_roc'):>7.4f} | "
        f"{_mean(best_local_results,'accuracy'):>7.4f} {_mean(best_local_results,'f1'):>7.4f} "
        f"{_mean(best_local_results,'precision'):>8.4f} {_mean(best_local_results,'recall'):>7.4f} {_mean(best_local_results,'auc_roc'):>7.4f}"
    )
    print(sep)


# ------------------------------------------------------------------ #
# Server
# ------------------------------------------------------------------ #

def scaffold_server(args, model, device, domains_path, client_distributions,
                    max_client_participants, project_root):
    """
    SCAFFOLD server — single time step.

    Algorithm (Karimireddy et al., 2020):
      Initialise global model x, global control variate c = 0.

      Each global iteration:
        1. Broadcast (x, c) to all clients.
        2. Each client:
             a. Initialises local model from x.
             b. Runs K corrected SGD steps:
                    g_corrected = g_local - c_i + c
             c. Computes:
                    model_delta   = x_local - x
                    control_delta = (x - x_local) / (K * lr)
                    c_i_new       = c_i + control_delta
             d. Returns (model_delta, control_delta).
        3. Server aggregates:
                x   ← x + (1/N) * Σ model_delta_i
                c   ← c + (1/N) * Σ control_delta_i
        4. Evaluate updated global model across all clients.

      Per iteration:
        • Train each client → evaluate local model → save best local model.
        • Aggregate → evaluate global model across all clients → save best global.

      After all iterations:
        • Per-client convergence plots.
        • Load & evaluate best global model on all clients.
        • Load & evaluate best local models on all clients.
        • Side-by-side comparison table.
    """
    print("\n--- Starting SCAFFOLD ---")

    base_save_dir = os.path.join(project_root, args.save_dir, args.exp_name, args.algorithm)
    models_dir = os.path.join(base_save_dir, "models")
    plots_dir  = os.path.join(base_save_dir, "plots")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir,  exist_ok=True)

    time_step = 0  # single time step

    # ── initialise clients ──────────────────────────────────────────
    client_list = [
        ScaffoldClient(
            client_id=i,
            args=args,
            domain_path=domains_path,
            assigned_domains=client_distributions[i],
            device=device,
            model=model,
        )
        for i in range(max_client_participants)
    ]
    print(f"Initialized {len(client_list)} SCAFFOLD clients.")

    # ── global control variate c (zero-init, same shape as model params) ──
    server_control = {
        k: torch.zeros_like(v, device=device)
        for k, v in model.named_parameters()
    }

    # ── metric history ──────────────────────────────────────────────
    per_iter_global = {}
    per_iter_local  = {}
    client_global_hist = {c.client_id: [] for c in client_list}
    client_local_hist  = {c.client_id: [] for c in client_list}

    best_global_acc = -1.0
    best_local_acc  = {c.client_id: -1.0 for c in client_list}

    round_times = []

    # ================================================================ #
    # Global iteration loop
    # ================================================================ #
    for iteration in range(args.global_iters):
        round_start = time.perf_counter()
        print(f"\n{'='*55}")
        print(f"  SCAFFOLD — Global Iteration {iteration + 1} / {args.global_iters}")
        print(f"{'='*55}")

        global_state = copy.deepcopy(model.state_dict())

        # ------------------------------------------------------------ #
        # STEP 1 — Local training + immediate local evaluation
        # ------------------------------------------------------------ #
        print(f"\n  [Step 1] Local training & evaluation")
        model_deltas   = []
        control_deltas = []
        per_iter_local[iteration] = {}
        l_totals = dict(loss=0.0, accuracy=0.0, f1=0.0, precision=0.0, recall=0.0, auc_roc=0.0)

        for c in client_list:
            model_delta, control_delta = c.train(
                global_state, server_control, time_step
            )
            model_deltas.append(model_delta)
            control_deltas.append(control_delta)

            # evaluate the just-trained local model
            loss, acc, f1, prec, rec, auc_roc = c.evaluate_local_model_full(time_step)
            row = dict(loss=loss, accuracy=acc, f1=f1, precision=prec, recall=rec, auc_roc=auc_roc)
            per_iter_local[iteration][c.client_id] = row
            client_local_hist[c.client_id].append(row)
            for k in l_totals:
                l_totals[k] += row[k]

            # save best local model per client
            if acc > best_local_acc[c.client_id]:
                best_local_acc[c.client_id] = acc
                local_path = os.path.join(
                    models_dir, f"best_local_model_client_{c.client_id}.pth"
                )
                torch.save(c.local_model.state_dict(), local_path)
                print(f"  --> [Saved] Best local model client {c.client_id} (acc={acc:.4f})")

        avg_l = {k: v / len(client_list) for k, v in l_totals.items()}
        print(f"\n  [Avg Local]  Acc={avg_l['accuracy']:.4f}  F1={avg_l['f1']:.4f}  "
              f"Prec={avg_l['precision']:.4f}  Rec={avg_l['recall']:.4f}  Loss={avg_l['loss']:.4f} AUC={avg_l['auc_roc']:.4f}")

        # ------------------------------------------------------------ #
        # STEP 2 — Server aggregation (model + control variate)
        # ------------------------------------------------------------ #
        print(f"\n  [Step 2] Server aggregation & global model evaluation")
        n = len(model_deltas)

        # x ← x + (1/N) * Σ delta_i
        with torch.no_grad():
            new_state = {k: v.clone() for k, v in global_state.items()}
            for key in new_state:
                new_state[key] = new_state[key].to(device) + \
                    sum(d[key].to(device) for d in model_deltas) / n
        model.load_state_dict(new_state)

        # c ← c + (1/N) * Σ control_delta_i
        with torch.no_grad():
            for name in server_control:
                server_control[name] = server_control[name].to(device) + \
                    sum(d[name].to(device) for d in control_deltas) / n

        # ── evaluate global model ───────────────────────────────────
        per_iter_global[iteration] = {}
        g_totals = dict(loss=0.0, accuracy=0.0, f1=0.0, precision=0.0, recall=0.0, auc_roc=0.0)

        for c in client_list:
            loss, acc, f1, prec, rec, auc_roc = c.evaluate_global_model(model.state_dict(), time_step)
            row = dict(loss=loss, accuracy=acc, f1=f1, precision=prec, recall=rec, auc_roc=auc_roc)
            per_iter_global[iteration][c.client_id] = row
            client_global_hist[c.client_id].append(row)
            for k in g_totals:
                g_totals[k] += row[k]

        avg_g = {k: v / len(client_list) for k, v in g_totals.items()}
        print(f"\n  [Avg Global] Acc={avg_g['accuracy']:.4f}  F1={avg_g['f1']:.4f}  "
              f"Prec={avg_g['precision']:.4f}  Rec={avg_g['recall']:.4f}  Loss={avg_g['loss']:.4f} AUC={avg_g['auc_roc']:.4f}")

        # save best global model
        if avg_g["accuracy"] > best_global_acc:
            best_global_acc = avg_g["accuracy"]
            best_global_path = os.path.join(models_dir, "best_global_model.pth")
            torch.save(model.state_dict(), best_global_path)
            print(f"  --> [Saved] Best global model (acc={best_global_acc:.4f})")

        round_end = time.perf_counter() # Stop the timer
        round_duration = round_end - round_start
        round_times.append(round_duration)
        print(f"\n  [Timing] Iteration {iteration + 1} completed in {round_duration:.2f} seconds")
    # ================================================================ #
    # Convergence plots
    # ================================================================ #
    print("\n--- Generating convergence plots ---")
    for c in client_list:
        path = _plot_client_convergence(
            c.client_id,
            client_global_hist[c.client_id],
            client_local_hist[c.client_id],
            plots_dir,
        )
        print(f"  [Plot] Client {c.client_id} -> {path}")

    # ================================================================ #
    # Final evaluation — best global model across ALL clients
    # ================================================================ #
    print("\n--- Final Evaluation: Best Global Model ---")
    best_global_path = os.path.join(models_dir, "best_global_model.pth")
    model.load_state_dict(torch.load(best_global_path, map_location=device))

    best_global_results = {}
    for c in client_list:
        loss, acc, f1, prec, rec, auc_roc = c.evaluate_global_model(model.state_dict(), time_step)
        best_global_results[c.client_id] = dict(loss=loss, accuracy=acc, f1=f1, precision=prec, recall=rec, auc_roc=auc_roc)

    # ================================================================ #
    # Final evaluation — best local model for each client
    # ================================================================ #
    print("\n--- Final Evaluation: Best Local Models ---")
    best_local_results = {}
    for c in client_list:
        local_path = os.path.join(models_dir, f"best_local_model_client_{c.client_id}.pth")
        loss, acc, f1, prec, rec, auc_roc = c.evaluate_model(
            torch.load(local_path, map_location=device), time_step
        )
        best_local_results[c.client_id] = dict(
            loss=loss, accuracy=acc, f1=f1, precision=prec, recall=rec, auc_roc=auc_roc
        )
        print(f"  [Client {c.client_id}] [Best Local] "
              f"Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Loss={loss:.4f}")

    # ================================================================ #
    # Comparison table
    # ================================================================ #
    _print_comparison_table(client_list, best_global_results, best_local_results)

    # ================================================================ #
    # Save results
    # ================================================================ #
    print("\n--- SCAFFOLD Training Complete ---")

    total_time = sum(round_times)
    avg_round_time = total_time / len(round_times) if round_times else 0

    results = {
        "global_metrics_per_iteration": per_iter_global,
        "local_metrics_per_iteration":  per_iter_local,
        "final_best_global_model":      best_global_results,
        "final_best_local_models":      best_local_results,
        "timing_seconds": {
            "per_round": round_times,
            "total_training_time": total_time,
            "average_round_time": avg_round_time
        }
    }

    results_folder = base_save_dir
    filename = f"metrics_scaffold_{args.exp_name}.json"
    save_results_as_json(filename, results, project_root, results_folder)