import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from ditto_client import DittoClient
from utils import save_results_as_json
import time
import random

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _plot_client_convergence(client_id, global_hist, local_hist, plots_dir):
    """
    2-panel convergence plot per client: Accuracy and F1 (local vs global).
    global_hist / local_hist: list of dicts {accuracy, f1, precision, recall, loss}
    """
    # X-axis for the global model (which updates every round)
    global_iters = list(range(1, len(global_hist) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Client {client_id} — Ditto Convergence (Local vs Global)", fontsize=13)

    for ax, metric, ylabel in zip(
        axes,
        ["accuracy", "f1"],
        ["Accuracy", "F1 Score (macro)"],
    ):
        ax.plot(global_iters, [m[metric] for m in global_hist], marker="o", label="Global model")
        
        if local_hist:
            local_iters = [m.get("iteration", i+1) for i, m in enumerate(local_hist)]
            ax.plot(local_iters, [m[metric] for m in local_hist],  marker="s", linestyle="--", label="Local model")
            
        ax.set_xlabel("Global Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, f"client_{client_id}_convergence.png")
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

def ditto_server(args, model, device, domains_path, client_distributions, max_client_participants, project_root, lam):
    """
    Ditto — single time step.

    """
    print("\n--- Starting Ditto ---")

    base_save_dir = os.path.join(project_root, args.save_dir, args.exp_name, args.algorithm)
    models_dir = os.path.join(base_save_dir, "models")
    plots_dir  = os.path.join(base_save_dir, "plots")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir,  exist_ok=True)

    time_step = 0  # single time step

    # ---- initialise clients ----
    client_list = [
        DittoClient(
            client_id=i,
            args=args,
            domain_path=domains_path,
            assigned_domains=client_distributions[i],
            device=device,
            model=model,
            lam=lam,  
        )
        for i in range(max_client_participants)
    ]
    print(f"Initialized {len(client_list)} clients.")

    # ---- metric history (indexed by iteration) ----
    per_iter_global = {}   # per_iter_global[iter][client_id] = {acc, f1, ...}
    per_iter_local  = {}   # per_iter_local[iter][client_id]  = {acc, f1, ...}

    client_global_hist = {c.client_id: [] for c in client_list}
    client_local_hist  = {c.client_id: [] for c in client_list}

    best_global_acc = -1.0
    best_local_acc  = {c.client_id: -1.0 for c in client_list}

    round_times = []
    # ------------------------------------------------------------------ #
    # Global iteration loop
    # ------------------------------------------------------------------ #
    for iteration in range(args.global_iters):
        round_start = time.perf_counter()
        print(f"\n{'='*55}")
        print(f"  Global Iteration {iteration + 1} / {args.global_iters}")
        print(f"{'='*55}")

        # -------------------------------------------------------------- #
        # STEP 1 — Local training + immediate local evaluation
        # Each client initialises from the current global weights, trains,
        # then is evaluated on its OWN test data BEFORE aggregation.
        # -------------------------------------------------------------- #
        print(f"\n  [Step 1] Local training & evaluation")
        local_states = []
        per_iter_local[iteration] = {}
        l_totals = dict(loss=0.0, accuracy=0.0, f1=0.0, precision=0.0, recall=0.0, auc_roc=0.0)
        selected_clients = random.sample(client_list, max(1, int(len(client_list) * args.client_fraction)))
        for c in selected_clients:
            # train — updates c.local_model, returns its state_dict
            trained_state = c.train(model.state_dict(), time_step=time_step)
            local_states.append(trained_state)

            # evaluate the just-trained local model (c.local_model still holds it)
            loss, acc, f1, prec, rec, auc_roc = c.evaluate_local_model_full(time_step)
            
            row = dict(iteration=iteration+1, loss=loss, accuracy=acc, f1=f1, precision=prec, recall=rec, auc_roc=auc_roc)
            
            per_iter_local[iteration][c.client_id] = row
            client_local_hist[c.client_id].append(row)
            for k in l_totals:
                l_totals[k] += row[k]

            # save best local model per client
            if acc > best_local_acc[c.client_id]:
                best_local_acc[c.client_id] = acc
                local_path = os.path.join(models_dir, f"best_local_model_client_{c.client_id}.pth")
                torch.save(c.personalized_model.state_dict(), local_path)
                print(f"  --> [Saved] Best local model client {c.client_id} (acc={acc:.4f})")

        avg_l = {k: v / len(selected_clients) for k, v in l_totals.items()}
        print(f"\n  [Avg Local]  Acc={avg_l['accuracy']:.4f}  F1={avg_l['f1']:.4f}  "
              f"Prec={avg_l['precision']:.4f}  Rec={avg_l['recall']:.4f}  Loss={avg_l['loss']:.4f} AUC={avg_l['auc_roc']:.4f}")

        # -------------------------------------------------------------- #
        # STEP 2 — Ditto aggregation → global model evaluation
        # -------------------------------------------------------------- #
        print(f"\n  [Step 2] Ditto aggregation & global model evaluation")
        global_state = model.state_dict()
        n = len(local_states)
        for key in global_state:
            global_state[key] = sum(s[key] for s in local_states) / n
        model.load_state_dict(global_state)

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
    # ------------------------------------------------------------------ #
    # Convergence plots — one per client
    # ------------------------------------------------------------------ #
    print("\n--- Generating convergence plots ---")
    for c in client_list:
        path = _plot_client_convergence(
            c.client_id,
            client_global_hist[c.client_id],
            client_local_hist[c.client_id],
            plots_dir,
        )
        print(f"  [Plot] Client {c.client_id} -> {path}")

    # ------------------------------------------------------------------ #
    # Final evaluation — best global model across ALL clients
    # ------------------------------------------------------------------ #
    print("\n--- Final Evaluation: Best Global Model ---")
    best_global_path = os.path.join(models_dir, "best_global_model.pth")
    model.load_state_dict(torch.load(best_global_path, map_location=device))

    best_global_results = {}
    for c in client_list:
        loss, acc, f1, prec, rec, auc_roc = c.evaluate_global_model(model.state_dict(), time_step)
        best_global_results[c.client_id] = dict(loss=loss, accuracy=acc, f1=f1, precision=prec, recall=rec, auc_roc=auc_roc)

    # ------------------------------------------------------------------ #
    # Final evaluation — best local model for each client
    # ------------------------------------------------------------------ #
    print("\n--- Final Evaluation: Best Local Models ---")
    best_local_results = {}
    for c in client_list:
        local_path = os.path.join(models_dir, f"best_local_model_client_{c.client_id}.pth")
        loss, acc, f1, prec, rec, auc_roc = c.evaluate_model(
            torch.load(local_path, map_location=device), time_step
        )
        best_local_results[c.client_id] = dict(loss=loss, accuracy=acc, f1=f1, precision=prec, recall=rec, auc_roc=auc_roc)
        print(f"  [Client {c.client_id}] [Best Local] "
              f"Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Loss={loss:.4f} AUC={auc_roc:.4f}")

    # ------------------------------------------------------------------ #
    # Side-by-side comparison table
    # ------------------------------------------------------------------ #
    _print_comparison_table(client_list, best_global_results, best_local_results)

    # ------------------------------------------------------------------ #
    # Save all results to JSON
    # ------------------------------------------------------------------ #
    print("\n--- Training Complete ---")
    total_time = sum(round_times)
    avg_round_time = total_time / len(round_times) if round_times else 0

    results = {
        "global_metrics_per_iteration": per_iter_global,
        "local_metrics_per_iteration":  per_iter_local,
        "final_best_global_model":      best_global_results,
        "final_best_local_models":      best_local_results,
        "hyperparameters": args.__dict__,
        "timing_seconds": {
            "per_round": round_times,
            "total_training_time": total_time,
            "average_round_time": avg_round_time
        }
    }

    results_folder = base_save_dir
    filename = f"metrics_ditto_{args.exp_name}.json"
    save_results_as_json(filename, results, project_root, results_folder)