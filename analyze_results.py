import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_experiment_data(project_root, save_dir, exp_name):
    """
    Scans the results directory for all algorithms and loads their JSON metrics.
    """
    algorithms = ['fedavg', 'scaffold', 'fedprox', 'ditto', 'centralized']
    exp_path = os.path.join(project_root, save_dir, exp_name)
    
    data = {}
    
    for algo in algorithms:
        algo_dir = os.path.join(exp_path, algo)
        json_filename = f"metrics_{algo}_{exp_name}.json"
        json_path = os.path.join(algo_dir, json_filename)
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data[algo] = json.load(f)
            print(f"Loaded {algo} data from {json_path}")
        else:
            print(f"Warning: Data for '{algo}' not found at {json_path}. Skipping.")
            
    return data

def extract_convergence_trends(data):
    """
    Extracts the average global AND local accuracy, F1 score, AUC-ROC, and Loss per round/epoch.
    """
    global_trends = {}
    local_trends = {}
    
    for algo, metrics in data.items():
        g_acc, g_f1, g_auc, g_loss = [], [], [], []
        l_acc, l_f1, l_auc, l_loss = [], [], [], []
        
        if algo == 'centralized':
            # Centralized stores an array of epoch dictionaries
            history = metrics.get('metrics_history', [])
            g_acc = [epoch.get('accuracy', 0) for epoch in history]
            g_f1 = [epoch.get('f1', 0) for epoch in history]
            g_auc = [epoch.get('auc_roc', 0.5) for epoch in history]
            g_loss = [epoch.get('train_loss', 0) for epoch in history]
            
            # Centralized has no "local" distinct from "global", so we copy it as a baseline
            l_acc, l_f1, l_auc, l_loss = g_acc, g_f1, g_auc, g_loss
        else:
            # Global History
            global_history = metrics.get('global_metrics_per_iteration', {})
            iterations = sorted(global_history.keys(), key=int)
            for it in iterations:
                clients_data = global_history[it]
                g_acc.append(np.mean([c.get('accuracy', 0) for c in clients_data.values()]))
                g_f1.append(np.mean([c.get('f1', 0) for c in clients_data.values()]))
                g_auc.append(np.mean([c.get('auc_roc', 0.5) for c in clients_data.values()]))
                g_loss.append(np.mean([c.get('loss', 0) for c in clients_data.values()]))
                
            # Local History
            local_history = metrics.get('local_metrics_per_iteration', {})
            iterations = sorted(local_history.keys(), key=int)
            for it in iterations:
                clients_data = local_history[it]
                l_acc.append(np.mean([c.get('accuracy', 0) for c in clients_data.values()]))
                l_f1.append(np.mean([c.get('f1', 0) for c in clients_data.values()]))
                l_auc.append(np.mean([c.get('auc_roc', 0.5) for c in clients_data.values()]))
                l_loss.append(np.mean([c.get('loss', 0) for c in clients_data.values()]))
                
        global_trends[algo] = {'accuracy': g_acc, 'f1': g_f1, 'auc': g_auc, 'loss': g_loss}
        local_trends[algo] = {'accuracy': l_acc, 'f1': l_f1, 'auc': l_auc, 'loss': l_loss}
        
    return global_trends, local_trends

def extract_final_metrics(data):
    """
    Extracts the final best performance and timing metrics into a Pandas DataFrame.
    """
    records = []
    
    for algo, metrics in data.items():
        if algo == 'centralized':
            history = metrics.get('metrics_history', [])
            if history:
                best_epoch = max(history, key=lambda x: x.get('accuracy', 0))
                final_g_acc = best_epoch.get('accuracy', 0)
                final_g_f1 = best_epoch.get('f1', 0)
            else:
                final_g_acc, final_g_f1 = 0, 0
                
            final_l_acc, final_l_f1 = final_g_acc, final_g_f1
            total_time = metrics.get('timing_seconds', {}).get('total_training_time', 0)
        else:
            # Average the final best global model across all clients
            final_global = metrics.get('final_best_global_model', {})
            if final_global:
                final_g_acc = np.mean([c.get('accuracy', 0) for c in final_global.values()])
                final_g_f1 = np.mean([c.get('f1', 0) for c in final_global.values()])
            else:
                final_g_acc, final_g_f1 = 0, 0
                
            # Average the final best local model across all clients
            final_local = metrics.get('final_best_local_models', {})
            if final_local:
                final_l_acc = np.mean([c.get('accuracy', 0) for c in final_local.values()])
                final_l_f1 = np.mean([c.get('f1', 0) for c in final_local.values()])
            else:
                final_l_acc, final_l_f1 = 0, 0
                
            total_time = metrics.get('timing_seconds', {}).get('total_training_time', 0)
            
        records.append({
            'Algorithm': algo.capitalize(),
            'Global Acc': final_g_acc,
            'Global F1': final_g_f1,
            'Local Acc': final_l_acc,
            'Local F1': final_l_f1,
            'Total Time (s)': total_time
        })
        
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by='Global Acc', ascending=False).reset_index(drop=True)
    return df

def plot_comparisons(trends, output_dir, prefix="Global"):
    """
    Plots the convergence of Accuracy, F1, AUC-ROC, and Loss over rounds/epochs in a 2x2 grid.
    Prefix defines if it's Global or Local.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'fedavg': '#1f77b4', 'scaffold': '#ff7f0e', 'fedprox': '#2ca02c', 
              'ditto': '#d62728', 'centralized': '#9467bd'}
    
    for algo, trend in trends.items():
        if not trend['accuracy']:
            continue
        x_axis = range(1, len(trend['accuracy']) + 1)
        label = algo.capitalize()
        color = colors.get(algo, '#333333')
        
        # Plot
        axes[0, 0].plot(x_axis, trend['accuracy'], label=label, color=color, linewidth=2)
        axes[0, 1].plot(x_axis, trend['f1'], label=label, color=color, linewidth=2)
        axes[1, 0].plot(x_axis, trend['auc'], label=label, color=color, linewidth=2)
        axes[1, 1].plot(x_axis, trend['loss'], label=label, color=color, linewidth=2)

    # Configure axes
    axes[0, 0].set_title(f'{prefix} Accuracy Convergence', fontsize=14)
    axes[0, 0].set_xlabel('Global Round / Centralized Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    axes[0, 0].legend()

    axes[0, 1].set_title(f'{prefix} F1 Score Convergence', fontsize=14)
    axes[0, 1].set_xlabel('Global Round / Centralized Epoch')
    axes[0, 1].set_ylabel('Macro F1 Score')
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    axes[0, 1].legend()

    axes[1, 0].set_title(f'{prefix} AUC-ROC Convergence', fontsize=14)
    axes[1, 0].set_xlabel('Global Round / Centralized Epoch')
    axes[1, 0].set_ylabel('AUC-ROC')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    axes[1, 0].legend()

    axes[1, 1].set_title(f'{prefix} Loss Convergence', fontsize=14)
    axes[1, 1].set_xlabel('Global Round / Centralized Epoch')
    axes[1, 1].set_ylabel('Loss (Cross Entropy)')
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    axes[1, 1].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{prefix.lower()}_algorithm_comparison_convergence.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Generated plot: {plot_path}")
    plt.close()

def plot_client_distributions(data, output_dir):
    """
    Generates a 1x2 grouped boxplot showing the distribution of final client 
    Accuracies and F1 Scores for the Global vs Local models.
    Also plots the Centralized baseline as a horizontal reference line.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plot_data = []
    centralized_acc = None
    centralized_f1 = None
    
    for algo, metrics in data.items():
        if algo == 'centralized':
            # Extract the best centralized epoch to use as a baseline
            history = metrics.get('metrics_history', [])
            if history:
                best_epoch = max(history, key=lambda x: x.get('accuracy', 0))
                centralized_acc = best_epoch.get('accuracy', 0)
                centralized_f1 = best_epoch.get('f1', 0)
            continue
            
        # 1. Get Global Model Metrics for all clients
        final_global = metrics.get('final_best_global_model', {})
        for client_id, results in final_global.items():
            plot_data.append({
                'Algorithm': algo.capitalize(),
                'Evaluation Type': 'Global Model',
                'Accuracy': results.get('accuracy', 0),
                'F1 Score': results.get('f1', 0)
            })
            
        # 2. Get Local Model Metrics for all clients
        final_local = metrics.get('final_best_local_models', {})
        for client_id, results in final_local.items():
            plot_data.append({
                'Algorithm': algo.capitalize(),
                'Evaluation Type': 'Local Model',
                'Accuracy': results.get('accuracy', 0),
                'F1 Score': results.get('f1', 0)
            })
            
    df = pd.DataFrame(plot_data)
    
    if df.empty:
        print("No client data found for boxplots.")
        return

    # Create a 1x2 subplot grid
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.set_theme(style="whitegrid")
    
    palette = {'Global Model': '#1f77b4', 'Local Model': '#ff7f0e'}
    
    # ---------------------------------------------------------
    # LEFT PLOT: Accuracy Boxplot
    # ---------------------------------------------------------
    sns.boxplot(
        data=df, x='Algorithm', y='Accuracy', hue='Evaluation Type',
        palette=palette, width=0.6, boxprops=dict(alpha=0.8), ax=axes[0]
    )
    
    # Add Centralized Baseline Line
    if centralized_acc is not None:
        axes[0].axhline(y=centralized_acc, color='#d62728', linestyle='--', linewidth=2.5, 
                        label=f'Centralized Baseline ({centralized_acc:.2f})')
        
    axes[0].set_title('Distribution of Final Client Accuracies', fontsize=16, pad=15)
    axes[0].set_xlabel('Algorithm', fontsize=14)
    axes[0].set_ylabel('Accuracy', fontsize=14)
    axes[0].set_ylim(0.0, 1.05)
    
    # Reconstruct legend to include the axhline
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles, labels=labels, title='Model Evaluated', 
                   title_fontsize='13', fontsize='12', loc='lower right')
    
    # ---------------------------------------------------------
    # RIGHT PLOT: F1 Score Boxplot
    # ---------------------------------------------------------
    sns.boxplot(
        data=df, x='Algorithm', y='F1 Score', hue='Evaluation Type',
        palette=palette, width=0.6, boxprops=dict(alpha=0.8), ax=axes[1]
    )
    
    # Add Centralized Baseline Line
    if centralized_f1 is not None:
        axes[1].axhline(y=centralized_f1, color='#d62728', linestyle='--', linewidth=2.5, 
                        label=f'Centralized Baseline ({centralized_f1:.2f})')
        
    axes[1].set_title('Distribution of Final Client F1 Scores', fontsize=16, pad=15)
    axes[1].set_xlabel('Algorithm', fontsize=14)
    axes[1].set_ylabel('Macro F1 Score', fontsize=14)
    axes[1].set_ylim(0.0, 1.05)
    
    # Reconstruct legend to include the axhline
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles=handles, labels=labels, title='Model Evaluated', 
                   title_fontsize='13', fontsize='12', loc='lower right')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'client_metrics_distribution_boxplot.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Generated Boxplot: {plot_path}")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Federated Learning Results")
    parser.add_argument('--project_root', type=str, default='.', help='Root directory of the project')
    parser.add_argument('--save_dir', type=str, default='results', help='Base results directory')
    parser.add_argument('--exp_name', type=str, default='debug_run', help='Name of the experiment to analyze')
    args = parser.parse_args()

    print(f"--- Analyzing Experiment: {args.exp_name} ---")
    
    # 1. Load Data
    data = load_experiment_data(args.project_root, args.save_dir, args.exp_name)
    if not data:
        print("No data found. Ensure your algorithms have run and saved JSON files to the correct paths.")
        return

    # 2. Extract Trends & Final Metrics
    global_trends, local_trends = extract_convergence_trends(data)
    summary_df = extract_final_metrics(data)

    # 3. Print Summary Table
    print("\n" + "="*85)
    print("FINAL PERFORMANCE SUMMARY (Averaged across clients for FL)")
    print("="*85)
    print(summary_df.to_string(
        index=False, 
        formatters={
            'Global Acc': '{:.4f}'.format,
            'Global F1': '{:.4f}'.format,
            'Local Acc': '{:.4f}'.format,
            'Local F1': '{:.4f}'.format,
            'Total Time (s)': '{:.1f}'.format
        }
    ))
    print("="*85)

    # 4. Generate Plots
    analysis_dir = os.path.join(args.project_root, args.save_dir, args.exp_name, "analysis")
    
    print("\n--- Generating convergence plots ---")
    plot_comparisons(global_trends, analysis_dir, prefix="Global")
    plot_comparisons(local_trends, analysis_dir, prefix="Local")
    
    print("\n--- Generating distribution boxplots ---")
    plot_client_distributions(data, analysis_dir)
    
    # Save the summary table as a CSV
    csv_path = os.path.join(analysis_dir, "summary_metrics.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary table saved to: {csv_path}")

if __name__ == "__main__":
    main()