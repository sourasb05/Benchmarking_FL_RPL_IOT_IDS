import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
import utils
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch.nn.functional as F
import numpy as np

def centralized_training(args, model, device, domains_path, domains, project_root):
    print("\n" + "="*55)
    print("--- Starting Centralized Baseline Training ---")
    print("="*55)

    # Dynamic save paths matching your new structure
    base_save_dir = os.path.join(project_root, args.save_dir, args.exp_name, "centralized")
    models_dir = os.path.join(base_save_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # STEP 1: Pool all data together
    # ------------------------------------------------------------------ #
    print("\n[Step 1] Pooling all datasets...")
    all_train_datasets = []
    all_test_datasets = []

    for key, files in domains.items():
        # Load each domain individually
        tr_l, te_l = utils.load_data(
            domains_path, key, files, 
            window_size=args.window_size, 
            step_size=args.step_size, 
            batch_size=args.batch_size, 
            n_raw_features=getattr(args, 'n_raw_features', None)
        )
        # Extract the underlying TensorDataset
        all_train_datasets.append(tr_l.dataset)
        all_test_datasets.append(te_l.dataset)

    # Fuse them into one massive dataset
    central_train_dataset = ConcatDataset(all_train_datasets)
    central_test_dataset = ConcatDataset(all_test_datasets)

    # Create master DataLoaders
    central_train_loader = DataLoader(central_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    central_test_loader = DataLoader(central_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, persistent_workers=True)

    print(f"Total Centralized Training Samples: {len(central_train_dataset)}")
    print(f"Total Centralized Testing Samples:  {len(central_test_dataset)}")

    # ------------------------------------------------------------------ #
    # STEP 2: Setup Training
    # ------------------------------------------------------------------ #
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # In FedAvg, total data passes = global_iters * local_epochs
    # To keep the comparison fair, we match the total number of epochs
    total_epochs = args.global_iters * args.local_epochs 
    
    epoch_times = []
    metrics_history = []
    best_acc = -1.0

    # ------------------------------------------------------------------ #
    # STEP 3: Standard Centralized Training Loop
    # ------------------------------------------------------------------ #
    print(f"\n[Step 2] Starting Training for {total_epochs} total epochs...")
    for epoch in range(total_epochs):
        epoch_start = time.perf_counter()
        
        model.train()
        total_loss = 0.0
        
        for data, target in central_train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(central_train_loader)
        
        # -------------------------------------------------------------- #
        # STEP 4: Evaluation
        # -------------------------------------------------------------- #
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in central_test_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                probs = F.softmax(output, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
    
        avg_test_loss = test_loss / len(central_test_loader)
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        

        try:
            auc_roc = roc_auc_score(all_targets, all_probs)
        except ValueError as e:
            auc_roc = 0.5
        
        epoch_end = time.perf_counter()
        duration = epoch_end - epoch_start
        epoch_times.append(duration)
        
        print(f"  Epoch [{epoch+1}/{total_epochs}] | Time: {duration:.2f}s | "
              f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | "
              f"Acc: {acc:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | AUC: {auc_roc:.4f}")

        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss,
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "auc_roc": auc_roc,
            "time_seconds": duration
        })
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            best_model_path = os.path.join(models_dir, "best_centralized_model.pth")
            torch.save(model.state_dict(), best_model_path)

    # ------------------------------------------------------------------ #
    # STEP 5: Save Results
    # ------------------------------------------------------------------ #
    print("\n--- Centralized Training Complete ---")
    
    results = {
        "metrics_history": metrics_history,
        "final_best_accuracy": best_acc,
        "hyperparameters": args.__dict__,
        "timing_seconds": {
            "per_epoch": epoch_times,
            "total_training_time": sum(epoch_times),
            "average_epoch_time": sum(epoch_times) / len(epoch_times)
        }
    }
    
    filename = f"metrics_centralized_{args.exp_name}.json"
    utils.save_results_as_json(filename, results, project_root, base_save_dir)