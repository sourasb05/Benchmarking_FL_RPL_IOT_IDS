import copy

import torch
import torch.nn as nn
import torch.optim as optim
import utils
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch.nn.functional as F


class ScaffoldClient:
    """
    SCAFFOLD client — Karimireddy et al., 2020 (Algorithm 1, Option II).

    Practical implementation for adaptive optimizers (Adam)
    ────────────────────────────────────────────────────────
    The paper derives SCAFFOLD for SGD where the correction (c - c_i) is
    added to the gradient before the step.  With Adam the adaptive scaling
    distorts the correction signal, so the standard approach used in
    practice is to apply the correction as a direct parameter nudge *after*
    the Adam step:

        x ← x - lr * (c_i - c)       # post-step correction

    This is equivalent to subtracting the correction from the update,
    matching the paper's intent while keeping Adam's benefits.

    Control variate update (Option II)
    ────────────────────────────────────
    After K local steps:
        c_i_new = c_i - c + (1 / (K * lr)) * (x_global - x_local)
        Δc_i    = c_i_new - c_i_old
    Server accumulates:
        x ← x + (1/N) Σ Δx_i
        c ← c + (1/N) Σ Δc_i
    """

    def __init__(self, client_id, args, domain_path, assigned_domains, device, model):
        self.client_id        = client_id
        self.args             = args
        self.domain_path      = domain_path
        self.assigned_domains = assigned_domains
        self.device           = device

        self.local_model = copy.deepcopy(model).to(device)
        self.eval_model  = copy.deepcopy(model).to(device)
        self.criterion   = nn.CrossEntropyLoss()

        # ── data loaders ─────────────────────────────────────────────
        self.train_domains_loader = {}
        self.test_domains_loader  = {}
        domains = utils.create_domains(domain_path, assigned_domains)
        for key, files in domains.items():
            self.train_domains_loader[key], self.test_domains_loader[key] = utils.load_data(
                self.domain_path, key, files,
                window_size=self.args.window_size,
                step_size=self.args.step_size,
                batch_size=self.args.batch_size,
                n_raw_features=getattr(self.args, 'n_raw_features', None),
            )
        self.domain_keys = list(self.train_domains_loader.keys())

        # ── local control variate  c_i  (zero-initialised) ───────────
        self.control_variate = {
            name: torch.zeros_like(p, device=device)
            for name, p in self.local_model.named_parameters()
        }

    # ----------------------------------------------------------------- #
    # Training
    # ----------------------------------------------------------------- #

    def train(self, global_model_state, server_control_variate, time_step):
        """
        SCAFFOLD local training with Adam + post-step correction.

        Parameters
        ----------
        global_model_state     : server model state_dict
        server_control_variate : {param_name: tensor}  — server's  c
        time_step              : domain index

        Returns
        -------
        model_delta   : {k: x_local[k] - x_global[k]}
        control_delta : {k: c_i_new[k] - c_i_old[k]}
        """
        self.local_model.load_state_dict(copy.deepcopy(global_model_state))
        self.local_model.train()

        # Use Adam — same as FedAvg so the base training is identical.
        # The SCAFFOLD correction is applied as a separate post-step nudge.
        optimizer = optim.Adam(self.local_model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        loader    = self.train_domains_loader[self.domain_keys[time_step]]
        K         = self.args.local_epochs * len(loader)   # total local steps

        for _epoch in range(self.args.local_epochs):
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output, _ = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=5.0)
                optimizer.step()

                # ── SCAFFOLD post-step correction ─────────────────────
                # After Adam updates x, apply:  x ← x - lr * (c_i - c)
                # i.e. nudge parameters in the direction of (c - c_i).
                with torch.no_grad():
                    for name, param in self.local_model.named_parameters():
                        ci     = self.control_variate[name]
                        c_serv = server_control_variate[name]
                        param.data.add_(ci - c_serv, alpha=-self.args.lr)

        # ── Option II control variate update ─────────────────────────
        # c_i_new = c_i - c + (1 / (K * lr)) * (x_global - x_local)
        control_delta = {}
        new_control   = {}
        with torch.no_grad():
            for name, param in self.local_model.named_parameters():
                x_global = global_model_state[name].to(self.device)
                x_local  = param.data
                ci_old   = self.control_variate[name]
                c_serv   = server_control_variate[name]

                ci_new  = ci_old - c_serv + (x_global - x_local) / (K * self.args.lr)
                delta_c = ci_new - ci_old

                new_control[name]   = ci_new.clone()
                control_delta[name] = delta_c.clone()

        self.control_variate = new_control

        # ── model delta ───────────────────────────────────────────────
        model_delta = {}
        with torch.no_grad():
            for k in global_model_state:
                model_delta[k] = (
                    self.local_model.state_dict()[k].to(self.device)
                    - global_model_state[k].to(self.device)
                )

        return model_delta, control_delta

    # ----------------------------------------------------------------- #
    # Evaluation
    # ----------------------------------------------------------------- #

    def evaluate_model(self, model_state, time_step):
        """
        Evaluate a model (global or local) on this client's test data.
        Returns loss, accuracy, f1, precision, recall.
        """
        self.eval_model.load_state_dict(model_state)
        self.eval_model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for data, target in self.test_domains_loader[self.domain_keys[time_step]]:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.eval_model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                probs = F.softmax(output, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                
        evaluation_loss = total_loss / len(self.test_domains_loader[self.domain_keys[time_step]])
        accuracy  = accuracy_score(all_targets, all_preds)
        f1        = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall    = recall_score(all_targets, all_preds, average='macro', zero_division=0)

        try:
            print(all_targets, all_probs)
            auc_roc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            auc_roc = 0.5

        return evaluation_loss, accuracy, f1, precision, recall, auc_roc

    def evaluate_global_model(self, model_state, time_step):
        loss, acc, f1, prec, rec, auc_roc = self.evaluate_model(model_state, time_step)
        print(f"  [Client {self.client_id}] [Global] domain={self.domain_keys[time_step]} "
              f"Loss={loss:.4f} Acc={acc:.4f} F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} AUC={auc_roc:.4f}")
        return loss, acc, f1, prec, rec, auc_roc

    def evaluate_local_model_full(self, time_step):
        loss, acc, f1, prec, rec, auc_roc = self.evaluate_model(self.local_model.state_dict(), time_step)
        print(f"  [Client {self.client_id}] [Local]  domain={self.domain_keys[time_step]} "
              f"Loss={loss:.4f} Acc={acc:.4f} F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} AUC={auc_roc:.4f}")
        return loss, acc, f1, prec, rec, auc_roc