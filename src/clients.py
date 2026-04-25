import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
import copy
import utils
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

class Client:
    def __init__(self, client_id, args, domain_path, assigned_domains, device, model):
        self.client_id = client_id
        self.args = args
        self.domain_path = domain_path
        self.assigned_domains = assigned_domains
        self.device = device
        
        self.local_model = copy.deepcopy(model).to(device)
        self.eval_model = copy.deepcopy(model).to(device) 
        self.criterion = nn.CrossEntropyLoss()
        
        # Local metrics storage
        self.local_loss_history = []

        # Data Loaders
        self.train_domains_loader = {}
        self.test_domains_loader = {}

        all_test_datasets = []
        all_train_datasets = []

        domains = utils.create_domains(domain_path, assigned_domains)
        for key, files in domains.items():
            train_loader, test_loader = utils.load_data(
                self.domain_path, key, files, 
                window_size=self.args.window_size, 
                step_size=self.args.step_size, 
                batch_size=self.args.batch_size, 
                n_raw_features=getattr(self.args, 'n_raw_features', None),
                device = device
            )
            all_train_datasets.append(train_loader.dataset)
            all_test_datasets.append(test_loader.dataset)
        
        combined_train_dataset = ConcatDataset(all_train_datasets)
        combined_test_dataset = ConcatDataset(all_test_datasets)

        self.train_domains_loader = DataLoader(combined_train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.test_domains_loader = DataLoader(combined_test_dataset, batch_size=self.args.batch_size, shuffle=False)

        # ----------------------------------------------------------------- #
        # Algorithm-Specific Initializations
        # ----------------------------------------------------------------- #
        if self.args.algorithm == 'scaffold':
            # Local control variate (zero-initialised)
            self.control_variate = {
                name: torch.zeros_like(p, device=device)
                for name, p in self.local_model.named_parameters()
            }
        elif self.args.algorithm == 'ditto':
            # Personalized model and optimizer for Ditto
            self.personalized_model = copy.deepcopy(model).to(device)
            self.personalized_optimizer = optim.SGD(self.personalized_model.parameters(), lr=self.args.lr)

    # ----------------------------------------------------------------- #
    # Training Router
    # ----------------------------------------------------------------- #
    def train(self, global_model_state, time_step, server_control_variate=None):
        """
        Routes the local training to the correct algorithm strategy.
        """
        if self.args.algorithm == 'scaffold':
            return self._train_scaffold(global_model_state, server_control_variate, time_step)
        elif self.args.algorithm == 'fedprox':
            return self._train_fedprox(global_model_state, time_step)
        elif self.args.algorithm == 'ditto':
            return self._train_ditto(global_model_state, time_step)
        else: # fedavg
            return self._train_fedavg(global_model_state, time_step)

    # ----------------------------------------------------------------- #
    # Standard FedAvg Training
    # ----------------------------------------------------------------- #
    def _train_fedavg(self, global_model_state, time_step):
        self.local_model.load_state_dict(global_model_state)
        self.local_model.train()
        optimizer = optim.SGD(self.local_model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for epoch in range(self.args.local_epochs):
            epoch_loss = 0.0
            for data, target in self.train_domains_loader:
                
                optimizer.zero_grad(set_to_none=True)
                output, _ = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                #torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.detach()

            avg_loss = epoch_loss.item() / len(self.train_domains_loader)
            self.local_loss_history.append(avg_loss)
            
        return self.local_model.state_dict()

    # ----------------------------------------------------------------- #
    # FedProx Training
    # ----------------------------------------------------------------- #
    def _train_fedprox(self, global_model_state, time_step):
        self.local_model.load_state_dict(global_model_state)
        self.local_model.train()
        optimizer = optim.SGD(self.local_model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        
        global_params = {k: v.detach().clone().to(self.device) for k, v in global_model_state.items()}
                
        for epoch in range(self.args.local_epochs):
            epoch_loss = 0.0
            for data, target in self.train_domains_loader:
                optimizer.zero_grad(set_to_none=True)
                output, _ = self.local_model(data)
                loss = self.criterion(output, target)

                prox = 0.0
                for name, param in self.local_model.named_parameters():
                    prox += torch.sum((param - global_params[name]) ** 2)

                # Add proximal term penalty
                total_loss = loss + (self.args.mu / 2) * prox
                total_loss.backward()
                
                #torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += total_loss.detach()

            avg_loss = epoch_loss.item() / len(self.train_domains_loader)
            self.local_loss_history.append(avg_loss)
            
        return self.local_model.state_dict()

    # ----------------------------------------------------------------- #
    # Scaffold Training
    # ----------------------------------------------------------------- #
    def _train_scaffold(self, global_model_state, server_control_variate, time_step):
        self.local_model.load_state_dict(global_model_state)
        self.local_model.train()
        
        optimizer = optim.SGD(self.local_model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        loader = self.train_domains_loader
        K = self.args.local_epochs * len(loader)

        for epoch in range(self.args.local_epochs):
            for data, target in loader:

                optimizer.zero_grad(set_to_none=True)
                output, _ = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()

                #torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=5.0)
                optimizer.step()

                # SCAFFOLD post-step correction
                with torch.no_grad():
                    for name, param in self.local_model.named_parameters():
                        ci = self.control_variate[name]
                        c_serv = server_control_variate[name]
                        param.data.add_(ci - c_serv, alpha=self.args.lr)

        # Control variate update
        control_delta = {}
        new_control = {}
        with torch.no_grad():
            for name, param in self.local_model.named_parameters():
                x_global = global_model_state[name].to(self.device)
                x_local = param.data
                ci_old = self.control_variate[name]
                c_serv = server_control_variate[name]

                ci_new = ci_old - c_serv + (x_global - x_local) / (K * self.args.lr)
                delta_c = ci_new - ci_old

                new_control[name] = ci_new.clone()
                control_delta[name] = delta_c.clone()

        self.control_variate = new_control

        # Model delta
        model_delta = {}
        with torch.no_grad():
            for k in global_model_state:
                model_delta[k] = (
                    self.local_model.state_dict()[k].to(self.device)
                    - global_model_state[k].to(self.device)
                )

        return model_delta, control_delta

    # ----------------------------------------------------------------- #
    # Ditto Training
    # ----------------------------------------------------------------- #
    def _train_ditto(self, global_model_state, time_step):
        if time_step == 0:
            self.personalized_model.load_state_dict(global_model_state)

        self.local_model.load_state_dict(global_model_state)
        self.local_model.train()
        optimizer = optim.SGD(self.local_model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        global_ref = {n: p.clone().detach() for n, p in self.local_model.named_parameters()}
        
        # 1. Standard Global Model Training
        for epoch in range(self.args.local_epochs):
            for data, target in self.train_domains_loader:
                
                optimizer.zero_grad(set_to_none=True)
                output, _ = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=5.0)
                optimizer.step()

        # 2. Ditto Proximal Update for Personalized Model
        self.personalized_model.train()
        for epoch in range(self.args.local_epochs):
            epoch_loss = 0.0
            for data, target in self.train_domains_loader:
                self.personalized_optimizer.zero_grad(set_to_none=True)
                output, _ = self.personalized_model(data)
                
                loss = self.criterion(output, target)
                
                proximal_term = 0.0
                for name, param in self.personalized_model.named_parameters():
                    proximal_term += (param - global_ref[name]).norm(2)**2
                
                total_loss = loss + (self.args.lam / 2) * proximal_term
                total_loss.backward()
                self.personalized_optimizer.step()
                epoch_loss += total_loss.detach()
                
            avg_loss = epoch_loss.item() / len(self.train_domains_loader)
            self.local_loss_history.append(avg_loss)
                
        return self.local_model.state_dict()

    # ----------------------------------------------------------------- #
    # Shared Evaluation Methods
    # ----------------------------------------------------------------- #
    def evaluate_model(self, model_state, time_step):
        self.eval_model.load_state_dict(model_state)
        self.eval_model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for data, target in self.test_domains_loader:
                output, _ = self.eval_model(data)
                loss = self.criterion(output, target)
                total_loss += loss.detach() 
                
                preds = torch.argmax(output, dim=1)
                probs = F.softmax(output, dim=1)[:, 1] 
                
                all_preds.append(preds)
                all_targets.append(target)
                all_probs.append(probs)

        all_preds = torch.cat(all_preds).cpu().numpy()
        all_targets = torch.cat(all_targets).cpu().numpy()
        all_probs = torch.cat(all_probs).cpu().numpy()

        evaluation_loss = total_loss.item() / len(self.test_domains_loader)
        accuracy  = accuracy_score(all_targets, all_preds)
        f1        = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall    = recall_score(all_targets, all_preds, average='macro', zero_division=0)

        try:
            auc_roc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            auc_roc = 0.5

        return evaluation_loss, accuracy, f1, precision, recall, auc_roc

    def evaluate_global_model(self, model_state, time_step):
        loss, acc, f1, prec, rec, auc_roc = self.evaluate_model(model_state, time_step)
        if not self.args.benchmark:
            print(f"  [Client {self.client_id}] [Global] domain={self.assigned_domains} "
                  f"Loss={loss:.4f} Acc={acc:.4f} F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} AUC={auc_roc:.4f}")
        return loss, acc, f1, prec, rec, auc_roc

    def evaluate_local_model_full(self, time_step):
        # For Ditto, the local metrics should reflect the personalized model.
        # For all others, it reflects the standard local model.
        if self.args.algorithm == 'ditto':
            model_state = self.personalized_model.state_dict()
        else:
            model_state = self.local_model.state_dict()
            
        loss, acc, f1, prec, rec, auc_roc = self.evaluate_model(model_state, time_step)
        
        if not self.args.benchmark:
            print(f"  [Client {self.client_id}] [Local]  domain={self.assigned_domains} "
                  f"Loss={loss:.4f} Acc={acc:.4f} F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} AUC={auc_roc:.4f}")
        return loss, acc, f1, prec, rec, auc_roc