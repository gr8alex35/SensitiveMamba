import os
import torch
import numpy as np
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from .model import SensitiveMamba
from scipy.stats import iqr
from torch.optim import Optimizer
from collections import OrderedDict
from utils import EarlyStop, Logger
from base import ADTrainer, get_best_threshold, adjust_predicts
from base.metrics import *
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

class Trainer(ADTrainer):
    def __init__(self, model, optimizer, alpha, max_epoch, model_save_dir,
                 scheduler=None, use_prob=True, model_save_suffix=''):
        self.model: SensitiveMamba = model
        self.optimizer: Optimizer = optimizer
        self.alpha: float = alpha
        self.max_epoch: int = max_epoch
        self.scheduler = scheduler
        self.device = next(model.parameters()).device
        self.early_stop = EarlyStop(tol_num=10, min_is_best=True)
        self.metrics = ('precision', 'recall', 'f1')
        self._mse_loss = nn.MSELoss(reduction='none')
        self._use_prob = use_prob

        self.beta = 0.5
        self.gamma = 1

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        self.logger = Logger().get_logger(
            str(model), os.path.join(model_save_dir, f'log_{model.__class__.__name__}{model_save_suffix}.txt'))

        self.model_save_name = f'{model.__class__.__name__}{model_save_suffix}.pkl'
        self.model_save_path: str = os.path.join(model_save_dir, self.model_save_name)

        self.loss_keys = ('rec', 'prob')

    def _show_param_nums(self):
        params_num = 0
        for p in filter(lambda p: p.requires_grad, self.model.parameters()):
            params_num += np.prod(p.size())
        self.logger.info('Trainable params num: {}'.format(params_num))
        # self.logger.info(self.model)

    def show_metric_results(self, preds, preds_adjust, labels, prefix='Test'):
        temp_prefix = ('Before PA', 'After PA')
        for i, pred_result in enumerate((preds, preds_adjust)):
            test_results = self.get_metric_results(pred_result, labels, self.metrics)
            log_info = f'{prefix}\t{temp_prefix[i]}\t'
            for metric_name, ret in zip(self.metrics, test_results):
                log_info += '{}:\t{:.4f}\t'.format(metric_name, ret)
            self.logger.info(log_info)
        
        auc = roc_auc_score(labels, preds)
        auc_adj = roc_auc_score(labels, preds_adjust)
        print(f"📈 AUC          = {auc:.4f}")
        print(f"📈 AUC (adjust) = {auc_adj:.4f}")

    ## Contrastive Loss (TS2Vec Style Augmentation)
    def augment_time_series(self, x, crop_ratio=0.9, mask_ratio=0.3):
        B, T, D = x.shape
        crop_len = int(T * crop_ratio)

        # Crop 1
        start1 = torch.randint(0, T - crop_len + 1, (1,)).item()
        x_crop1 = x[:, start1:start1 + crop_len, :]
        x_masked1 = x_crop1.clone()

        mask_len = int(crop_len * mask_ratio)
        for b in range(B):
            m_start = torch.randint(0, crop_len - mask_len + 1, (1,)).item()
            x_masked1[b, m_start:m_start + mask_len, :] = 0

        # Crop 2
        start2 = torch.randint(0, T - crop_len + 1, (1,)).item()
        x_crop2 = x[:, start2:start2 + crop_len, :]
        x_masked2 = x_crop2.clone()

        for b in range(B):
            m_start = torch.randint(0, crop_len - mask_len + 1, (1,)).item()
            x_masked2[b, m_start:m_start + mask_len, :] = 0

        return x_masked1, x_masked2  
    
    # temporal cl
    def temporal_contrastive_loss(self, h1, h2, temperature=0.1):
        B, T, D = h1.shape
        h1_norm = F.normalize(h1, dim=-1)
        h2_norm = F.normalize(h2, dim=-1)
        
        loss = 0
        for t in range(T):
            z1 = h1_norm[:, t, :]  # [B, D]
            z2 = h2_norm[:, t, :]  # [B, D]
            
            pos_sim = torch.sum(z1 * z2, dim=-1) / temperature  # [B]
            sim_matrix = torch.matmul(z1, z2.T) / temperature   # [B, B]
            sim_matrix = sim_matrix.masked_fill(torch.eye(B).bool().to(h1.device), float('-inf'))
            
            log_prob = pos_sim - torch.logsumexp(sim_matrix, dim=1)
            loss += -log_prob.mean()
        return loss / T
    
    # instance cl
    def instance_contrastive_loss(self, h1, h2, temperature=0.1):
        B, T, D = h1.shape
        h1_flat = F.normalize(h1.reshape(B * T, D), dim=-1)
        h2_flat = F.normalize(h2.reshape(B * T, D), dim=-1)

        pos_sim = torch.sum(h1_flat * h2_flat, dim=-1) / temperature
        sim_matrix = torch.matmul(h1_flat, h2_flat.T) / temperature
        sim_matrix.fill_diagonal_(float('-inf'))

        log_prob = pos_sim - torch.logsumexp(sim_matrix, dim=1)
        return -log_prob.mean()

    def hierarchical_contrastive_loss(self, h1, h2, beta=0.5, temporal_unit=0, temperature=0.1):
        z1, z2 = h1, h2
        B, T, D = z1.shape
        device = z1.device
        loss = 0.0
        depth = 0

        while T >= 1:
            if beta > 0:
                inst_cl = self.instance_contrastive_loss(z1, z2, temperature)
                loss += beta * inst_cl

            if (1 - beta) > 0 and depth >= temporal_unit:
                temp_cl = self.temporal_contrastive_loss(z1, z2, temperature)
                loss += (1 - beta) * temp_cl

            if T == 1:
                break

            # MaxPool over time (kernel=2)
            z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)  # [B, T//2, D]
            z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

            T = z1.shape[1]
            depth += 1

        return loss / depth


    def loss_func(self, x_hat, x, log_var_recip, h=None, h_aug1=None, h_aug2=None, with_weight=True, epoch=0):
        rec_loss = self._mse_loss(x_hat, x)
        # grad_loss = gradient_loss(x, x_hat)

        sigma_loss = rec_loss * log_var_recip.exp() - log_var_recip
        if with_weight:
            var = (-log_var_recip).exp().detach()
            mean_var = var.mean(dim=0, keepdim=True) ** self.alpha
            weighted_loss = (var * sigma_loss / mean_var).mean()
        else:
            weighted_loss = sigma_loss.mean()

        total_loss = weighted_loss # + self.beta * grad_loss

        # Hierarchical Contrastive loss phase
        contrastive = torch.tensor(0.0, device=x.device)
        if h_aug1 is not None and h_aug2 is not None:
            contrastive = self.hierarchical_contrastive_loss(
                h_aug1, h_aug2,
                beta=self.beta,
                temporal_unit=1,
                temperature=0.1
            )

        total_loss += self.gamma * contrastive

        return rec_loss.mean(), total_loss, contrastive

    def train_one_epoch(self, data_loader, epoch):
        loss_dict = OrderedDict(**{k: 0 for k in self.loss_keys})
        for og in data_loader:
            og = og.type(torch.float32).to(self.device)

            # augmentation for contrastive learning
            x_crop, x_masked = self.augment_time_series(og)
            # encode original and augmented views
            rec, log_var_recip, h = self.model(og, return_rep=True)
            # TS2Vec Style Contrastive Learning
            _, _, h_aug1 = self.model(x_crop, return_rep=True)
            _, _, h_aug2 = self.model(x_masked, return_rep=True)

            rec_loss, prob_loss, cl_loss = self.loss_func(
                rec, og, log_var_recip,
                h=h, h_aug1=h_aug1, h_aug2=h_aug2, epoch=epoch
            )

            for k, v in zip(loss_dict, (rec_loss, prob_loss)):
                loss_dict[k] += v.item()

            loss = prob_loss if self._use_prob else rec_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        for k in loss_dict:
            loss_dict[k] /= len(data_loader)
        return loss_dict

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        loss_dict = OrderedDict(**{k: 0 for k in self.loss_keys})
        for og in data_loader:
            og = og.type(torch.float32).to(self.device)
            
            rec, log_var_recip = self.model(og)
            rec_loss, prob_loss, _ = self.loss_func(rec, og, log_var_recip, with_weight=False)

            for k, v in zip(loss_dict, (rec_loss, prob_loss)):
                loss_dict[k] += v.item()

        for k in loss_dict:
            loss_dict[k] /= len(data_loader)

        return loss_dict

    def train(self, train_data_loader, eval_data_loader):
        min_loss_valid = float('inf')
        key = self.loss_keys[int(self._use_prob)]

        for epoch in range(1, self.max_epoch + 1):
            
            self.model.train()
            loss_dict_train = self.train_one_epoch(train_data_loader, epoch)
            loss_dict_val = self.evaluate(eval_data_loader)

            if loss_dict_val[key] < min_loss_valid:
                min_loss_valid = loss_dict_val[key]
                torch.save(self.model.state_dict(), self.model_save_path)

            log_msg += 'Epoch {:02}/{:02}\tTrain:'.format(epoch, self.max_epoch)
            for k, v in loss_dict_train.items():
                log_msg += '\t{}:{:.6f}'.format(k, v)
            log_msg += '\tValid:'
            for k, v in loss_dict_val.items():
                log_msg += '\t{}:{:.6f}'.format(k, v)
            self.logger.info(log_msg)

            if self.early_stop.reach_stop_criteria(loss_dict_val[key]):
                break

            if self.scheduler:
                self.scheduler.step()
        self.logger.info('Training is over...')
        

    @torch.no_grad()
    def _get_anomaly_score(self, data_loader, load_state=True, select_pos='mid'):
        assert select_pos in ('mid', 'tail')
        if load_state:
            self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()

        errors, labels = [], []
        recons_all, inputs_all = [], []
        for og, label in data_loader:
            og = og.type(torch.float32).to(self.device)
            
            rec, log_var_recip = self.model(og)

            anomaly_score = self._mse_loss(rec, og)
            if self._use_prob:
                anomaly_score = anomaly_score * log_var_recip.exp() - log_var_recip
            
            pos = og.size(1) // 2 if select_pos == 'mid' else og.size(1) - 1
            errors.append(anomaly_score[:, pos])
            labels.append(label[:, pos])
            recons_all.append(rec[:, pos].detach().cpu())
            inputs_all.append(og[:, pos].detach().cpu())

        errors = torch.cat(errors, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()

        ## Visulalizatin Code
        recons_all = torch.cat(recons_all, dim=0).numpy()
        inputs_all = torch.cat(inputs_all, dim=0).numpy()
        save_dir = os.path.dirname(self.model_save_path)
        plot_save_path = os.path.join(save_dir, f"reconstruction_vs_gt_pos{pos}.png")

        num_features = inputs_all.shape[1]
        rows = num_features
        cols = 2  # Left: GT/Rec, Right: Error

        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 2), sharex=True)

        for f in range(num_features):
            # ax_left: GT vs Rec
            ax_left = axes[f, 0] if rows > 1 else axes[0]
            ax_left.plot(inputs_all[:, f], label='GT', color='tab:blue', linewidth=1.0)
            ax_left.plot(recons_all[:, f], label='Rec', color='tab:orange', linewidth=1.0)
            ax_left.set_ylabel(f'F{f}')
            ax_left.grid(True)
            if f == 0:
                ax_left.legend(loc='upper right', fontsize='small')

            # ax_right: Error
            ax_right = axes[f, 1] if rows > 1 else axes[1]
            ax_right.plot(errors[:, f], label='Err', color='red', linewidth=1.0)
            ax_right.set_ylabel(f'F{f}')
            ax_right.grid(True)
            if f == 0:
                ax_right.set_title('Anomaly Score')

        # set_xlabel
        axes[-1, 0].set_xlabel('Sample Index')
        axes[-1, 1].set_xlabel('Sample Index')

        # suptitle + savefig
        fig.suptitle(f'Reconstruction + Error View at Time Index {pos}', fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(plot_save_path)
        plt.close()

        print(f"✅ GT/Rec + Error dual-column subplot saved at: {plot_save_path}")

        return errors, labels

    def test(self, data_loader, ignore_dims=None, select_pos='mid'):
        errors, labels = self._get_anomaly_score(data_loader, True, select_pos)
        labels = labels.astype(int)
        preds, preds_adjust = self.get_pred_results(errors, labels, ignore_dims)
        self.show_metric_results(preds, preds_adjust, labels)
        self.plot_preds_with_fn_full(preds, preds_adjust, labels)

        f1 = get_f1(preds, labels)
        f1_adj = get_f1(preds_adjust, labels)
        auc = roc_auc_score(labels, preds)
        auc_adj = roc_auc_score(labels, preds_adjust)

        print(f"📊 F1           = {f1:.4f}")
        print(f"📊 F1 (adjust)  = {f1_adj:.4f}")
        print(f"📈 AUC          = {auc:.4f}")
        print(f"📈 AUC (adjust) = {auc_adj:.4f}")

        print("Length of preds:", len(preds))
        print("Length of labels:", len(labels))

        return np.stack((preds, preds_adjust, labels), axis=0)

    def get_pred_results(self, errors: np.ndarray, labels: np.ndarray, ignore_dims=None):
        # Normalization
        median, iqr_ = np.median(errors, axis=0), iqr(errors, axis=0)
        errors = (errors - median) / (iqr_ + 1e-9)

        if ignore_dims:
            errors[:, ignore_dims] = 0
        final_errors = errors.max(axis=1)

        thr, _ = get_best_threshold(final_errors, labels, adjust=False)
        preds = (final_errors >= thr).astype(int)

        thr_adjust, _ = get_best_threshold(final_errors, labels, adjust=True)
        preds_adjust = (final_errors >= thr_adjust).astype(int)
        preds_adjust = adjust_predicts(preds_adjust, labels)

        return preds, preds_adjust

    # Visualizing code for false negative
    def plot_preds_with_fn_full(self, preds, preds_adjust, labels):
        preds = np.array(preds)
        preds_adjust = np.array(preds_adjust)
        labels = np.array(labels)
        save_dir = os.path.dirname(self.model_save_path)
        plot_save_path = os.path.join(save_dir, f"fn_visualize.png")

        x = np.arange(len(labels))

        fig, axes = plt.subplots(2, 2, figsize=(20, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

        for col, (pred, title) in enumerate(zip([preds, preds_adjust], ['Raw Prediction', 'Adjusted Prediction'])):
            # (0, col): Predicted label vs Ground truth
            axes[0, col].plot(x, pred, label='Prediction', color='blue', alpha=0.7)
            fn_mask = (labels == 1) & (pred == 0)
            axes[0, col].fill_between(x, 0, 1, where=fn_mask, color='orange', alpha=0.3,
                                    transform=axes[0, col].get_xaxis_transform(), label='False Negative')
            axes[0, col].set_title(title)
            axes[0, col].legend()
            axes[0, col].grid(True)

            # (1, col): Ground truth + False negative
            axes[1, col].plot(x, labels, label='Ground Truth', color='red', alpha=0.5)
            axes[1, col].set_xlabel("Time Index")
            axes[1, col].grid(True)
            if col == 0:
                axes[1, col].set_ylabel("Label Only")
            axes[1, col].legend()

        plt.tight_layout()
        plt.savefig(plot_save_path)
        plt.close()
        print(f"✅ False negative visualization saved to: {plot_save_path}")