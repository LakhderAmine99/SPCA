"""Validation-only selection. Test evaluation is a separate, explicit operation."""
import copy
import random
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def classification_metrics(logits, labels):
    prediction = logits.argmax(-1).detach().cpu().numpy()
    truth = labels.detach().cpu().numpy()
    return {'macro_f1': float(f1_score(truth, prediction, average='macro', zero_division=0)),
            'micro_f1': float(f1_score(truth, prediction, average='micro', zero_division=0)),
            'loss': float(F.cross_entropy(logits, labels).item())}


def link_metrics(logits, labels):
    scores, truth = logits.detach().cpu().numpy(), labels.detach().cpu().numpy()
    return {'auc': float(roc_auc_score(truth, scores)), 'ap': float(average_precision_score(truth, scores)),
            'loss': float(F.binary_cross_entropy_with_logits(logits, labels).item())}


class ValidationCheckpoint:
    def __init__(self, metric='macro_f1'):
        self.metric, self.metrics, self.state, self.epoch = metric, None, None, 0

    def consider(self, model, metrics, epoch):
        key = (metrics[self.metric], -metrics['loss'])
        if not all(np.isfinite(x) for x in key):
            raise FloatingPointError('Nonfinite validation metric or loss')
        if self.metrics is None or key > (self.metrics[self.metric], -self.metrics['loss']):
            self.metrics = dict(metrics)
            self.state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            self.epoch = epoch
            return True
        return False


def fit(model, train, validation, settings, *, task='nc', epochs=500, patience=100,
        auxiliary_weight=1.0):
    """Only training/validation inputs are accepted; there is no test callback.

    For HeCo/GTC the encoder learns its contrastive objective and the linear
    evaluation head learns from detached embeddings. Both advance each epoch,
    making the manuscript's validation checkpoint rule available to these models.
    """
    if epochs < 1 or patience < 1:
        raise ValueError('epochs and patience must be positive')
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    checkpoint = ValidationCheckpoint('macro_f1' if task == 'nc' else 'auc')
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        contrastive = (getattr(getattr(model, 'encoder', None), 'name', None) in {'heco', 'gtc'})
        if task == 'nc':
            if contrastive:
                # Train the linear evaluation head without changing the two-view objective.
                embeddings = model.encoder(detach_backbone=True)
                logits = model.classifier(embeddings)
                auxiliary = model.auxiliary_loss()
                loss = F.cross_entropy(logits[train['mask']], train['labels'][train['mask']]) + auxiliary_weight * auxiliary
            else:
                logits = model()
                loss = F.cross_entropy(logits[train['mask']], train['labels'][train['mask']])
        else:
            logits = model(train['edge_label_index'])
            loss = F.binary_cross_entropy_with_logits(logits, train['edge_label'])
            loss = loss + auxiliary_weight * model.auxiliary_loss()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            if task == 'nc':
                metrics = classification_metrics(model()[validation['mask']], validation['labels'][validation['mask']])
            else:
                metrics = link_metrics(model(validation['edge_label_index']), validation['edge_label'])
        stale = 0 if checkpoint.consider(model, metrics, epoch) else stale + 1
        if stale >= patience:
            break
    model.load_state_dict(checkpoint.state)
    model.eval()
    return {'validation': checkpoint.metrics, 'best_epoch': checkpoint.epoch, 'epochs_completed': epoch}


@torch.no_grad()
def evaluate_test(model, test, task='nc'):
    model.eval()
    if task == 'nc':
        return classification_metrics(model()[test['mask']], test['labels'][test['mask']])
    return link_metrics(model(test['edge_label_index']), test['edge_label'])


def summarize(runs):
    if not runs:
        raise ValueError('Cannot summarize no runs')
    return {metric: {'mean': float(np.mean([r[metric] for r in runs])),
                     'std': float(np.std([r[metric] for r in runs], ddof=1)) if len(runs) > 1 else None,
                     'runs': len(runs)} for metric in runs[0]}
