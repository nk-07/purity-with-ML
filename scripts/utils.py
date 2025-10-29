import torch
import torch.nn as nn
import torch.nn.functional as F
from LibMTL.metrics import AbsMetric
from LibMTL.loss import AbsLoss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from src.smarts import purity_classes, fg_names

class BCELoss(AbsLoss):
    def __init__(self, pos_weight=None):
        super(BCELoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def compute_loss(self, pred, gt):
        pred_prob = torch.sigmoid(pred)
        pred_labels = (pred_prob > 0.5).float()
        return self.loss_fn(pred, gt.float())

class MultiMarginLoss(AbsLoss):
    def __init__(self):
        super(MultiMarginLoss, self).__init__()
        self.loss_fn = nn.MultiMarginLoss()
        
    def compute_loss(self, pred, gt):
        loss = self.loss_fn(pred, gt)
        return loss

class MultiLabelF1Metric(AbsMetric):
    """Calculates F1-score for multilabel classification."""
    def __init__(self, average='macro', threshold=0.5, filename='predictions.csv'):
        super(MultiLabelF1Metric, self).__init__()
        if average not in ['micro', 'macro', 'weighted', 'samples']:
            raise ValueError("Average must be either: 'micro', 'macro', 'weighted', or 'samples'")
        self.average = average
        self.threshold = threshold
        self.preds = []
        self.gts = []
        self.filename = filename 

    def update_fun(self, pred, gt):
        probs = torch.sigmoid(pred)
        binary_preds = (probs > self.threshold).int()
        self.preds.append(binary_preds.cpu().numpy())
        self.gts.append(gt.cpu().numpy())

    def score_fun(self):
        if not self.gts:
            return [0.0]

        y_pred = np.vstack(self.preds)
        y_true = np.vstack(self.gts)

        tp_matrix = np.logical_and(y_pred == 1, y_true == 1)
        fp_matrix = np.logical_and(y_pred == 1, y_true == 0)
        fn_matrix = np.logical_and(y_pred == 0, y_true == 1)

        #Save predictions to csv
        #df = pd.DataFrame(y_pred)
        #df.to_csv(self.filename, index=False, header=False)
        
        if self.average == 'micro':
            tp = np.sum(tp_matrix)
            fp = np.sum(fp_matrix)
            fn = np.sum(fn_matrix)
            return [self._calculate_f1(tp, fp, fn)[2]]

        if self.average in ['macro', 'weighted']:
            tp = np.sum(tp_matrix, axis=0)
            fp = np.sum(fp_matrix, axis=0)
            fn = np.sum(fn_matrix, axis=0)

            pr_per_class, rc_per_class, f1_per_class = self._calculate_f1(tp, fp, fn)
            #_print_f1(y_true,pr_per_class, rc_per_class, f1_per_class)

            if self.average == 'macro':
                return [np.mean(f1_per_class)]

            else: # weighted
                support = np.sum(y_true, axis=0)
                return [np.average(f1_per_class, weights=support)]

        if self.average == 'samples':
            tp = np.sum(tp_matrix, axis=1)
            fp = np.sum(fp_matrix, axis=1)
            fn = np.sum(fn_matrix, axis=1)

            pr_per_sample, rc_per_sample, f1_per_sample = self._calculate_f1(tp, fp, fn)
            return [np.mean(f1_per_sample)]

    def _calculate_f1(self, tp, fp, fn):
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)
        return precision, recall, f1

    def reinit(self):
        super(MultiLabelF1Metric, self).reinit()
        self.preds = []
        self.gts = []

class F1Metric(AbsMetric):
    """Calculates F1-score for multiclass classification."""
    def __init__(self, average='macro',filename='predictions.csv'):
        super(F1Metric, self).__init__()
        if average not in ['micro', 'macro', 'weighted']:
            raise ValueError("Average must be either: 'micro', 'macro', 'weighted''")
        self.average = average
        self.preds = []
        self.gts = []
        self.filename = filename

    def update_fun(self, pred, gt):
        pred_labels = F.softmax(pred, dim=-1).max(-1)[1]
        self.preds.extend(pred_labels.cpu().numpy())
        self.gts.extend(gt.cpu().numpy())
        
    def score_fun(self):
        if not self.gts or not self.preds:
            return [0.0]
   
        labels = np.unique(np.concatenate((self.gts, self.preds)))
        num_classes = len(labels)
        label_map = {label: i for i, label in enumerate(labels)}

        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        for gt, pred in zip(self.gts, self.preds):
            if gt in label_map and pred in label_map:
                conf_matrix[label_map[gt], label_map[pred]] += 1
        
        tp = np.diag(conf_matrix)
        fp = np.sum(conf_matrix, axis=0) - tp
        fn = np.sum(conf_matrix, axis=1) - tp
        
        if self.average == 'micro':
            total_tp = np.sum(tp)
            total_fp = np.sum(fp)
            total_fn = np.sum(fn)
            
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            return [f1]

        precision_per_class = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        recall_per_class = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        
        f1_per_class = np.divide(2 * precision_per_class * recall_per_class,
                                 precision_per_class + recall_per_class,
                                 out=np.zeros_like(tp, dtype=float),
                                 where=(precision_per_class + recall_per_class) != 0)

        #Save predictions to csv
        #df = pd.read_csv(self.filename, header=None)
        #df[df.shape[1]] = self.preds
        #df.to_csv(self.filename, index=False, header=False)
        
        """
        #Visualize heatmap
        plt.figure(figsize=(4, 4))
        base = plt.cm.get_cmap('BuPu')
        colors = base(np.linspace(0, 0.9, 256))
        white_bupu = [(1, 1, 1)] + list(colors[0:])
        cmap = LinearSegmentedColormap.from_list("BuPu2", white_bupu)

        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

        annot_matrix = []
        for i in range(conf_matrix.shape[0]):
            row = []
            for j in range(conf_matrix.shape[1]):
                row.append(f'{conf_matrix[i, j]}')
            annot_matrix.append(row)


        hmap = sns.heatmap(
            conf_matrix_norm,
            annot=annot_matrix, 
            cmap=cmap,
            square=True, 
            cbar=False, 
            annot_kws={'size': 9}, 
            #xticklabels=purity_classes,
            #yticklabels=purity_classes,
            fmt=''
            )

        plt.xlabel('Predicted Labels',fontsize = 7, ha='center')
        plt.ylabel('True Labels',fontsize = 7, ha='center')
        hmap.set_xticklabels(hmap.get_xmajorticklabels(), fontsize=5, rotation=45)
        hmap.set_yticklabels(hmap.get_ymajorticklabels(), fontsize=5)
        hmap.tick_params(axis='both',
            which='major',  
            width=0.5)
        plt.tight_layout()
        plt.show()


        plt.figure(figsize=(10, 5))
       
        hmap = sns.heatmap(
            conf_matrix_norm, 
            annot=annot_matrix, 
            cmap=cmap,
            #square=True, 
            cbar=False, 
            annot_kws={'size': 10}, 
            xticklabels=purity_classes,
            yticklabels=purity_classes,
            fmt=''
            )

        plt.xlabel('Predicted Labels', fontsize=8, ha='center')
        plt.ylabel('True Labels', fontsize=8, ha='center')
        hmap.set_xticklabels(hmap.get_xmajorticklabels(), fontsize=8, rotation=45)
        hmap.set_yticklabels(hmap.get_ymajorticklabels(), fontsize=8)
        hmap.tick_params(axis='both', which='major', width=0.5)
        plt.tight_layout()
        plt.show()
        """

        if self.average == 'macro':
            """
            np.set_printoptions(precision=2)    
            print(precision_per_class)
            print(np.mean(precision_per_class))
            print(recall_per_class)
            print(np.mean(recall_per_class))
            print(f1_per_class)
            """
            return [np.mean(f1_per_class)]
        
        if self.average == 'weighted':
            support = tp + fn
            total_support = np.sum(support)
            if total_support == 0:
                return [0.0]
            
            f1_weighted = np.sum(f1_per_class * support) / total_support
            return [f1_weighted]

        return [0.0]

    def reinit(self):
        super(F1Metric, self).reinit()
        self.preds = []
        self.gts = []

def _print_f1(y_test,pr,re,f1,label_names=fg_names):
    data = pd.DataFrame(y_test)
    count = []
    for i in range(len(label_names)):
        count.append(int(sum(data[i])))

    # Order functional groups in order of highest to lowest F1-score.
    names = label_names#[label for _, label in sorted(zip(fs, label_names),reverse = True)]
    
    re_f1 = f1#sorted(fs,reverse = True)
    re_pr = []
    re_rc = []
    re_count = []
    for name in names:
        idx = label_names.index(name)
        # Precision.
        re_pr.append(pr[idx])
        # Recall.
        re_rc.append(re[idx])
        # Sample count.
        re_count.append(count[idx])

    result = pd.DataFrame(
        {'F-score': re_f1,
         'Precision': re_pr,
         'Recall': re_rc,
         'Frequency': re_count
        }, index=names)

    result.index.name = 'FGs'
    print(result.round(2))
    
    avg_f1 = sum(f1) / len(f1)
    avg_pr = sum(pr) / len(pr)
    avg_rc = sum(re) / len(re)
    
    print("\nAverage Scores:")
    print(f"F1-score: {avg_f1:.4f}")
    print(f"Precision: {avg_pr:.4f}")
    print(f"Recall: {avg_rc:.4f}\n")

