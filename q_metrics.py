import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import (roc_curve, precision_recall_curve, auc, 
                             f1_score,  confusion_matrix, precision_score,
                             recall_score)

def accuracy(y_true, y_pred):
  N = np.prod(y_true.shape)
  return (y_pred == y_true).sum() / N

def IoULoss(y_true, y_pred):
    #flatten label and prediction tensors
    smooth = 1e-6
    inputs = y_pred
    targets = y_true
    
    intersection = np.sum(targets * inputs)
    # print(intersection)
    total = np.sum(targets) + np.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return IoU

def get_optimal_threshold(y_true, y_pred, metric="accuracy"):
  tr = np.arange(0, 1, 0.01)
  
  if metric == "accuracy":
    acc = []
    for t in tqdm(tr):
      acc1 = accuracy(y_true, y_pred > t)
      acc.append(acc1)
    
    best_tr = tr[np.argmax(acc)]
    best_metric = np.max(acc)

  if metric == "f1_score":
    precision, recall, tr = precision_recall_curve(y_true, y_pred)
    # best_tr = tr[np.argmax(((1+1.5**2)*precision*recall/((1.5**2)*precision + recall)))]
    best_tr = tr[np.argmax((2 * precision * recall / (precision + recall + 1e-8)))]
    best_metric = np.max(2 * precision * recall / (precision + recall + 1e-8))
  if metric == "IOU":
    iou = []
    for t in tqdm(tr):
      iou1 = IoULoss(y_true, y_pred > t)
      iou.append(iou1)
    
    best_tr = tr[np.argmax(iou)]
    best_metric = np.max(iou)
  
  return best_tr, best_metric