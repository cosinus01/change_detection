import cv2
import numpy as np
from tqdm.notebook import tqdm
from q_metrics import accuracy, f1_score, IoULoss

def load_data(files, path, flag=-1):
    data = []
    for f in files:
        image = cv2.imread(path + f, flag)
        if flag == 1:
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data.append(image)
    return np.array(data)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def make_data_generator(files_list, path, batch_size=16):
  n = len(files_list)
  for files in tqdm(batch(files_list, batch_size), total=n // batch_size):
      images_old = load_data(files, path + "/old/", 1)
      images_new = load_data(files, path + "/new/", 1)
      labels = load_data(files, path + "/label/", -1)

      yield (np.moveaxis(images_old, 3, 1) / 255,
             np.moveaxis(images_new, 3, 1) / 255, 
             labels)

def priory_metrics(files, path, batch_size=4):
  acc_list_rand = []
  f1_list_rand = []
  iou_list_rand = []
  acc_list_zero = []
  f1_list_zero = []
  iou_list_zero = []

  dataset = make_data_generator(files, path=path, batch_size=batch_size)
  for _, _, masks in dataset:
    y_pred_rand = np.random.rand(masks.shape[0], masks.shape[1], masks.shape[2])
    y_pred_zero = np.zeros_like(masks)
    
    y_pred_labels_rand = y_pred_rand > 0.5
    y_pred_labels_zero = y_pred_zero > 0.5

    acc_rand = accuracy(masks, y_pred_labels_rand)
    acc_zero = accuracy(masks, y_pred_labels_zero)
    f1_sc_rand = f1_score(masks.flatten(), y_pred_labels_rand.flatten())
    f1_sc_zero = f1_score(masks.flatten(), y_pred_labels_zero.flatten())
    iou_score_rand = IoULoss(masks, y_pred_labels_rand)
    iou_score_zero = IoULoss(masks, y_pred_labels_zero)
    
    acc_list_rand.append(acc_rand)
    f1_list_rand.append(f1_sc_rand)
    iou_list_rand.append(iou_score_rand)
    acc_list_zero.append(acc_zero)
    f1_list_zero.append(f1_sc_zero)
    iou_list_zero.append(iou_score_zero)
  return {"acc rand": np.mean(acc_list_rand), "acc zero": np.mean(acc_list_zero),
          "f1 rand": np.mean(f1_list_rand), "f1 zero":  np.mean(f1_list_zero),
          "IOU rand": np.mean(iou_list_rand), "IOU zero":  np.mean(iou_list_zero)}
  