import torch
import numpy as np
from tqdm.notebook import tqdm
from utils import make_data_generator
from q_metrics import get_optimal_threshold

def train(model, files_train, files_valid, n_epochs, loss_fn, batch_size=4, name_model="model_CD", scheduler_=True):
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
   
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    print("start training")   
    for epoch_index in tqdm(range(n_epochs)):
      model.train()
      print('Epoch: ' + str(epoch_index) + ' of ' + str(n_epochs))
      DATA_TRAIN = make_data_generator(files_train, path="train", batch_size=batch_size)
      DATA_VALID = make_data_generator(files_valid, path="test", batch_size=batch_size)
      for images_old, images_new, label in DATA_TRAIN:
        images_old_tensor = torch.tensor(images_old, dtype=torch.float, device="cuda")
        images_new_tensor = torch.tensor(images_new, dtype=torch.float, device="cuda")
        label_tensor = torch.tensor(label.astype(float), dtype=torch.long, device="cuda")
        optimizer.zero_grad()
        output = model(images_old_tensor, images_new_tensor)
        loss = loss_fn(output, label_tensor)
        loss.backward()
        optimizer.step()
            
      if scheduler_:
        scheduler.step()
      model.eval()
      
      y_pred = np.array([])
      y_true = np.array([])
      for images_old, images_new, label in DATA_VALID:
        images_old_tensor = torch.tensor(images_old, dtype=torch.float, device="cuda")
        images_new_tensor = torch.tensor(images_new, dtype=torch.float, device="cuda")
        output = model(images_old_tensor, images_new_tensor)
        output_numpy = output.cpu().detach().numpy()
        output_numpy_exp = np.exp(output_numpy)
        output_numpy_exp_1d = output_numpy_exp[:, 1, :, :]
        y_true = np.concatenate((y_true, label.flatten()))
        y_pred = np.concatenate((y_pred, output_numpy_exp_1d.flatten()))
        # output_numpy_labels = np.argmax(output_numpy, axis=1)
      print("calculating accuracy...")
      tr_acc, acc = get_optimal_threshold(y_true, y_pred, metric="accuracy")
      print("calculating f1 score...")
      tr_f1, f1 = get_optimal_threshold(y_true, y_pred, metric="f1_score")
      print("calculating IOU...")
      tr_iou, iou = get_optimal_threshold(y_true, y_pred, metric="IOU")
      print('Epoch: {}, accuracy={:.2f}, f1={:.2f}, iou={:.2f}'.format(epoch_index,
                                                                       acc,
                                                                       f1,
                                                                       iou))
      print('Epoch: {}, tr_accuracy={:.2f}, tr_f1={:.2f}, tr_iou={:.2f}'.format(epoch_index,
                                                                       tr_acc,
                                                                       tr_f1,
                                                                       tr_iou))
      with open(f"drive/MyDrive/ncOMZ/hit/change_detection/{name_model}_epoch_{epoch_index}.pth", "wb") as fp:
        torch.save(model.state_dict(), fp)