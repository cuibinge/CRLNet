import argparse
import os
import cv2
import numpy as np
import json
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
import imageio as imageio
import time
from model.CRLNet import CRLNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from utils.Dataset import MyDataset

from tqdm import tqdm
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_name = ['Seagrass bed',
              'Spartina alterniflora',
              'Reed',
              'Tamarix',
              'Tidal flat',
              'Sparse vegetation',
              'Sea',
              'Yellow River',
              'Pond',
              'Cloud']

def z_score_normal(image_data):
    B1, B2, B3, B4 = cv2.split(image_data)
    B_mean = np.mean(B1)
    B_std = np.std(B1)
    B1_normalization = ((B1 - B_mean) / B_std).astype('float32')

    B_mean = np.mean(B2)
    B_std = np.std(B2)
    B2_normalization = ((B2 - B_mean) / B_std).astype('float32')

    B_mean = np.mean(B3)
    B_std = np.std(B3)
    B3_normalization = ((B3 - B_mean) / B_std).astype('float32')

    B_mean = np.mean(B4)
    B_std = np.std(B4)
    B4_normalization = ((B4 - B_mean) / B_std).astype('float32')

    image_data = cv2.merge([B1_normalization, B2_normalization, B3_normalization, B4_normalization])
    return image_data

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def train(model, data_loader, optimizer1, criterion, args):
    model.train()
    total_start_time = time.time()
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        print('======================epoch:{}/{}========================='.format(epoch, args.epoch))
        epoch_loss = 0
        i = 0
        lambda_topo = 0.6 / (epoch + 1)

        for data in data_loader:
            i += 1
            img, label = data
            img, label = img.to(device), label.to(device)
            output, topo_loss = model(img.float())
            seg_loss = criterion(output, label.long())
            total_loss = seg_loss + lambda_topo * topo_loss
            optimizer1.zero_grad()
            total_loss.backward()
            optimizer1.step()
            epoch_loss += seg_loss.item()
            print('\r', 'step: ', i, 
                  ' seg_loss: {:.6f}'.format(seg_loss.item()), 
                  ' topo_loss: {:.6f}'.format(topo_loss.item()), 
                  ' total_loss: {:.6f}'.format(total_loss.item()), 
                  end='', flush=True)
        avg_seg_loss = epoch_loss / len(data_loader)
        print('\n avg_seg_loss:{:.6f} \n'.format(avg_seg_loss))

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        elapsed_time = epoch_end_time - total_start_time
        remaining_time = epoch_duration * (args.epoch - epoch - 1)

        elapsed_time_str = format_time(elapsed_time)
        remaining_time_str = format_time(remaining_time)
        epoch_duration_str = format_time(epoch_duration)

        print(f"Epoch {epoch}/{args.epoch} completed in {epoch_duration_str}.")
        print(f"Elapsed time: {elapsed_time_str} | Estimated remaining time: {remaining_time_str}")

        if epoch > 100 and epoch % 5 == 0:
            weight_name = 'epoch_' + str(epoch) + '_loss_' + str('{:.6f}'.format(avg_seg_loss)) + '.pt'
            torch.save(model.state_dict(), os.path.join(args.weights_path, weight_name))
            print('epoch: {} | loss: {:.6f} | Saving model... \n'.format(epoch, avg_seg_loss))

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    total_training_time_str = format_time(total_training_time)
    print(f"Training completed in {total_training_time_str}.")

def predict_and_postprocess(model, img_data, img_size, overlap=0.5, num_classes=10, device='cuda'):
    model.eval()
    row, col, dep = img_data.shape
    stride = int(img_size * (1 - overlap))
    padding_h = ((row - 1) // stride + 1) * stride + img_size - stride
    padding_w = ((col - 1) // stride + 1) * stride + img_size - stride
    padding_img = np.zeros((padding_h, padding_w, dep), dtype='float32')
    padding_img[:row, :col, :] = img_data[:row, :col, :]
    padding_prob_sum = np.zeros((padding_h, padding_w, num_classes), dtype='float32')
    count_map = np.zeros((padding_h, padding_w), dtype='float32')
    total_patches = ((padding_h - img_size) // stride + 1) * ((padding_w - img_size) // stride + 1)
    with tqdm(total=total_patches, desc="Predicting patches") as pbar:
        for i in range(0, padding_h - img_size + 1, stride):
            for j in range(0, padding_w - img_size + 1, stride):
                img_data_ = padding_img[i:i + img_size, j:j + img_size, :]
                img_data_ = img_data_[np.newaxis, :, :, :]
                img_data_ = np.transpose(img_data_, (0, 3, 1, 2))
                img_data_ = torch.from_numpy(img_data_).to(device)
                with torch.no_grad():
                    y_pre, topo_loss = model(img_data_)
                    y_prob = torch.squeeze(y_pre, dim=0).softmax(dim=0).cpu().numpy()
                    y_prob = np.transpose(y_prob, (1, 2, 0))
                padding_prob_sum[i:i + img_size, j:j + img_size, :] += y_prob[:img_size, :img_size, :]
                count_map[i:i + img_size, j:j + img_size] += 1
                pbar.update(1)
    avg_prob = padding_prob_sum / count_map[..., np.newaxis]
    prob_map = avg_prob[:row, :col]
    prob_map = prob_map.astype(np.float32)
    height, width = prob_map.shape[:2]
    d = dcrf.DenseCRF2D(width, height, num_classes)
    unary = -np.log(prob_map.transpose(2, 0, 1).reshape((num_classes, -1))).copy()
    d.setUnaryEnergy(unary)
    img_data_uint8 = (img_data * 255).astype(np.uint8) if img_data.dtype != np.uint8 else img_data
    d.addPairwiseGaussian(sxy=8, compat=8)
    pairwise_bilateral = create_pairwise_bilateral(sdims=(80, 80), schan=(13,), img=img_data_uint8, chdim=2)
    d.addPairwiseEnergy(pairwise_bilateral, compat=8)
    Q = d.inference(3)
    result = np.argmax(np.array(Q).reshape((num_classes, height, width)), axis=0).astype('uint8')
    return result

def calculation(y_label, y_pre, row, col):
    y_label = np.reshape(y_label, (row * col, 1))
    y_pre = np.reshape(y_pre, (row * col, 1))
    y_label.astype('float64')
    y_pre.astype('float64')
    precision = precision_score(y_label, y_pre, average=None)
    recall = recall_score(y_label, y_pre, average=None)
    f1 = f1_score(y_label, y_pre, average=None)
    kappa = cohen_kappa_score(y_label, y_pre)
    return precision, recall, f1, kappa

def estimate(y_label, y_pred, model_hdf5_name, class_name, dirname):
    acc = np.mean(np.equal(y_label, y_pred) + 0)
    print('=================================================================================================')
    print('The estimate result of {} are as follows:'.format(model_hdf5_name))
    print('The acc of {} is {}'.format(model_hdf5_name, acc))
    precision, recall, f1, kappa = calculation(y_label, y_pred, y_label.shape[0], y_label.shape[1])
    for i in range(len(class_name)):
        print('{}    F1: {:.5f}, Precision: {:.5f}, Recall: {:.5f}, kappa: {:.5f}'.format(class_name[i], f1[i],
                                                                                          precision[i], recall[i],
                                                                                          kappa))
    if len(f1) == len(class_name):
        result = {}
        for i in range(len(class_name)):
            result[class_name[i]] = []
            tmp = {}
            tmp['Recall'] = str(round(recall[i], 5))
            tmp['Precision'] = str(round(precision[i], 5))
            tmp['F1'] = str(round(f1[i], 5))
            result[class_name[i]].append(tmp)
        result['Model Name'] = [model_hdf5_name]
        result['Accuracy'] = str(round(acc, 5))
        result['kappa'] = str(kappa)
        txt_name = "epoch_" + model_hdf5_name.split("_")[1] + "_acc_" + str(round(acc, 5))
        with open(os.path.join(dirname, txt_name + '.txt'), 'a', encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False))
    else:
        print("======================================>Estimate error!===========================================")
    return acc

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Focal Loss for multi-class classification.

        Args:
            alpha (float): Weighting factor for balancing positive and negative examples. Default is 1.0.
            gamma (float): Focusing parameter to reduce the impact of well-classified examples. Default is 2.0.
            reduction (str): Specifies the reduction to apply to the output. 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Compute the Focal Loss.

        Args:
            logits: Predicted logits of shape (N, C, H, W) for multi-class segmentation or (N, C) for classification.
            targets: Ground truth of shape (N, H, W) for segmentation or (N) for classification, with class indices.
        
        Returns:
            Focal loss value.
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # Shape: (N, C, H, W)

        # Convert targets to one-hot encoding
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()  # Shape: (N, C, H, W)

        # Calculate the focal loss
        pt = (probs * targets_one_hot).sum(dim=1)  # Get probabilities of the true class. Shape: (N, H, W)
        log_pt = torch.log(pt + 1e-6)  # Avoid log(0) with a small epsilon

        focal_loss = -self.alpha * (1 - pt) ** self.gamma * log_pt

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--mode', type=str, default='pre_train', help='')
    parser.add_argument('--image_path', type=str, default=r"cut_train_images")
    parser.add_argument('--label_path', type=str, default=r"cut_train_labels")
    parser.add_argument('--weights_path', type=str, default='checkpoints', help='the path saving weights')
    parser.add_argument('--result_path', type=str, default='Result', help='the path saving result')
    parser.add_argument('--val_image_path', type=str, default=r'image.tif', help='val_imageset path')
    parser.add_argument('--val_label_path', type=str, default=r'label.tif', help='val_labelset path')
    parser.add_argument('--in_ch', type=int, default=4)
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epoch', type=int, default=200, help='train_epochs')
    args = parser.parse_args()

    criterion = FocalLoss(alpha=1.0, gamma=5.0, reduction='mean')

    mydataset = MyDataset(args.image_path, args.label_path)
    data_loader = DataLoader(dataset=mydataset, batch_size=args.batch, shuffle=True, pin_memory=True)
    print("The images in Dataset: %d" % len(mydataset))

    model = CRLNet(args).to(device)

    total = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    print("Number of parameters: %.2fM" % (total / 1e6))


    if not os.path.exists(args.weights_path):
        os.makedirs(args.weights_path)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    optimizer1 = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    train(model, data_loader, optimizer1, criterion, args)

    label_data = imageio.imread(args.val_label_path) - 1
    image_data = imageio.imread(args.val_image_path)

    image_data = z_score_normal(image_data)

    weights = os.listdir(args.weights_path)

    best_acc = 0

    for w in weights:
        model.load_state_dict(torch.load(os.path.join(args.weights_path, w), map_location=device))
        output = predict_and_postprocess(model, image_data, img_size=128, overlap=0.5, num_classes=10, device='cuda')
        acc = estimate(label_data, output, w, class_name, args.result_path)
        print('The acc of {} is {}'.format(w, acc))
        save_name = "epoch_" + w.split("_")[1] + "_acc_" + str(acc) + ".tif"
        imageio.imwrite(os.path.join(args.result_path, save_name), output)
        print("Sucessfully saved to " + os.path.join(args.result_path, save_name))
        print('=================================================================================================')

