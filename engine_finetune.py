# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math
import sys
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score,multilabel_confusion_matrix, confusion_matrix
from pycm import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import matthews_corrcoef, f1_score
from collections import Counter

import pdb



def misc_measures(confusion_matrix):
    
    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []
    
    for i in range(1, confusion_matrix.shape[0]):
        cm1=confusion_matrix[i]
        acc.append(1.*(cm1[0,0]+cm1[1,1])/np.sum(cm1))
        sensitivity_ = 1.*cm1[1,1]/(cm1[1,0]+cm1[1,1])
        sensitivity.append(sensitivity_)
        specificity_ = 1.*cm1[0,0]/(cm1[0,1]+cm1[0,0])
        specificity.append(specificity_)
        precision_ = 1.*cm1[1,1]/(cm1[1,1]+cm1[0,1])
        precision.append(precision_)
        G.append(np.sqrt(sensitivity_*specificity_))
        f1_score = 2*precision_*sensitivity_/(precision_+sensitivity_)
        F1_score_2.append(f1_score)
        mcc = (cm1[0,0]*cm1[1,1]-cm1[0,1]*cm1[1,0])/np.sqrt((cm1[0,0]+cm1[0,1])*(cm1[0,0]+cm1[1,0])*(cm1[1,1]+cm1[1,0])*(cm1[1,1]+cm1[0,1]))
        if np.isnan(mcc) or np.isnan(f1_score):
            print('MCC or F1 is nan')
            print(cm1)
        mcc_.append(mcc)
        
    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()
    
    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_





def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    # pdb.set_trace()
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    #     print(targets)

    for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # print(targets)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs, fea_vec = model(samples)
            # print(f"Fearure vector shape: {fea_vec.shape}")
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(data_loader, model, device, task, epoch, mode, num_class):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if not os.path.exists(task):
        os.makedirs(task, exist_ok=True)

    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    
    # switch to evaluation mode
    model.eval()
    feature_dict = {}
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-2]
        Pid = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label=F.one_hot(target.to(torch.int64), num_classes=num_class)

        # compute output
        with torch.cuda.amp.autocast():
            output, fea_vec = model(images)
            loss = criterion(output, target)
            # print(f"Fearure vector shape: {fea_vec.shape}")
            prediction_softmax = nn.Softmax(dim=1)(output)
            _,prediction_decode = torch.max(prediction_softmax, 1)
            _,true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            true_label_onehot_list.extend(true_label.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())

            for i in range(len(Pid)):
                if Pid[i] not in feature_dict:
                    feature_dict[Pid[i]] = {"Pred": [], "True": []}
                feature_dict[Pid[i]]["Pred"].append(prediction_decode[i].item())
                feature_dict[Pid[i]]["True"].append(target[i].item())

        acc1,_ = accuracy(output, target, topk=(1,2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    confusion_matrix_1 = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list,labels=[i for i in range(num_class)])
    acc, sensitivity, specificity, precision, G, _, _ = misc_measures(confusion_matrix_1)
    mcc = matthews_corrcoef(true_label_decode_list, prediction_decode_list)
    F1 = f1_score(true_label_decode_list, prediction_decode_list,average='macro')
    
    auc_roc = roc_auc_score(true_label_onehot_list, prediction_list,multi_class='ovr',average='macro')
    auc_pr = average_precision_score(true_label_onehot_list, prediction_list,average='macro')          
            
    metric_logger.synchronize_between_processes()

    # plot t-SNE of feature dict
    # Flatten the feature dictionary into arrays for t-SN
    # with open(task+'_feature_dict.csv', 'w') as f:
    #     for key, value in feature_dict.items():
    #         f.write("%s,%s\n"%(key,value))
    count = 0
    for key, value in feature_dict.items():
        # Calculate the frequency of each element in 'Pred'
        pred_counts = Counter(value['Pred'])
        print(f'Pred counts: {pred_counts}')
        # Get all elements sorted by their frequency in descending order
        sorted_pred = [item for item, count in pred_counts.most_common()]
        # Store the sorted list back in the dictionary
        feature_dict[key]['Pred'] = sorted_pred
        # print(f'Sorted Pred: {sorted_pred}')
        
        # Calculate the frequency of each element in 'True'
        true_counts = Counter(value['True'])
        # Get all elements sorted by their frequency in descending order
        sorted_true = [item for item, count in true_counts.most_common()]
        # Store the sorted list back in the dictionary
        feature_dict[key]['True'] = sorted_true

        # Compare the top two from Pred with the topmost from True
        top_pred = sorted_pred[:2]  # Get the top 2 predictions
        top_true = sorted_true[0]   # Get the top 1 true label

        print(f'Top Pred: {top_pred} Top True: {top_true}')

        # Check if the top true label is in the top two predictions
        if top_true in top_pred:
            count += 1

    # for key, value in feature_dict.items():
    #     feature_dict[key]['Pred'] = max(set(value['Pred']), key=value['Pred'].count)
    #     feature_dict[key]['True'] = max(set(value['True']), key=value['True'].count)
    #     print(f'Feature dict: {key} - {feature_dict[key]}')
    #     count += feature_dict[key]['True'] == feature_dict[key]['Pred']
    if mode == 'val':
        print(f'Validation Accuracy per patient: {count/len(feature_dict) * 100}%')
    print('Sklearn Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} MCC: {:.4f}'.format(acc, auc_roc, auc_pr, F1, mcc)) 
    results_path = task+'_metrics_{}.csv'.format(mode)
    if mode == 'val':
        cm = confusion_matrix(true_label_decode_list, prediction_decode_list)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix Validation')
        # cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
        plt.savefig(task+'confusion_matrix_val.jpg',dpi=600,bbox_inches ='tight')
    # Check file exists and is empty
    file_exists = os.path.isfile(results_path)
    file_empty = os.stat(results_path).st_size == 0 if file_exists else True
    header = ['acc', 'sensitivity', 'specificity', 'precision', 'auc_roc', 'auc_pr', 'F1', 'mcc', 'loss']
    with open(results_path,mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        # If file is empty
        if file_empty:
            wf.writerow(header)
        data2=[[acc,sensitivity,specificity,precision,auc_roc,auc_pr,F1,mcc,metric_logger.loss]]
        for i in data2:
            wf.writerow(i)
            
    
    if mode=='test':
        print(f'Test Accuracy per patient: {count/len(feature_dict) * 100}%')
        cm = confusion_matrix(true_label_decode_list, prediction_decode_list)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix Validation')
        # cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
        plt.savefig(task+'confusion_matrix_test.jpg',dpi=600,bbox_inches ='tight')

        # features = []
        # labels = []

        # for label, feature_list in feature_dict.items():
        #     for feature in feature_list:
        #         features.append(feature.cpu().detach().numpy())
        #         labels.append(label)

        # features = np.concatenate(features)  # Convert list of arrays to a single array

        # # Apply t-SNE to reduce feature dimensions
        # tsne = TSNE(n_components=2, random_state=42)
        # reduced_features = tsne.fit_transform(features)

        # # Plot the t-SNE result
        # plt.figure(figsize=(10, 8))
        # scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='tab10', alpha=0.7)
        # plt.colorbar(scatter, ticks=range(num_class))
        # plt.title("t-SNE of Feature Vectors")
        # plt.xlabel("t-SNE component 1")
        # plt.ylabel("t-SNE component 2")
        # plt.savefig(task+'t-SNE.jpg',dpi=600,bbox_inches ='tight')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},auc_roc

