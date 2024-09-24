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
import logging

logger = logging.getLogger(__name__)


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
    
    feature_dict = {}
    for data_iter_step, (samples, targets, info) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # pdb.set_trace()

        Pid = info['NicolaID']
        slices = info['Slice']
        timepoints = info['Timepoint']

        # for key, value in feature_dict.items():
        #     print(f'Feature dict: {key} - {value}')

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs, fea_vec = model(samples)
            # print(f"Fearure vector shape: {fea_vec.shape}")
            loss = criterion(outputs, targets)
            prediction_softmax = nn.Softmax(dim=1)(outputs)
            _, prediction_decode = torch.max(prediction_softmax, 1)

            # for i in range(len(Pid)):
            #     if Pid[i] not in feature_dict:
            #         feature_dict[Pid[i]] = {}
            #     if timepoints[i] not in feature_dict[Pid[i]]:
            #         feature_dict[Pid[i]][timepoints[i]] = {}
            #     if slices[i] not in feature_dict[Pid[i]][timepoints[i]]:
            #         feature_dict[Pid[i]][timepoints[i]][slices[i]] = {"correct": 0}

            # for key, value in feature_dict.items():
            #     print(f'Feature dict: {key} - {value}')

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
    slice_preds = {}
    class_dict = {}
    patient_dict = {"correct_count":0, "confidence": 0.0}
    total_slices = 0
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-2]
        info = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label=F.one_hot(target.to(torch.int64), num_classes=num_class)

        Pid = info['NicolaID']
        slices = info['Slice']
        timepoints = info['Timepoint']

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

            for i in range(len(slices)):
                # if key not in slice_preds:
                #     slice_preds[key] = {"correct": 0}
                # if prediction_decode[i].item() == true_label_decode[i].item():
                #     slice_preds[key]["correct"] += 1
                # if slices[i] not in slice_preds:
                #     slice_preds[slices[i]] = {"correct": 0, "occurence": 0}
                # if prediction_decode[i].item() == true_label_decode[i].item():
                #     slice_preds[slices[i]]["correct"] += 1
                # slice_preds[slices[i]]["occurence"] += 1
                # if slices[i] not in feature_dict:
                #     feature_dict[slices[i]] = {}
                if Pid[i] not in feature_dict:
                    feature_dict[Pid[i]] = {"True": true_label_decode[i].item()}
                if slices[i] not in feature_dict[Pid[i]]:
                    feature_dict[Pid[i]][slices[i]]= {"0":patient_dict, "1": patient_dict, "2": patient_dict}
                if prediction_decode[i].item() == 0:
                    feature_dict[Pid[i]][slices[i]]["0"]["correct_count"] += 1
                    feature_dict[Pid[i]][slices[i]]["0"]["confidence"] += prediction_softmax[i][0].item() / (len(slices) * len(data_loader))
                elif prediction_decode[i].item() == 1:
                    feature_dict[Pid[i]][slices[i]]["1"]["correct_count"] += 1
                    feature_dict[Pid[i]][slices[i]]["1"]["confidence"] += prediction_softmax[i][1].item() / (len(slices) * len(data_loader))
                elif prediction_decode[i].item() == 2:
                    feature_dict[Pid[i]][slices[i]]["2"]["correct_count"] += 1
                    feature_dict[Pid[i]][slices[i]]["2"]["confidence"] += prediction_softmax[i][2].item() / (len(slices) * len(data_loader))

            # logging.info(f'Slice Predictions: {slice_preds}')

            # for i in range(len(Pid)):
            #     total_slices += 1

            #     true_label = true_label_decode[i].item()
            #     pid = Pid[i]
            #     timepoint = timepoints[i]
            #     slice_ = slices[i]

            #     # Ensure nested dictionary structure exists
            #     if true_label not in feature_dict:
            #         feature_dict[true_label] = {}

            #     # if pid not in feature_dict[true_label]:
            #     #     feature_dict[true_label][pid] = {}

            #     # if timepoint not in feature_dict[true_label][pid]:
            #     #     feature_dict[true_label][pid][timepoint] = {}

            #     if slice_ not in feature_dict[true_label]:
            #         feature_dict[true_label][slice_] = {"correct": 0}

            #     # Increment 'correct' if prediction is accurate
            #     if prediction_decode[i].item() == true_label:
            #         feature_dict[true_label][slice_]["correct"] += 1

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
    # count = 0
    # logging.info(f'Feature Dictionary created for {mode}')
    # keys_patient = {}
    # Iterate through the feature_dict to find the max key and compare with "True" value
    # for slice_key, patient_dict in feature_dict.items():
    #     if slice_key not in keys_patient:
    #         keys_patient[slice_key] = []

    #     for patient, patients in patient_dict.items():
    #         # Find the key of the maximum value in the current patient's dictionary (among "0", "1", "2")
    #         valid_keys = {k: v for k, v in patients.items() if k in ["0", "1", "2"]}
            
    #         if valid_keys:
    #             max_key = max(valid_keys, key=lambda k: valid_keys[k])

    #             # Check if the max_key matches the value of the "True" key
    #             true_value = str(patients.get("True", None))  # Ensure "True" key's value is a string
    #             if max_key == true_value:  # Match max_key with the value of "True"
    #                 keys_patient[slice_key].append(max_key)
    #             else:
    #                 print(f'Incorrect prediction for patient {patient} in slice {slice_key}')
    #         else:
    #             print(f'No valid keys found for patient {patient} in slice {slice_key}')

    # Print the results
    # unique_counts = {}  # Store the unique count for each key
    # logging.info(f'Keys patient dictionary created for {mode}')
    logging.info(f'-------------------------------------------')
    # logging.info(f'Feature Dictionary created for {mode}')
    logging.info(f'Feature dict: {feature_dict}')
    logging.info(f'-------------------------------------------')
    max_count_ = {}
    # pdb.set_trace()
    patient_robust_labels = {}
    for pid, slices in feature_dict.items():
        # Initialize patient_robust_labels with default values
        true_label = str(feature_dict[pid].get("True", ""))
        patient_robust_labels[pid] = {
            "Best class": "", 
            "Best confidence": 0.0, 
            "Best correct count": 0, 
            "True": true_label, 
            "relevant slices": []
        }

        # Create a list to store the slices with their best class and corresponding correct_count and confidence
        slice_info_list = []

        # Iterate through slices and determine the best class for each slice
        for slice_key, class_data in slices.items():
            if slice_key == "True":  # Skip the "True" key
                continue

            # Find the best class (highest correct_count and confidence) for each slice
            best_class = None
            best_correct_count = 0
            best_confidence = 0.0

            for class_label, data in class_data.items():
                if data["correct_count"] > best_correct_count or (data["correct_count"] == best_correct_count and data["confidence"] > best_confidence):
                    best_class = class_label
                    best_correct_count = data["correct_count"]
                    best_confidence = data["confidence"]

            # Add the slice and its best class info to the list
            slice_info_list.append({
                "slice": slice_key,
                "best_class": best_class,
                "correct_count": best_correct_count,
                "confidence": best_confidence
            })

        # Sort the slice_info_list first by correct_count and then by confidence
        sorted_slices = sorted(slice_info_list, key=lambda x: (x["correct_count"], x["confidence"]), reverse=True)

        # Update the Best class, confidence, and correct count with the top slice from the sorted list
        if sorted_slices:
            patient_robust_labels[pid]["Best class"] = sorted_slices[0]["best_class"]
            patient_robust_labels[pid]["Best correct count"] = sorted_slices[0]["correct_count"]
            patient_robust_labels[pid]["Best confidence"] = sorted_slices[0]["confidence"]

        # Filter relevant slices: these are the ones where the best class matches the true label
        relevant_slices = [slice_info["slice"] for slice_info in sorted_slices if slice_info["best_class"] == true_label]
        
        # Store the relevant slices
        patient_robust_labels[pid]["relevant slices"] = relevant_slices[:10]

    logging.info(f'Patient Robust Labels: {patient_robust_labels}')
    logging.info(f'-------------------------------------------')
    for key, value in patient_robust_labels.items():
        print(f'Patient: {key}, Best Class: {value["Best class"]}, True: {value["True"]}, Relevant Slices: {value["relevant slices"]}')
        logging.info(f'Patient: {key}, Best Class: {value["Best class"]}, True: {value["True"]}, Relevant Slices: {value["relevant slices"]}')

    # compute pateint accuracy
    patient_count = 0
    for key, value in patient_robust_labels.items():
        if value["Best class"] == value["True"]:
            patient_count += 1

    patient_acc = patient_count / len(patient_robust_labels)
    logging.info(f'Number of patients: {len(patient_robust_labels)}')
    logging.info(f'Patient Accuracy: {patient_acc}')
    

    # Iterate through the feature_dict to find the top 5 slices with the highest "correct" values for each class
    # for cls_key, slices in feature_dict.items():  # Iterate over classes and their corresponding slices
    #     # Sort slices by the "correct" count in descending order
    #     sorted_slices = sorted(slices.items(), key=lambda x: x[1]["correct"], reverse=True)
        
    #     # Get the top 5 slices (or fewer if there are less than 5 slices)
    #     top_slices = sorted_slices[:5]
        
    #     # Store the top slices and their corresponding "correct" values for this class
    #     max_count_[cls_key] = [{"slice": slice_key, "correct": slice_info["correct"]} for slice_key, slice_info in top_slices]

    # # Print or use the max_count_ dictionary to see the result
    # for cls, top_slices_info in max_count_.items():
    #     print(f"Class: {cls}, Top Slices Info: {top_slices_info}")


    # logging.info(f'Feature dict: {feature_dict}')
    # max_count_ = {"max":0, "patient":"", "timepoint": "","slice": "", "probability": 0.0}
    # max_count_ = {}
    # for cls_key, cls_value in feature_dict.items():
    #     for key, value in cls_value.items():
    #         for timepoint, slices in value.items():
    #             max_count_[cls_key] = []
    #             max_pred = max(slices.values(), key=lambda x: x["correct"])
    #             for slice_key, patients_count in slices.items():
    #                 val = patients_count["correct"]
    #                 if val == max_pred:
    #                     max_count_[cls_key].append({"patient": key, "timepoint": timepoint, "slice": slice_key, "correct": val})
                        # norm_slice_key = int(slice_key) / len(slices.keys())
                        # # print(f'Normalised Slice: {norm_slice_key}')
                        # norm_slice_key = str(norm_slice_key)
                        # if norm_slice_key not in slice_preds:
                        #     slice_preds[norm_slice_key] = {"correct": 0, "probability": 0.0}
                        # slice_preds[norm_slice_key]["correct"] += val
                        # slice_preds[norm_slice_key]["probability"] = slice_preds[norm_slice_key]["correct"] / total_slices
                        # if slice_preds[norm_slice_key]["correct"] > max_count_["max"]:
                        #     max_count_["max"] = slice_preds[norm_slice_key]["correct"]
                        #     max_count_["patient"] = key
                        #     max_count_["timepoint"] = timepoint
                        #     max_count_['slice'] = slice_key
                        #     max_count_["probability"] = slice_preds[norm_slice_key]["correct"] / total_slices
                        # logging.info(f'Key: {key}, Timepoint: {timepoint}, Slice: {slice_key}, Patients: {patients_count}')
                # logging.info(f'Slice Predictions: {slice_preds}')
    
    # logging.info(f'Slice Predictions: {slice_preds}')
    # logging.info(f'Max count dict per class : {max_count_}')
    logging.info(f'-------------------------------------------')
    # logging.info(keys_patient)
    # for key, value in keys_patient.items():
    #     # Count unique patients (elements) per key
    #     print(f'Key: {key}, Patients: {value}, Mode: {mode}')
    #     unique_count = len(set(value))
    #     unique_counts[key] = unique_count  # Store the count
    #     # print(f'Key: {key}, Unique Patients: {unique_count}, Mode: {mode}')

    # logging.info(f'---------------------------------')
    # logging.info(f'Unique Counts: {unique_counts}')
    
    # if mode == 'val':
    #     import matplotlib.pyplot as plt

    #     # Sample data from max_count_
    #     classes = list(max_count_.keys())
    #     max_correct_values = []
    #     slice_keys = []
    #     class_labels = []

    #     for cls in classes:
    #         top_slices = max_count_[cls]  # Get the top slices for this class
    #         for slice_info in top_slices:
    #             max_correct_values.append(slice_info['correct'])
    #             slice_keys.append(slice_info['slice'])
    #             class_labels.append(cls)  # Add the class label for each slice

    #     # Create the figure and plot the bar chart
    #     plt.figure(figsize=(20, 10))

    #     # Create a colormap to assign different colors to each class
    #     colors = plt.cm.get_cmap('tab20', len(max_correct_values))

    #     # Plot the bars with different colors
    #     bars = plt.bar(range(len(max_correct_values)), max_correct_values, color=[colors(i) for i in range(len(max_correct_values))])

    #     # Add slice key and class information inside each bar
    #     for bar, slice_key, cls, correct in zip(bars, slice_keys, class_labels, max_correct_values):
    #         plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
    #                 f'Slice: {slice_key}\nCorrect: {correct}', ha='center', va='center', color='black', fontsize=8, rotation=90)

    #     # Normalize the Y-axis scale to handle large differences between values
    #     plt.yscale('log')  # Log scale for better visualization of bars with low/high values
    #     plt.ylim(1, max(max_correct_values) * 1.5)  # Adjust Y-limit to give some padding on top

    #     # Set x-ticks to the class labels for better visualization
    #     plt.xticks(range(len(max_correct_values)), class_labels, rotation=90)

    #     # Add labels and title
    #     plt.xlabel('Class and Slices')
    #     plt.ylabel('Correct Predictions (log scale)')
    #     plt.title('Top 5 Correct Predictions per Class with Corresponding Slices')

    #     # Display the plot
    #     plt.tight_layout()
    #     plt.savefig(task+'_top5_val_max_correct_predictions_per_class.jpg', dpi=150, bbox_inches='tight')

        # import matplotlib.pyplot as plt
        # keys = list(slice_preds.keys())
        # correct_values = [v["correct"] for v in slice_preds.values()]

        # # Create the plot
        # plt.figure(figsize=(10, 6))
        # print(f'Plotted Graph for {mode}')
        # plt.plot(keys, correct_values, marker='o', color='b', linestyle='-', markersize=5)
        
        # # Adding labels and title
        # plt.xlabel('Slice')
        # plt.ylabel('Correct Predictions')
        # plt.title('Correct predictions frequency (Validation set)')
        
        # # Show only the first and last x-ticks (min and max)
        # plt.xticks([0, len(keys)-1], [keys[0], keys[-1]])
        # plt.tight_layout()
        # plt.savefig(task+'_slice_count_val.jpg', dpi=600, bbox_inches='tight')
            
        # Calculate the frequency of each element in 'Pred'
        # pred_counts = Counter(value['Pred'])
        # # print(f'Pred counts: {pred_counts}')
        # # Get all elements sorted by their frequency in descending order
        # sorted_pred = [item for item, count in pred_counts.most_common()]
        # # Store the sorted list back in the dictionary
        # feature_dict[key]['Pred'] = sorted_pred
        # # print(f'Sorted Pred: {sorted_pred}')
        
        # # Calculate the frequency of each element in 'True'
        # true_counts = Counter(value['True'])
        # # Get all elements sorted by their frequency in descending order
        # sorted_true = [item for item, count in true_counts.most_common()]
        # # Store the sorted list back in the dictionary
        # feature_dict[key]['True'] = sorted_true

        # # Compare the top two from Pred with the topmost from True
        # top_pred = sorted_pred[:2]  # Get the top 2 predictions
        # top_true = sorted_true[0]   # Get the top 1 true label

        # # print(f'Top Pred: {top_pred} Top True: {top_true}')

        # # Check if the top true label is in the top two predictions
        # if top_true in top_pred:
        #     count += 1

    # for key, value in feature_dict.items():
    #     feature_dict[key]['Pred'] = max(set(value['Pred']), key=value['Pred'].count)
    #     feature_dict[key]['True'] = max(set(value['True']), key=value['True'].count)
    #     print(f'Feature dict: {key} - {feature_dict[key]}')
    #     count += feature_dict[key]['True'] == feature_dict[key]['Pred']
    # if mode == 'val':
    #     print(f'Validation Accuracy per patient: {count/len(feature_dict) * 100}%')
    print('Sklearn Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} MCC: {:.4f}'.format(acc, auc_roc, auc_pr, F1, mcc)) 
    
    results_path = task + '_metrics_{}.csv'.format(mode)
    # Ensure the directory for the results file exists
    directory = os.path.dirname(results_path)
    if not os.path.exists(directory) and directory != '':
        os.makedirs(directory)

    # Check if the file exists
    file_exists = os.path.isfile(results_path)
    # If the file exists, check if it is empty
    file_empty = os.stat(results_path).st_size == 0 if file_exists else True
    # Define the header for the CSV file
    header = ['acc', 'sensitivity', 'specificity', 'precision', 'auc_roc', 'auc_pr', 'F1', 'mcc', 'loss']
    # Open the file in append mode and add data
    with open(results_path, mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        # If the file does not exist or is empty, write the header first
        if not file_exists or file_empty:
            wf.writerow(header)
        # Data to write to the CSV file
        data2 = [[acc, sensitivity, specificity, precision, auc_roc, auc_pr, F1, mcc, metric_logger.loss]]
        # Write the data to the CSV file
        for row in data2:
            wf.writerow(row)
                
    
    if mode=='test':
        pass
        # import matplotlib.pyplot as plt

        # # Sample data from max_count_
        # classes = list(max_count_.keys())
        # max_correct_values = []
        # slice_keys = []
        # class_labels = []

        # for cls in classes:
        #     top_slices = max_count_[cls]  # Get the top slices for this class
        #     for slice_info in top_slices:
        #         max_correct_values.append(slice_info['correct'])
        #         slice_keys.append(slice_info['slice'])
        #         class_labels.append(cls)  # Add the class label for each slice

        # # Create the figure and plot the bar chart
        # plt.figure(figsize=(20, 10))

        # # Create a colormap to assign different colors to each class
        # colors = plt.cm.get_cmap('tab20', len(max_correct_values))

        # # Plot the bars with different colors
        # bars = plt.bar(range(len(max_correct_values)), max_correct_values, color=[colors(i) for i in range(len(max_correct_values))])

        # # Add slice key and class information inside each bar
        # for bar, slice_key, cls, correct in zip(bars, slice_keys, class_labels, max_correct_values):
        #     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
        #             f'Slice: {slice_key}\nCorrect: {correct}', ha='center', va='center', color='black', fontsize=8, rotation=90)

        # # Normalize the Y-axis scale to handle large differences between values
        # plt.yscale('log')  # Log scale for better visualization of bars with low/high values
        # plt.ylim(1, max(max_correct_values) * 1.5)  # Adjust Y-limit to give some padding on top

        # # Set x-ticks to the class labels for better visualization
        # plt.xticks(range(len(max_correct_values)), class_labels, rotation=90)

        # # Add labels and title
        # plt.xlabel('Class')
        # plt.ylabel('Correct Predictions (log scale)')
        # plt.title('Top 5 Correct Predictions per Class with Corresponding Slices')

        # # Display the plot
        # plt.tight_layout()
        # plt.savefig(task+'_top5_test_max_correct_predictions_per_class.jpg', dpi=150, bbox_inches='tight')

        # print(f'Test Accuracy per patient: {count/len(feature_dict) * 100}%')
        # cm = confusion_matrix(true_label_decode_list, prediction_decode_list)
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
        # plt.xlabel('Predicted Label')
        # plt.ylabel('True Label')
        # plt.title('Confusion Matrix Validation')
        # # cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
        # plt.savefig(task+'confusion_matrix_test.jpg',dpi=600,bbox_inches ='tight')

        # slice patient count graph
        # import matplotlib.pyplot as plt
        # keys = list(slice_preds.keys())
        # correct_values = [v["correct"] for v in slice_preds.values()]
        # # Plotting the values
        # plt.plot(keys, correct_values, marker='o')
        # # Adding labels and title
        # plt.xlabel('Slice')
        # plt.ylabel('Correct Predictions')
        # plt.title('Correct predictions frequency (Validation set)')
        # # Show only a few equidistant x-ticks: minimum, maximum, and some in between
        # num_ticks = 10  # Number of x-ticks you want to display
        # tick_positions = np.linspace(0, len(keys) - 1, num_ticks, dtype=int)  # Generate equidistant indices
        # plt.xticks(tick_positions, [keys[i] for i in tick_positions], rotation=25)
        # # Show grid and plot
        # plt.grid(True)
        # plt.savefig(task+'_slice_count_test.jpg', dpi=600, bbox_inches='tight')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},auc_roc

