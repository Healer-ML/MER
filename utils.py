import numpy as np
import torch
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 这里可以传入一个类别权重列表
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        # 使用加权交叉熵损失并关闭默认的 reduction
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)  # 获取预测的概率

        # 如果 alpha 是一个张量，将其作为每个类别的权重
        if self.alpha is not None:
            at = self.alpha[targets]  # 使用每个类别的 alpha
            focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            # alpha 是单个值
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()
def confusionMatrix(gt, pred, show=False):
    # Ensure confusion_matrix returns a full matrix even if only one class is present
    cm = confusion_matrix(gt, pred, labels=[0, 1])

    # Flatten the confusion matrix
    TN_recog, FP_recog, FN_recog, TP_recog = cm.ravel()

    # Calculate F1-score, Average Recall, Average Precision
    f1_score = (2 * TP_recog) / (2 * TP_recog + FP_recog + FN_recog) if (2 * TP_recog + FP_recog + FN_recog) > 0 else 0
    average_recall = TP_recog / (TP_recog + FN_recog) if (TP_recog + FN_recog) > 0 else 0
    average_precision = TP_recog / (TP_recog + FP_recog) if (TP_recog + FP_recog) > 0 else 0

    num_samples = len([x for x in gt if x == 1])

    if show:
        print(f"Confusion Matrix: TP: {TP_recog}, FP: {FP_recog}, FN: {FN_recog}, TN: {TN_recog}")

    return f1_score, average_recall, TP_recog, FP_recog, FN_recog, TN_recog, num_samples, average_precision, average_recall

def recognition_evaluation(final_gt, final_pred, show=False,label_dict=None):
    precision_list = []
    recall_list = []
    f1_list = []
    ar_list = []
    TP_all = 0
    FP_all = 0
    FN_all = 0
    TN_all = 0
    try:
        for emotion, emotion_index in label_dict.items():
            # Ignore the 'neutral' emotion
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog, TP_recog, FP_recog, FN_recog, TN_recog, num_samples, precision_recog, recall_recog = confusionMatrix(
                    gt_recog, pred_recog, show)

                if show:
                    print(f"{emotion.title()} Emotion:")
                    print(f"TP: {TP_recog}, FP: {FP_recog}, FN: {FN_recog}, TN: {TN_recog}")

                # Accumulate TP, FP, FN, TN across npy emotions
                TP_all += TP_recog
                FP_all += FP_recog
                FN_all += FN_recog
                TN_all += TN_recog

                # Append precision and recall for each emotion
                precision_list.append(precision_recog)
                recall_list.append(recall_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                print(f"Error in recognition evaluation for emotion {emotion}: {e}")
                pass

        # Handle nan values in the lists
        precision_list = [0 if np.isnan(x) else x for x in precision_list]
        recall_list = [0 if np.isnan(x) else x for x in recall_list]

        # Calculate mean precision and recall
        precision_all = np.mean(precision_list) if len(precision_list) > 0 else 0
        recall_all = np.mean(recall_list) if len(recall_list) > 0 else 0

        # Calculate overall F1-score
        f1_all = (2 * precision_all * recall_all) / (precision_all + recall_all) if (
                                                                                                precision_all + recall_all) > 0 else 0

        # Calculate UF1 and UAR
        UF1 = np.mean(f1_list) if len(f1_list) > 0 else 0
        UAR = np.mean(ar_list) if len(ar_list) > 0 else 0

        if show:
            print('------ After adding ------')
            print(f'TP: {TP_all}, FP: {FP_all}, FN: {FN_all}, TN: {TN_all}')
            print(f'Precision: {round(precision_all, 4)}, Recall: {round(recall_all, 4)}')

        return np.nan_to_num(UF1), np.nan_to_num(UAR), np.nan_to_num(f1_all)  # Return 0 if nan

    except Exception as e:
        print(f"Error in overall evaluation: {e}")
        return 0, 0, 0
def recognition_eval(data,label_dict):
    all_preds = []
    all_truths = []
    label = list(label_dict.values())
    for subject, values in data.items():
        all_preds.extend(values['pred'])
        all_truths.extend(values['truth'])
    uar = recall_score(all_truths, all_preds, labels=label, average='macro')
    uf1 = f1_score(all_truths, all_preds, labels=label, average='macro')
    accuracy = accuracy_score(all_truths, all_preds)
    print(f"Finall-Metrics*****UAR: {uar:.4f},UF1: {uf1:.4f},Accuracy: {accuracy:.4f}****")
