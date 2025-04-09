
import argparse
import json
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from collections import Counter
from label import process_dataset
from tools import FocalLoss, recognition_eval,recognition_evaluation
from model.model import MultiBranchNet
import pandas as pd
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def parse_args():
    parser = argparse.ArgumentParser(description="LOSO Cross-Validation for Micro-Expression Recognition")
    parser.add_argument('--image_folder', type=str, default='E:\\code\\ME\\ME_new\\flow_feature', help='Path to the image folder')
    parser.add_argument('--save_path', type=str, default='log\exp_finall', help='Path to save the models')
    parser.add_argument('--xls_path', type=str, default='three.xlsx', help='Path to Excel file')
    parser.add_argument('--dataset', type=str, default='DFME', help='Dataset name')
    parser.add_argument('--STSNet', type=bool, default=False, help='Structured input processing')
    parser.add_argument('--all', type=bool, default=False, help='Include all data')
    parser.add_argument('--cls', type=int, default=7, help='Number of classes for classification emotion')
    parser.add_argument('--load_pretrain', action='store_true', help='Load pre-trained weights if available')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--label_path', type=str, default='label_dict/',help='Label dictionary for expression types')
    parser.add_argument('--train', default=False, help='Train the model if set, otherwise only evaluate')
    return parser.parse_args()
def prepare_data(args):

    if args.all:
        args.dataset = 'three'
        args.image_folder = args.image_folder + f'\\STSNet\\'
    else:
        args.image_folder = args.image_folder + f'\\{args.dataset}\\flow\\'
    save_dir = os.path.join(args.save_path, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_excel(args.xls_path)
    results = []
    if args.STSNet:
        for _, row in df.iterrows():
            dataset, sub, filename_o, imagename, label = row['dataset'], str(row['sub']).zfill(3), row['filename_o'], row['imagename'], row['label']
            image_path = os.path.join(args.image_folder, imagename)
            if os.path.exists(image_path):
                results.append({'dataset': dataset, 'path': image_path, 'sub': sub, 'labels': label})
            else:
                print(f'{imagename} does not exist')
    else:
        pattern = get_dataset_pattern(args.dataset)
        for imagename in os.listdir(args.image_folder):
            if imagename.endswith('.png'):
                image_path = os.path.join(args.image_folder, imagename)
                match = re.match(pattern, imagename)
                if match:
                    sub, label = match.group(1), match.group(2)
                    results.append({'dataset': args.dataset, 'path': image_path, 'sub': sub, 'labels': label})
                else:
                    print(f"Filename '{imagename}' does not match the expected pattern")
    return results, save_dir
def get_dataset_pattern(dataset_name):
    patterns = {
        "casme2": r'(sub\d+)_.*_(\d+)\.png',
        "CAS(ME)^3": r'(spNO\.\d+)_.*_(\d+)\.png',
        "smic": r'^(s\d+).*_(\d+)\.png$',
        "DFME": r'^(sub\d+).*_(\d+)\.png$',
        "samm": r'(\d{3})_.*_(\d)\.png$'
    }
    return patterns.get(dataset_name, r'.*')
class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([transforms.ToTensor()])
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        image = self.load_image(data['path'])
        image = self.transform(image)
        label = int(data['labels'])
        return image, label
    @staticmethod
    def load_image(path):
        return Image.open(path).convert('RGB')

def calculate_weights(label_counts, label_map):
    class_counts = [label_counts.get(label, 0) for label in label_map.values()]
    return [1.0 / count if count > 0 else 0.0 for count in class_counts]

if __name__ == "__main__":
    args = parse_args()
    dataset, save_dir = prepare_data(args)
    if args.cls == 3:
        with open(f'{args.label_path}cls.json', 'r') as json_file:
            label_dict = json.load(json_file)
        print('label_dict:',label_dict)
    elif args.dataset=='DFME':
        with open(f'{args.label_path}DFME.json', 'r') as json_file:
            label_dict = json.load(json_file)
        print('label_dict:',label_dict)
    elif args.cls == 5:
        if args.dataset == 'casme2':
            file_path='/media/yu/data/ME/camse2/CASME2-coding-20140508.xlsx'
            dataset = process_dataset(args.dataset, args.cls,dataset, file_path)
            with open(f'{args.label_path}casme2_cls5.json', 'r') as json_file:
                label_dict = json.load(json_file)
            print('label_dict:', label_dict)
        elif args.dataset == 'samm':
            file_path='E:\code\ME_new\log\samm.xlsx'
            dataset = process_dataset(args.dataset, args.cls,dataset, file_path)
            with open(f'{args.label_path}samm_cls5.json', 'r') as json_file:
                label_dict = json.load(json_file)
            print('label_dict:', label_dict)
    elif args.dataset == 'CAS(ME)^3' and args.cls == 7:
            file_path = 'E:\data\casme^3\cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx'
            dataset = process_dataset(args.dataset,  args.cls,dataset, file_path)
            with open(f'{args.label_path}casme3_cls7.json', 'r') as json_file:
                label_dict = json.load(json_file)
            print('label_dict:', label_dict)
    elif args.dataset == 'CAS(ME)^3' and args.cls == 4:
        file_path = 'E:\data\casme^3\cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx'
        dataset = process_dataset(args.dataset, args.cls, dataset, file_path)
        with open(f'{args.label_path}casme3_cls4.json', 'r') as json_file:
            label_dict = json.load(json_file)
        print('label_dict:', label_dict)
    label_map = {value: key for key, value in label_dict.items()}
    subs = set(data['sub'] for data in dataset)
    print("dataset:", args.dataset)
    print("Number of data:",len(dataset))
    full_dataset = CustomDataset(dataset)
    label_counts = Counter(int(data['labels']) for data in dataset)
    for label_name, label_index in label_map.items():
        count = label_counts.get(label_name, 0)
        print(f"标签 {label_index}: {count} 个样本")
    weights = calculate_weights(label_counts, label_dict)
    print("weighte:",weights)
    weights = torch.tensor(weights).to(device)
    loss_fn = FocalLoss(alpha=weights, gamma=2)
    total_pred = []
    total_gt = []
    all_accuracy_dict = {}
    all_features = []
    gls=[]
    bls=[]
    brs=[]
    tls=[]
    trs=[]
    all_preds = []
    all_labels = []
    all_imgs=[]
    for n_subname in subs:
        print(f'Processing subject {n_subname} as test set')
        train_indices = [i for i, data in enumerate(dataset) if data['sub'] != n_subname]
        test_indices = [i for i, data in enumerate(dataset) if data['sub'] == n_subname]
        train_subset = Subset(full_dataset, train_indices)
        test_subset = Subset(full_dataset, test_indices)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
        model = MultiBranchNet(args)
        model = model.to(device)
        print("Parameter Count: %d" % count_parameters(model))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_accuracy_for_each_subject = 0.0
        best_each_subject_pred = []
        weight_path = save_dir + n_subname + '.pth'

        if args.train:
            print('Mode: Training')
        else:
            if os.path.exists(weight_path):
                model.load_state_dict(torch.load(weight_path))
                # model.load_state_dict(torch.load(weight_path, map_location=torch.device('cuda:0')))
                print(f"Loaded pre-trained model: {weight_path}")
                args.epochs = 1
                train_loss = train_acc = 0
            else:
                print(f"Pre-trained model not found: {weight_path}")
            model.eval()
            current_pred, current_truth = [], []
            val_loss, num_val_correct, num_val_examples = 0.0, 0, 0
            with torch.no_grad():
                for img, gt_label in test_loader:
                    img, gt_label = img.to(device), gt_label.to(device)
                    pre_emotion, feature = model(img)
                    preds = torch.max(pre_emotion, 1)[1]
                    all_features.extend(feature.cpu().numpy())  
                    all_preds.extend(preds.cpu().numpy())  
                    num_val_correct += (preds == gt_label).sum().item()
                    num_val_examples += gt_label.size(0)
                    current_pred.extend(preds.cpu().tolist())
                    current_truth.extend(gt_label.cpu().tolist())
            val_loss /= len(test_loader.dataset)
            val_acc = num_val_correct / num_val_examples
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
        accuracydict = {'pred': current_pred, 'truth': current_truth}
        all_accuracy_dict[n_subname] = accuracydict
        print(f"Subject {n_subname} Ground Truth: {accuracydict['truth']}")
        print(f"Subject {n_subname} Predictions: {accuracydict['pred']}")

        UF1, UAR, F1_score = recognition_evaluation(current_truth, current_pred, show=True,
                                                    label_dict=label_dict)
        print(f'Subject {n_subname} | UF1: {UF1:.4f} | UAR: {UAR:.4f} | F1: {F1_score:.4f}')
    print('Accuracy Dictionary for All Subjects:', all_accuracy_dict)
    recognition_eval(all_accuracy_dict,label_dict)

