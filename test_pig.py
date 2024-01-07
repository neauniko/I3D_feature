import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from torchvision import datasets, transforms
import videotransforms

from pytorch_i3d import InceptionI3d

from PigDataset import Dataset_pig, load_video_data, read_label

def run(init_lr=0.1, max_steps=64e3, mode='rgb', batch_size=1, save_model='feature_npy'):
    # test_data_path = 'E:/pig/data/videonpy/dataset_test_npy'
    # test_label_path = 'E:/pig/data/videonpy/label/label_test.csv'
    test_data_path = '/media/ubuntu/Elements1/pig/data/videonpy/dataset_test_npy'
    test_label_path = '/media/ubuntu/Elements1/pig/data/videonpy/label/label_test.csv'
    # train_data_path = '/media/ubuntu/Elements/pig/LRCN_PyTorch-main/dataset_npy'
    # label_path = '/media/ubuntu/Elements/pig/LRCN_PyTorch-main/label/label.csv'
    test_video_infos = read_label(test_label_path)
    test_dataset = Dataset_pig(test_video_infos, test_data_path)
    dataloaders_test = DataLoader(test_dataset, batch_size=1, shuffle=True,
                                  num_workers=0, pin_memory=True)

    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(2)
    i3d.load_state_dict(torch.load('feature_npy/feature_npy002250.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.train(False)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    a = []
    c = []
    d = []
    e = []

    with torch.no_grad():
        for data in dataloaders_test:
            inputs, labels = data
            inputs = Variable(inputs.cuda())

            label = Variable(labels.cuda())

            output = i3d(inputs)
            _, b = torch.max(output.data, dim=1)
            # b = b.suqeeze(0)
            # if b>0.5 and label == 1 :
            #     tp = tp + 1
            # if b<0.5 and label ==0:
            #     tn = tn + 1


            if label == b:
                if label == 1.0:
                    tp = tp + 1
                    c.append(b)
                if label == 0.0:
                    d.append(b)
                    tn = tn + 1
            else:
                if label == 0.0:
                    a.append(b)
                    fp = fp + 1
                if label == 1.0:
                    fn = fn + 1
                    e.append(b)

            # max_value, max_index = output.max(1, keepdim=True

    print("Accuracy: " + str((tp + tn) * 1.0 * 100 / len(dataloaders_test)))
    # print("Recall: " + str(tp * 1.0 * 100 / (tp + fn) * 1.0))
    # print("Precision: " + str(tp * 1.0 * 100 / (tp + fp) * 1.0))




if __name__ == '__main__':
    # need to add argparse
    run()

