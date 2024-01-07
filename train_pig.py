import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from torchvision import datasets, transforms
import videotransforms

from pytorch_i3d import InceptionI3d

from PigDataset import Dataset_pig, load_video_data, read_label


def run(init_lr=0.1, max_steps=64e3, mode='rgb', batch_size=8, save_model='feature_npy'):
    #  读数据
    # train_data_path = 'E:/pig/data/videonpy/dataset_npy'
    # label_path = 'E:/pig/data/videonpy/label/label.csv'
    '/media/ubuntu/Elements1/pig/data/videonpy/label/label.csv'
    train_data_path = '/media/ubuntu/Elements1/pig/data/videonpy/dataset_npy'
    label_path = '/media/ubuntu/Elements1/pig/data/videonpy/label/label.csv'
    # train_data_path = '/media/ubuntu/Elements/pig/LRCN_PyTorch-main/dataset_npy'
    # label_path = '/media/ubuntu/Elements/pig/LRCN_PyTorch-main/label/label.csv'
    train_video_infos = read_label(label_path)
    train_dataset = Dataset_pig(train_video_infos, train_data_path)
    dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=0, pin_memory=True)



    # 模型设置
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(2)
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    ## 模型迭代
    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    loss_function = nn.CrossEntropyLoss()

    # 训练
    num_steps_per_update = 4
    steps = 0
    while steps < max_steps:
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)

        i3d.train(True)

        tot_loss = 0.0
        tot_cls_loss = 0.0
        num_iter = 0
        optimizer.zero_grad()

        for data in dataloaders:
            num_iter += 1
            inputs, labels = data

            inputs = Variable(inputs.cuda())
            t = inputs.size(2)

            labels = Variable(labels.cuda())
            # labels = labels.unsqueeze(-1)
            # labels = torch.tensor(labels, dtype=torch.float64)

            predit = i3d(inputs)
            # predit = torch.max(per_frame_logits, dim=2)[0]

            cls_loss = loss_function(predit, labels)
            tot_cls_loss += cls_loss
            loss = cls_loss / num_steps_per_update
            tot_loss += loss
            loss.backward()
            if num_iter == num_steps_per_update:
                steps += 1
                num_iter = 0
                optimizer.step()
                optimizer.zero_grad()
                lr_sched.step()
                if steps % 50 == 0:
                    print('Cls Loss: {:.4f}'.format(tot_cls_loss / (10 * num_steps_per_update)))
                    # save model
                    path = 'feature_npy/' + save_model + str(steps).zfill(6) + '.pt'
                    torch.save(i3d.module.state_dict(), path)
                    tot_loss = tot_cls_loss = 0.


if __name__ == '__main__':
    # need to add argparse
    run()