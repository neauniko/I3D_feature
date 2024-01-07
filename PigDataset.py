import torch
import torch.utils.data as data
import os
import json
import copy
import random
import pandas as pd
import tqdm
import numpy as np
from torchvision.transforms import transforms



def load_video_data(video_infos, data_path):
    data_dict = {}
    print('loading video frame data ...')
    for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
        data = np.load(os.path.join(data_path, video_name))
        data = np.transpose(data, [0, 3, 1, 2])
        data_dict[video_name] = data

    return data_dict



def get_label(data):
    # 标签转换映射
    class_labels_map = {}
    labels_class_map = {}
    index = 0
    for class_label in data['labels']:
        if class_label == '':
            continue
        class_labels_map[class_label] = index
        labels_class_map[index] = class_label
        index += 1
    return class_labels_map,labels_class_map


def get_video_list(train_path):
    video_list = []

    return video_list


def split_video(train_path):
    training_list = []

    video_list = get_video_list(train_path)



    for video_name in video_list:

        video_name_split = video_name.split('/')[-1]
        label = get_label(video_name)

        training_list.append(
            {
                'video_name': video_name_split,
                'label': label
            }
        )


    return training_list


def read_label(label_path):
    df_info = pd.DataFrame(pd.read_csv(label_path)).values[:]
    video_infos = {}
    for info in df_info:
        video_infos[info[0]] = {
            'label': info[1],
            'label_id': info[2],
        }
    return video_infos

def set_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    return transform


class Dataset_pig(data.Dataset):
    def __init__(self, train_video_infos, data_path):

        self.train_video_infos = train_video_infos
        self.data_path = data_path
        self.getdatadict()
        self.transform = set_transforms()

    def getdatadict(self):
        self.data_dict = {}
        print('loading video frame data ...')
        a = list(self.train_video_infos)
        for video_name in tqdm.tqdm((a), ncols=0):
            video_name_split = video_name.split('.')[0]

            data = np.load(os.path.join(self.data_path, video_name_split + '.npy'))
            # data = np.transpose(data, [0, 3, 1, 2])
            self.data_dict[video_name] = data

        self.video_list = list(self.data_dict.keys())
        print("video numbers: %d" % (len(self.video_list)))


    def __getitem__(self, idx):
        a = self.train_video_infos

        video_info = list(self.train_video_infos)[idx]
        video_data = self.data_dict[video_info]
        label = self.train_video_infos[video_info]['label_id']
        data = torch.empty(size=(75, 3, 224, 224)).cuda()
        a = torch.tensor(video_data)


        for t in range(a.size()[0]):
            video_frame = video_data[t,:,:,:]
            video_frame = self.transform(video_frame)
            data[t,:,:,:] = video_frame

        data = data.cpu().numpy()

        # video_data = np.from_tensor(video_data)
        data = np.transpose(data, [1, 0, 2, 3])

        return data, label


    def __len__(self):
        return len(self.train_video_infos)


