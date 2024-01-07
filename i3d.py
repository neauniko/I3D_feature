import os
import torch
import torch.nn as nn
import numpy as np
from common.thumos_dataset import THUMOS_Dataset ,get_video_info,load_video_data,detection_collate,get_video_anno

from pytorch_i3d import InceptionI3d



if __name__ == '__main__':

    batch_size = 32
    clip_length = 256
    stride = 256

    video_infos = get_video_info('E:/AFSD/annotation/val_video_info.csv')

    data_dict = load_video_data(video_infos,
                                      'E:/AFSD/dataset/video_npy/rgb_npy')

    i3d = InceptionI3d(2, in_channels=3)
    # i3d.replace_logits(157)
    # i3d.load_state_dict(torch.load('E:/AFSD/models/i3d_models/rgb_imagenet.pt'))
    i3d.load_state_dict(torch.load('E:/AFSD/models/i3d_models/feature_npy002300.pt'))
    i3d.cuda()
    i3d.eval()



    def forward_batch(b_data):
        # b_data = b_data.transpose([0, 4, 1, 2, 3])
        b_data = torch.from_numpy(b_data)  # b,c,t,h,w  # 40x3x16x224x224

        b_data = Variable(b_data.cuda(), volatile=True).float()
        b_features = i3d.extract_features(b_data)

        b_features = b_features.data.cpu().numpy()[:, :, 0, 0, 0]
        return b_features

    for video_name, video_data in data_dict.items():
        video_list = []

        frequency = 4
        chunk_size = 16

        sample_count = video_infos[video_name]['sample_count']
        c, t, h, w = video_data.shape

        clipped_length = t - 16
        clipped_length = (clipped_length // frequency) * frequency

        full_features = []

        frame_indices = []
        for i in range(clipped_length // frequency + 1):
            frame_indices.append(
                [j for j in range(i * frequency, i * frequency + chunk_size)])
        frame_indices = np.array(frame_indices)
        chunk_num = frame_indices.shape[0]
        # batch_num = int(np.ceil(chunk_num / batch_size))
        # frame_indices = np.array_split(frame_indices, batch_num, axis=0)
        #
        # for batch_id in range(batch_num):
        #     batch_data = load_rgb_batch(video_data, frame_indices[batch_id])
        for idx in range(chunk_num):
            start = idx * frequency
            end = start +16
            input_data = video_data[:, start:end, :, :]
            input_data = torch.from_numpy(input_data).float()
            input_data = ((input_data / 255.0) * 2.0 - 1.0).unsqueeze(0)
            features, feature_dict = i3d.extract_features(input_data.cuda())
            data_feature = feature_dict['Mixed_5c']
            net = nn.AvgPool3d(kernel_size=[2, 4, 4], stride=(1, 1, 1))
            data_feature = net(data_feature)
            full_features.append(data_feature.squeeze(0).squeeze(-1).squeeze(-1).data.cpu().numpy())

        features = np.concatenate(full_features, axis=1)
        np.save(os.path.join('E:/AFSD/feature_npy', video_name + '.npy'), features)




        # if sample_count <= clip_length:
        #     offsetlist = [0]
        # else:
        #     offsetlist = list(range(0, sample_count - clip_length + 1, stride))
        #     if (sample_count - clip_length) % stride:
        #         offsetlist += [sample_count - clip_length]
        #         last_index = sample_count - clip_length
        # feature = []
        # for offset in offsetlist:
        #     left, right = offset + 1, offset + clip_length
        #
        #     input_data = video_data[:, offset: offset + clip_length]
        #
        #     c, t, h, w = input_data.shape
        #     if t < clip_length:
        #         pad_t = clip_length - t
        #         zero_clip = np.zeros([c, pad_t, h, w], input_data.dtype)
        #         input_data = np.concatenate([input_data, zero_clip], 1)
        #
        #     input_data = torch.from_numpy(input_data).float()
        #     input_data = ((input_data / 255.0) * 2.0 - 1.0).unsqueeze(0)
        #     features, feature_dict = i3d.extract_features(input_data.cuda())
        #     data_feature = feature_dict['Mixed_4f']
        #     net = nn.AvgPool3d(kernel_size=[1, 7, 7], stride=(1, 1, 1))
        #     data_feature = net(data_feature)
        #     if last_index and last_index == offset:
        #         cha = offsetlist[-1] - offsetlist[-2]
        #         data_feature = data_feature[:, :, cha:, :, :]
        #     feature.append(data_feature.squeeze(0).data.cpu().numpy())
        # # if last_index and last_index == offset:
        # #     last = feature[-1]
        # #     feature.pop(-1)
        # #     feature_concat = np.concatenate(feature, axis=1)
        # #     feature_concat[:, last_index:, :, :] = last
        # # else:
        # feature_concat = np.concatenate(feature, axis=1)
        # a = sample_count//4
        #
        # feature_concat = feature_concat[:,:a,:,:]
        #
        # np.save(os.path.join('E:/AFSD/feature_npy', video_name + '.npy'), feature_concat)
