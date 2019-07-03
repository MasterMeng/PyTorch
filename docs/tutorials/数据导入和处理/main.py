#!/usr/bin/env python3

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# 或略警告
import warnings
warnings.filterwarnings('ignore')

plt.ion()

landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))


def show_landmarks(image, landmarks):
    '''
    展示带标记点的图像
    '''
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(5)


# plt.figure()
# show_landmarks(io.imread(os.path.join('data/faces', img_name)), landmarks)
# plt.show()


class FaceLandmarksDataset(Dataset):
    '''
    Face Landmarks Dataset
    '''

    def __init__(self, csv_file, root_dir, transform=None):
        '''
        param csv_file(string): 带注释的CSV文件路径
        param root_dit(string): 存储图像的路径
        param transform(callable,optional): 被应用到样本的可选transform操作
        '''
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    '''
    按给定的尺寸缩放图像  

    param output_size (tuple or int): 目标输出尺寸。如果是tuple，输出为匹配的输出尺寸；如果是int，则匹配较小的图像边缘，保证相同的长宽比例。
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        landmarks = landmarks*[new_w/w, new_h/h]
        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    '''
    随机裁剪图像

    param output_size (tuple or int): 目标输出尺寸。如果是int，正方形裁剪
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)

        image = image[top:top+new_h, left:left+new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    '''
    将ndarrays格式样本转换为Tensors
    '''

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}


# face_dataset = FaceLandmarksDataset(
#     csv_file='data/faces/face_landmarks.csv', root_dir='data/faces/')


# fig = plt.figure()

# for i in range(len(face_dataset)):
#     sample = face_dataset[i]

#     print(i, sample['image'].shape, sample['landmarks'].shape)

#     ax = plt.subplot(1, 4, i+1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_landmarks(**sample)

#     if i == 3:
#         plt.show()
#         break


# scale = Rescale(256)
# crop = RandomCrop(128)
# composed = transforms.Compose([Rescale(256), RandomCrop(224)])

# fig = plt.figure()
# sample = face_dataset[65]
# for i, tsfrm in enumerate([scale, crop, composed]):
#     transformed_sample = tsfrm(sample)

#     ax = plt.subplot(1, 3, i+1)
#     plt.tight_layout()
#     ax.set_title(type(tsfrm).__name__)
#     show_landmarks(**transformed_sample)
# plt.show()

transformed_dataset = FaceLandmarksDataset(
    csv_file='data/faces/face_landmarks.csv',
    root_dir='data/faces/',
    transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()])
)

# for i in range(len(transformed_dataset)):
#     sample = transformed_dataset[i]

#     print(i, sample['image'].size(), sample['landmarks'].size())

#     if i == 3:
#         break


dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)


def show_landmarks_batch(sample_batched):
    '''
    批量展示样本
    '''
    images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
    batch_size = len(sample_batched)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(
            landmarks_batch[i, :, 0].numpy() + i * im_size +
            (i+1)*grid_border_size,
            landmarks_batch[i, :, 1].numpy() + grid_border_size,
            s=10, marker='.', c='r'
        )
        plt.title('Batch from dataloader')


for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
