import math
import os
import pandas as pd
import numpy as np

import torch
from sklearn.utils import Bunch
from torch.utils.data import Dataset

global_xmin = [math.inf, math.inf]
global_xmax = [-math.inf, -math.inf]


class TSKeypointDataset(Dataset):
    def __init__(self, dataset_file, kp_dir, kp_names=None, seq_length=None, transform=None, target_transform=None,
                 device='cpu', in_ram=True):
        self.data_labels = pd.read_csv(dataset_file, dtype={'Video': str, 'ID': int, 'hard_vote': int})
        self.kp_dir = kp_dir
        self.transform = transform
        self.target_transform = target_transform
        self.kp_names = kp_names
        self.usecols = None
        self.seq_length = seq_length
        self.device = device

        self.labels_list = np.asarray([self.binarize_label(l) for l in self.data_labels.iloc[:, 2]])

        self.in_ram = in_ram
        self.dataset_items = None

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):

        vid_name, id, label = (self.data_labels.iloc[idx, 0],
                               self.data_labels.iloc[idx, 1],
                               self.data_labels.iloc[idx, 2])

        if self.in_ram:  # get keypoints loaded in ram

            if self.dataset_items is None:
                # read all the keypoints from the dataset if not loaded yet
                self.load_dataset_in_ram()

            keypoints = self.dataset_items[idx].copy()

        else:  # read keypoints from csv file
            kp_path = os.path.join(self.kp_dir, f"{vid_name}.csv")
            keypoints = self.read_kp_csv(kp_path, self.kp_names)

        if self.transform:
            keypoints = self.transform(keypoints)
        if self.target_transform:
            label = self.target_transform(label)

        keypoints = torch.tensor(keypoints, dtype=torch.float64).to(self.device)
        label = torch.tensor(label).to(self.device)

        return keypoints, label, id, vid_name

    def get_dataset_sklearn_style(self, pos_thr=None):

        data = self.data_labels.iloc[:, 0].to_numpy()
        ids = self.data_labels.iloc[:, 1].to_numpy()
        labels = self.data_labels.iloc[:, 2].to_numpy()

        if pos_thr is not None:
            labels = (labels >= pos_thr).astype(int)

        return Bunch(data=data, labels=labels, ids=ids)

    def read_kp_csv(self, kp_path, kp_names=None):

        if self.usecols is None:
            header = np.genfromtxt(kp_path, dtype=str, delimiter=',', max_rows=1)
            self.usecols = [0, 1]
            for i in range(2, header.shape[0]):
                if 'likelihood' in header[i]:
                    continue
                if kp_names is None:
                    self.usecols.append(i)
                else:
                    for kp_name in kp_names:
                        if kp_name in header[i]:
                            self.usecols.append(i)

        kp_df = np.genfromtxt(kp_path, dtype=str, delimiter=',', usecols=self.usecols)
        keypoints = np.round(kp_df[1:, 2:].astype(float))

        keypoints = keypoints.reshape((keypoints.shape[0], -1, 2))  # n_frames, n_keypoints, n_coordinates

        return keypoints

    def load_dataset_in_ram(self):
        dataset_items = []

        for i in range(self.data_labels.shape[0]):
            vid_name, id, label = (self.data_labels.iloc[i, 0],
                                   self.data_labels.iloc[i, 1],
                                   self.data_labels.iloc[i, 2])

            kp_path = os.path.join(self.kp_dir, f"{vid_name}.csv")
            keypoints = self.read_kp_csv(kp_path, self.kp_names)
            dataset_items.append(keypoints)

        self.dataset_items = dataset_items

    def trim_keypoints(self, keypoints, length=None, random=False):
        start = 0
        if length is None:
            length = keypoints.shape[0]
        if random:
            max_start = keypoints.shape[0] - length - 1
            start = torch.randint(start, max_start)

        return keypoints[start:start+length, :, :]

    def normalize(self, keypoints, a, b):
        # x_hat = (b - a) * ( (x - xmin) / (xmax - xmin) ) + a
        kp = keypoints.reshape((keypoints.shape[0], -1))
        xmin = np.min(kp, axis=0)
        xmax = np.max(kp, axis=0)

        kp = (b - a) * ((kp - xmin) / (xmax - xmin)) + a
        kp = kp.reshape((keypoints.shape[0], -1, 2))

        return kp

    class RandomXYJitter(object):
        def __init__(self, percentage):
            self.percentage = percentage

        def __call__(self, keypoints):
            if self.percentage <= 0:
                return keypoints

            # get nose and head
            head = keypoints[0, 6, :]
            nose = keypoints[0, 5, :]
            # euclidean distance for first frame between nose and head
            euclidean = np.linalg.norm(head - nose) * self.percentage
            # jitter depending on euclidean distance
            jittered_kp = np.random.normal(keypoints, euclidean)
            return jittered_kp

    class GaussianNoise(object):
        def __init__(self, percentage):
            self.percentage = percentage

        def __call__(self, keypoints):
            jittered_kp = np.zeros_like(keypoints)
            for k in range(keypoints.shape[1]):  # noise is computed per keypoint
                for xy in (0, 1):  # and per x,y coordinate
                    noise = np.random.normal(0, keypoints[:, k, xy].std(), keypoints[:, k, xy].size) * self.percentage
                    jittered_kp[:, k, xy] = keypoints[:, k, xy] + noise
            return jittered_kp
            
    class TrimKeypoints(object):

        def __init__(self, seq_length, random=False):
            self.seq_length = seq_length
            self.random = random

        def __call__(self, keypoints):
            start = 0
            if self.seq_length is None:
                self.seq_length = keypoints.shape[0]
            if self.random:
                max_start = keypoints.shape[0] - self.seq_length
                start = np.random.randint(start, max_start + 1)

            return keypoints[start:start + self.seq_length, :, :]

    class RelativeKeypoints(object):

        def __init__(self, kp_index, ref_frame=0, a=0, b=1):
            self.kp_index = kp_index
            self.ref_frame = ref_frame
            self.a = a
            self.b = b

        def __call__(self, keypoints):
            # normalize to 0 and 1
            xmax = np.max(keypoints, axis=(0, 1), keepdims=True)
            xmin = np.min(keypoints, axis=(0, 1), keepdims=True)

            kp = (self.b - self.a) * ((keypoints - xmin) / (xmax - xmin)) + self.a

            return kp

    @staticmethod
    def binarize_label(label, thr=1):
        binary_label = 0 if label <= thr else 1

        return binary_label
