from os.path import join, exists
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ContactVAEData(Dataset):
    def __init__(self, args):

        torch.manual_seed(0)
        data_path = '{}'.format(args.data_path)

        self.X = []
        self.Y = []
        self.SequencesNames = []

        with open(join(data_path, 'Input.txt'), "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                self.X.append(np.array([float(x) for x in lines[i].split(' ')]))
            f.close()

        with open(join(data_path, 'Output.txt'), "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                self.Y.append(np.array([float(y) for y in lines[i].split(' ')]))

            f.close()

        with open(join(data_path, 'SequencesNames.txt'), "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if 'mirror' in lines:
                    self.SequencesNames.append('_'.join(np.array(lines[i].rstrip().split('_'))[[0, 2]].tolist()) + '_mirrored')
                else:
                    self.SequencesNames.append(
                        '_'.join(np.array(lines[i].rstrip().split('_'))[[0, 2]].tolist()))

            f.close()

        self.X = np.stack(self.X[:-1], 0)
        self.Y = np.stack(self.Y[:-1], 0)
        self.obj_centers = self.X[:, -3:]
        self.X = self.X[:, :-3]

        self.X = torch.from_numpy(self.X).float()
        self.Y = torch.from_numpy(self.Y).float()
        self.obj_centers = torch.from_numpy(self.obj_centers).float()

        self.Y = self.Y[:self.X.shape[0]]
        self.SequencesNames = np.array(self.SequencesNames[:-1])
        self.SequencesNames = self.SequencesNames[:self.X.shape[0]]

        # Split training and testing
        test_idx = torch.randperm(self.X.shape[0])[:1000]
        if args.train:
            train_idx = torch.ones([self.X.shape[0]])
            train_idx[test_idx] = 0
            self.X = self.X[train_idx == 1]
            self.Y = self.Y[train_idx == 1]
            self.obj_centers = self.obj_centers[train_idx == 1]
            self.SequencesNames = self.SequencesNames[train_idx == 1]

        else:
            self.X = self.X[test_idx]
            self.Y = self.Y[test_idx]
            self.obj_centers = self.obj_centers[test_idx]
            self.SequencesNames = self.SequencesNames[test_idx]



        if args.normalize:
            self.x_mean = self.X.numpy().mean(0)
            self.x_std = self.X.numpy().std(0)
            self.x_std[self.x_std == 0] = 1

            self.y_mean = self.Y.numpy().mean(0)
            self.y_std = self.Y.numpy().std(0)
            self.y_std[self.y_std == 0] = 1

            self.X = (self.X - torch.from_numpy(self.x_mean).reshape(1, -1)) / torch.from_numpy(self.x_std).reshape(1, -1)
            self.Y = (self.Y - torch.from_numpy(self.y_mean).reshape(1, -1)) / torch.from_numpy(self.y_std).reshape(1, -1)

        print("input data shape {}".format(self.X.shape))
        print("output data shape {}".format(self.Y.shape))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx, :]
        return {'x': x, 'y': y, 'chair_name': self.SequencesNames[idx], 'chair_centre': self.obj_centers[idx]}