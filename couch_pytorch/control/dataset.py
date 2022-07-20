from os.path import join
import numpy as np
import torch
from torch.utils.data import Dataset
from copy import deepcopy

class ControlData(Dataset):
    def __init__(self, args):

        self.dtype = torch.float32
        self.L = args.L
        self.normalize = args.normalize
        self.input_data = []
        self.output_data = []
        self.sequences = []

        self.mix_contact = args.mix_contact
        self.num_contacts = args.num_contacts
        self.traj_length = args.traj_length
        self.negative = args.negative

        self.train = args.train

        data_path = args.data_path

        with open(join(data_path, 'Input.txt'), "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                self.input_data.append(np.array([float(x) for x in lines[i].split(' ')]))
            f.close()

        with open(join(data_path, 'Output.txt'), "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                self.output_data.append(np.array([float(y) for y in lines[i].split(' ')]))
            f.close()

        with open(join(data_path, 'Sequences.txt'), "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                self.sequences.append(np.array([float(seq) for seq in lines[i].split(' ')]))
            f.close()

        self.input_data = np.stack(self.input_data, 0)
        self.output_data = np.stack(self.output_data, 0)

        self.J = self.input_data[:, -66:]
        self.input_data = self.input_data[:, :-66]
        self.output_data = self.output_data[:, :-42]
        self.contacts = self.input_data[:, -self.traj_length*self.num_contacts*3:]
        self.input_data = self.input_data[:, :-42]


        self.sequences = np.stack(self.sequences, 0).astype(np.int)
        if self.mix_contact:
            self.motion_vecs = self.input_data[:, -2:]
            self.input_data = self.input_data[:, :-2]

        #################### only positive samples #################
        if self.negative == False:
            keep_idx = []
            for i in range(self.input_data.shape[0]):
                if self.input_data[i,
                   args.start_control:args.start_control + 7 * 3 * self.num_contacts].sum() != 0 and self.output_data[i,
                                                                                                     :7 * 3 * self.num_contacts].sum() != 0:
                    keep_idx.append(i)
            self.input_data = self.input_data[keep_idx]
            self.output_data = self.output_data[keep_idx]
            self.sequences = self.sequences[keep_idx]
            self.contacts = self.contacts[keep_idx]
            self.J = self.J[keep_idx]


        self.output_data = self.output_data[:self.input_data.shape[0]]
        self.sequences = self.sequences[:self.input_data.shape[0]]
        self.contacts = self.contacts[:self.input_data.shape[0]]
        self.J = self.J[:self.input_data.shape[0]]

        self.input_mean, self.input_std, self.output_mean, self.output_std = self.compute_statistics(self.input_data,
                                                                                                     self.output_data)


        N = self.input_data.shape[0]
        self.input_data = torch.tensor(self.input_data, dtype=torch.float32).split(self.L)
        self.output_data = torch.tensor(self.output_data, dtype=torch.float32).split(self.L)
        self.sequences = torch.tensor(self.sequences, dtype=torch.float32).split(self.L)
        self.contacts = torch.tensor(self.contacts, dtype=torch.float32).split(self.L)

        if self.mix_contact:
            self.motion_vecs = torch.tensor(self.motion_vecs, dtype=torch.float32).split(self.L)

        J = torch.tensor(self.J, dtype=torch.float32).split(self.L)

        if N % self.L != 0:
            self.input_data = self.input_data[:-1]
            self.output_data = self.output_data[:-1]
            self.sequences = self.sequences[:-1]
            self.contacts = self.contacts[:-1]

            if self.mix_contact:
                self.motion_vecs = self.motion_vecs[:-1]

            J = J[:-1]

        # Each rollout should contains frames from the same pose sequence only
        valid_ids = []
        for i, seq in enumerate(self.sequences):
            if seq[0] == seq[-1]:
                valid_ids.append(i)
        valid_ids = torch.tensor(valid_ids, dtype=torch.long)
        print("Total no of rollouts {}, valid {}, invalid {}".format(len(self.sequences), valid_ids.shape[0],
                                                                     len(self.sequences) - valid_ids.shape[0]))

        self.input_data = [self.input_data[id] for id in valid_ids]
        self.output_data = [self.output_data[id] for id in valid_ids]
        self.contacts = [self.contacts[id] for id in valid_ids]

        self.sequences = [self.sequences[id] for id in valid_ids]

        J = [J[id] for id in valid_ids]


    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        y = self.output_data[idx]
        x = self.input_data[idx]
        contact = self.contacts[idx]
        J = self.J[idx]
        if self.mix_contact == True:
            motion_vec = self.motion_vecs[idx]
        else:
            motion_vec = 0
        return {'x': x, 'y': y,
                # 'contact': torch.stack([self.left_contact[idx], self.right_contact[idx]], -2).reshape(-1, 2, 3)[0],
                'contact': contact,
                'motion_vec': motion_vec,
                'J': J}


    def compute_statistics(self, input_data, output_data):
        x_mean = input_data.mean(0)
        x_std = input_data.std(0)
        x_std[x_std == 0] = 1

        y_mean = output_data.mean(0)
        y_std = output_data.std(0)
        y_std[y_std == 0] = 1
        return x_mean, x_std, y_mean, y_std



    def dataset_get_mean_contact(self):

        prev_seq = self.sequences[0]

        avg_left = []
        avg_right = []
        avg_idx_right = []
        avg_idx_left = []

        self.contacts = self.contacts.reshape(-1, self.num_contacts, self.traj_length, 3)
        prev_contact = self.contacts[0]

        for i in range(self.sequences.shape[0]):
            contact = self.contacts[i]
            if self.sequences[i] != prev_seq or i == self.sequences.shape[0] - 1 or contact[0].sum() == 0 or\
                (i > 6 and np.linalg.norm(contact[0] - self.contacts[i-6, 0], 2, axis=-1).mean() > 0.15 and (contact[0]==0).sum()!= 0 and (self.contacts[i-6, 0]==0).sum()!=0):

                if len(avg_right) > 0:
                    tmp = deepcopy(self.contacts[avg_idx_right][:, 0].reshape(-1, 3))
                    tmp[(tmp.sum(-1) != 0).ravel()] = np.concatenate(avg_right, 0).mean(0)
                    for j in range(len(avg_idx_right)):
                        self.contacts[avg_idx_right[j]][0] = tmp.reshape(len(avg_idx_right), self.traj_length, 3)[j]
                else:
                    self.contacts[int(prev_seq.item()), 0] = 0.
                # if len(avg_idx_right) > 0:
                #     print(self.sequences[i], i, len(avg_idx_right))
                avg_right = []
                avg_idx_right = []
            else:
                avg_right.append(contact[0, contact[0].sum(-1) != 0])
                avg_idx_right.append(i)

            if self.sequences[i] != prev_seq or i == self.sequences.shape[0] - 1 or contact[1].sum() == 0 or  \
                (i > 6 and np.linalg.norm(contact[1] - self.contacts[i-6, 1], 2, axis=-1).mean() > 0.15 and (contact[1]==0).sum()!= 0 and (self.contacts[i-6, 1]==0).sum()!=0):

                if len(avg_left) > 0:
                    tmp = self.contacts[avg_idx_left][:, 1].reshape(-1, 3)
                    tmp[(tmp.sum(-1) != 0).ravel()] = np.concatenate(avg_left, 0).mean(0)
                    for j in range(len(avg_idx_left)):
                        self.contacts[avg_idx_left[j]][1] = tmp.reshape(len(avg_idx_left), self.traj_length, 3)[j]
                else:
                    self.contacts[int(prev_seq.item()), 1] = 0

                avg_left = []
                avg_idx_left = []

            else:
                avg_left.append(contact[1, contact[1].sum(-1) != 0])
                avg_idx_left.append(i)

            prev_seq = self.sequences[i]
            prev_contact = self.contacts[i]

        self.contacts = self.contacts.reshape(-1, self.num_contacts * self.traj_length * 3)

        return

    def to_data_list(self, Sequences, data):
        data_list = []
        prev_seq = Sequences[0]
        data_tmp = []

        for i in range(Sequences.shape[0]):
            if Sequences[i] != prev_seq or i == Sequences.shape[0] - 1:
                data_list.append(torch.from_numpy(np.array(data_tmp)).float())
                data_tmp = []

            else:
                data_tmp.append(data[i])
            prev_seq = Sequences[i]
        return data_list
