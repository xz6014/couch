"""
Helper functions.
Author: Xiaohan
"""
import torch
import numpy as np
import trimesh
import os
from os.path import join
import random
from psbody.mesh import Mesh
from copy import deepcopy
from rotation_conversions import *


def change_starting_point(dataset, ratio=0):
    for sub_name in dataset.sub_names():
        diff = deepcopy(dataset[sub_name]['positions_world'][0, 0]) * ratio
        dataset[sub_name]['positions_world'] += diff
        dataset[sub_name]['trans'] += diff
    return dataset

def compute_trajectory_goalvector(dataset, past_length, future_length, add_zero=False, reverse=False,
                                  to_root_only=False):


    for sub_name in dataset.sub_names():


        if not to_root_only:
            goal_vec_world = dataset[sub_name]['goal_pos'][None] - dataset[sub_name]['positions_world'][:, [0, 22, 23]]
        else:
            goal_vec_world = dataset[sub_name]['goal_pos'][None] - dataset[sub_name]['positions_world'][:, 0:1]

        goal_vec_world = goal_vec_world.transpose([1, 0, 2])

        trans_change_world = []
        if add_zero:
            T = goal_vec_world.shape[1] - past_length
        else:
            T = goal_vec_world.shape[1] - past_length - future_length
        goal_vec_world = np.concatenate([goal_vec_world, goal_vec_world[:, -1][:, None].repeat(future_length, 1)], 1)

        for i in range(past_length + 1 + future_length):
            if i == past_length:
                continue
            if i > past_length:
                if reverse:
                    trans_change_world.append(
                        goal_vec_world[:, i:T + i] - goal_vec_world[:, past_length:past_length + T])

                else:
                    trans_change_world.append(
                        goal_vec_world[:, past_length:past_length + T] - goal_vec_world[:, i:T + i])
            else:
                trans_change_world.append(goal_vec_world[:, i:i + T] - goal_vec_world[:, past_length:past_length + T])

        hand_traj = []
        positions_world = dataset[sub_name]['positions_world']
        for i in range(1, future_length + 1):
            hand_traj.append(positions_world[past_length + i:past_length + i + T, [0, 22, 23]] - positions_world[
                                                                                                 past_length:past_length + T,
                                                                                                 0:1])
        hand_traj = np.stack(hand_traj, 1)

        trans_change_world = np.stack(trans_change_world, 1).reshape(-1, 3).astype(np.float32)  # 3 * L * T * 3

        goal_vec_world = goal_vec_world[:, past_length:past_length + T].reshape(-1, 3).astype(np.float32)

        goal_rot = np.pi

        to_goal = axis_angle_to_matrix(
            torch.stack((torch.zeros(1), torch.tensor(goal_rot).unsqueeze(0), torch.zeros(1)), 1)).numpy().astype(
            np.float32)[0]

        goal_vec_goal = np.matmul(goal_vec_world, to_goal).reshape(3, -1, 3).transpose([1, 0, 2])
        goal_vec_world = goal_vec_world.reshape(3, -1, 3).transpose([1, 0, 2])

        trans_change_goal = np.matmul(trans_change_world, to_goal).reshape(3, past_length + future_length, -1, 3)
        trans_change_goal = trans_change_goal.transpose((2, 1, 0, 3)).reshape(-1, (past_length + future_length), 3, 3)

        trans_change_world = trans_change_world.reshape(3, past_length + future_length, -1, 3). \
            transpose((2, 1, 0, 3)).reshape(-1, (past_length + future_length), 3, 3)

        if not add_zero:
            for k, v in dataset[sub_name].items():
                if isinstance(v, float):
                    continue
                if isinstance(v, str):
                    continue
                if isinstance(v, int):
                    continue
                if isinstance(v, np.float32):
                    continue
                if (len(v) > 30) | (k == 'phase'):
                    dataset[sub_name][k] = v[past_length:past_length + T]
        dataset[sub_name]['goal_vec'] = goal_vec_goal
        dataset[sub_name]['goal_vec_world'] = goal_vec_world
        dataset[sub_name]['trajectory'] = trans_change_goal
        dataset[sub_name]['trajectory_world'] = trans_change_world
        dataset[sub_name]['hand_traj'] = hand_traj

        phase = dataset[sub_name]['phase']
        phase[:-1] = phase[1:] - phase[:-1]
        phase[-1] = phase[-2]
        dataset[sub_name]['phase_change'] = phase

        action_label = dataset[sub_name]['action_label']
        action_label = np.concatenate([action_label[1:], action_label[-1].reshape(1, -1)], 0)
        dataset[sub_name]['label_out'] = action_label

        assert dataset[sub_name]['rotations'].shape[0] == dataset[sub_name]['trajectory_world'].shape[0]
        # assert (dataset[sub_name]['lh'] - dataset[sub_name]['positions_world'][0, 22] - dataset[sub_name][ 'goal_vec_world'][0, 1]).sum() == 0
        # assert (dataset[sub_name]['rh'] - dataset[sub_name]['positions_world'][0, 23] - dataset[sub_name][ 'goal_vec_world'][0, 2]).sum() == 0

    return dataset


def split_data(dataset, mode, shift=0):
    for sub_name in dataset.sub_names():
        action_labels = dataset[sub_name]['action_label'][:, 0]
        idx = action_labels.tolist().index(next(filter(lambda x: x == 0, action_labels))) + shift
        for k in dataset[sub_name].keys():
            if k in ['positions_world', 'positions_local', 'rotations', 'root_rotation', 'trans', 'action_label',
                     'phase', 'output', 'extra_features', 'goal_angle', 'trajectory', 'trajectory_world', 'goal_vec',
                     'goal_vec_world', 'phase_change', 'label_out', 'hand_traj', 'foot_contact']:
                if mode == 'walk':
                    dataset[sub_name][k] = dataset[sub_name][k][:idx]
                else:
                    dataset[sub_name][k] = dataset[sub_name][k][idx:]
    return dataset


def query_goal_vecs_from_stage1(goal_positions, J_transformed, names, contact=None, past_length=10,
                                future_length=0, dataset=None):
    """
    Preparing input trajectories and goal vectors from the predicted walking poses to initialise Pose Prediction
    J: B * T * 3 * 3

    Expect contact points to be of the from a lists of length equivalent to number of sequences
    """
    traj_length = past_length + future_length
    goal_pos = []
    n_joints = J_transformed.shape[2]

    ### only used for walking
    if dataset is not None:
        assert n_joints == 1
        J_init = []
        for name in names:
            J_init.append(dataset[name]['positions_world'][:past_length, 0][:, None])
        J_init = np.stack(J_init, 0)
        J_init[0, :, :, 0] *= -1
        from psbody.mesh import MeshViewer
        from psbody.mesh import Mesh
        J_transformed = np.concatenate([J_init, J_transformed], 1)

    if contact is None:
        if n_joints == 1:
            num_contact = 1
        else:
            num_contact = 3
        for i in range(J_transformed.shape[0]):
            goal_pos.append(goal_positions[names[i]][:num_contact])
        goal_pos = np.stack(goal_pos, 0)  # B * 3 * 3
    else:
        goal_pos = contact

    goal_vec_world = goal_pos[:, np.newaxis] - J_transformed

    traj_world = []
    T = J_transformed.shape[1] - traj_length

    for i in range(traj_length + 1):
        if i == past_length:
            continue
        if i > past_length:
            traj_world.append(goal_vec_world[:, past_length:past_length + T] - goal_vec_world[:, i:T + i])
        else:
            traj_world.append(goal_vec_world[:, i:i + T] - goal_vec_world[:, past_length:past_length + T])

    traj_world = np.stack(traj_world, 2).astype(np.float32)  # B * T * L * 3 * 3
    goal_vec_world = goal_vec_world[:, past_length:past_length + T].astype(np.float32)  # B * T * 3 * 3

    goal_rot = np.pi
    to_goal = axis_angle_to_matrix(
        torch.stack((torch.zeros(1), -torch.tensor(goal_rot).unsqueeze(0), torch.zeros(1)), 1)).numpy().astype(
        np.float32)[0]

    goal_vec_goal = np.matmul(goal_vec_world.reshape(-1, 3), to_goal).reshape(len(names), -1, n_joints, 3)
    traj_goal = np.matmul(traj_world.reshape(-1, 3), to_goal).reshape(len(names), -1, traj_length, n_joints, 3)

    return goal_vec_goal, traj_goal, goal_vec_world, traj_world


def add_hand_pose(pose_params):
    if pose_params.shape[1] == 156:
        return pose_params
    else:
        ## load init hand params:
        lh_mean_pose = np.load('/BS/XZ_project2/work/Human-Chair-Interaction/pose/datasets/lh_mean.npy')[None]
        rh_mean_pose = np.load('/BS/XZ_project2/work/Human-Chair-Interaction/pose/datasets/rh_mean.npy')[None]
        bs = pose_params.shape[0]
        pose_params = np.concatenate([pose_params, np.zeros((bs, 156 - 72))], -1)
        pose_params[:, 66:111] = lh_mean_pose.repeat(bs, 0)
        pose_params[:, 111:] = rh_mean_pose.repeat(bs, 0)

    return pose_params


def walking_traj_finish(trans_world, goal):
    """
    trans_world B * T * 3, but care about xz
    goal_positions: near centroid B * 3
    return: starting index B
    """
    idx = []
    exclude_list = []
    threshold = 3.5
    goal = goal + np.array([0, 0, -4])
    dist = np.linalg.norm(trans_world[:, :, [0, 2]] - goal[:, np.newaxis, [0, 2]], 2, -1)
    for i in range(dist.shape[0]):
        if (dist[i] < threshold).sum() < 1:
            if np.argmin(dist[i]) == 0:
                idx.append(10)
            else:
                idx.append(np.argmin(dist[i]))
        else:
            idx.append(np.where(dist[i] < threshold)[0].min())
    return np.array(idx)


def comb_subject_npz(name, data2, data3):
    comb = {}
    for i in range(len(data2['names'])):
        comb[data2['names'][i]] = {'pose': data2['pose'][i],
                                   'trans': data2['trans'][i]}

    for i in range(len(data3['names'])):
        comb[data3['names'][i]] = {'pose': data3['pose'][i],
                                   'trans': data3['trans'][i]}

    np.savez('../results_pose_sit/{}_comb.npz'.format(name), params=comb)


def load_chair(sub_name, chair):
    chair_path = '/BS/XZ_project2/work/contact_prediction/dataset/chairs/'
    which_chair = sub_name.split('_')[:2]
    chair_exist = len(os.listdir(join(chair_path, '_'.join(which_chair)))) > 0
    if not chair_exist:
        return None
    if chair is not None:
        pass
    else:
        chair = random.sample(list(os.listdir(join(chair_path, '_'.join(which_chair)))), 1)
    chair_name = join(chair_path, '_'.join(which_chair), chair[0])
    chair_mesh = Mesh()
    chair_mesh.load_from_off(chair_name)
    chair_mesh.v *= 16.589719478653578
    chair_mesh.v[:, 1] -= chair_mesh.v[:, 1].min()
    chair_mesh.set_vertex_colors('snow')
    return chair_mesh, join(chair_path + 'watertight', '_'.join(which_chair), chair[0]).replace('.off', '.obj')


def filter_data(dataset, target_length, num_seq, test_subjects, mode='train'):
    sequences = []
    n_discarded = 0
    exclude_list = ['verica_chair2_right_053',
                    'verica_chair2_right_053_m']

    # choose training sequences based on keys
    for i, sub_name in enumerate(dataset.sub_names()):
        # break once this number, selection finishes
        if len(sequences) == num_seq:
            break
        if mode == 'train':
            if sub_name.split('_')[0] in test_subjects:
                continue
        else:
            if sub_name.split('_')[0] not in test_subjects:
                continue
        if sub_name[:-3] in exclude_list:
            continue
        if dataset[sub_name]['rotations'].shape[0] < target_length + 1:
            continue
        sequences.append(sub_name)
    print('{} on %d sequences'.format(mode.upper()) % len(sequences))
    return sequences


def compute_foot_contact(dataset, fid_l=(7, 10), fid_r=(8, 11)):
    """
    positions: [T, J, 3], trimmed (only "chosen_joints")
    fid_l, fid_r: indices of feet joints (in "chosen_joints")
    """
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    velfactor = np.array([0.05, 0.05]) * 4

    for sub_name in dataset.sub_names():
        feet_contact = []
        for fid_index in [fid_l, fid_r]:
            foot_vel = (dataset[sub_name]['positions_world'][1:, fid_index] -
                        dataset[sub_name]['positions_world'][:-1, fid_index]) ** 2  # [T - 1, 2, 3]
            foot_vel = np.sum(foot_vel, axis=-1)  # [T - 1, 2]
            foot_contact = (foot_vel < velfactor).astype(np.float)
            feet_contact.append(foot_contact)
        feet_contact = np.concatenate(feet_contact, axis=-1)  # [T - 1, 4]
        feet_contact = np.concatenate((feet_contact[0:1].copy(), feet_contact), axis=0)
        dataset[sub_name]['foot_contact'] = feet_contact
    return dataset


def create_contact_dict(dataset):
    goal_pos = {}
    for k in dataset.sub_names():
        if '_m' in k:
            continue
        subject_and_chair = '/'.join(['_'.join(k.split('_')[:2])] + dataset[k]['chair_name'])
        if not subject_and_chair in goal_pos.keys():
            goal_pos[subject_and_chair] = [[], [], []]
        goal_pos[subject_and_chair][0].append(dataset[k]['goal_pos'])
        goal_pos[subject_and_chair][1].append(dataset[k]['lh'])
        goal_pos[subject_and_chair][2].append(dataset[k]['rh'])
    return goal_pos


def shuffle_contact(dataset, contact_dict, names):
    import random
    goal_pos = []
    for n in names:
        if '_m' in n:
            continue
        subject_and_chair = '/'.join(n.split('_')[:2] + dataset[n]['chair_name'])
        goal_pos.append(random.sample(contact_dict[subject_and_chair], 1)[0])
    return np.stack(goal_pos, 0)


def shuffle_contact_dataset(dataset, contact_dict):
    for n in dataset.sub_names():
        if '_m' in n:
            continue
        subject_and_chair = '/'.join(['_'.join(n.split('_')[:2])] + dataset[n]['chair_name'])
        idx = np.random.randint(0, len(contact_dict[subject_and_chair][0]), 1)[0]
        dataset[n]['goal_pos'] = contact_dict[subject_and_chair][0][idx]
        dataset[n]['lh'] = contact_dict[subject_and_chair][1][idx]
        dataset[n]['rh'] = contact_dict[subject_and_chair][2][idx]
    return dataset


def shuffle_chair_dataset(dataset, contact_dict):
    import random
    for n in dataset.sub_names():
        if '_m' in n:
            continue
        subject_and_chair = random.sample(list(contact_dict.keys()), 1)[0]
        idx = np.random.randint(0, len(contact_dict[subject_and_chair][0]), 1)[0]
        dataset[n]['goal_pos'] = contact_dict[subject_and_chair][0][idx]
        dataset[n]['lh'] = contact_dict[subject_and_chair][1][idx]
        dataset[n]['rh'] = contact_dict[subject_and_chair][2][idx]
        dataset[n]['chair_new'] = subject_and_chair

    return dataset




def load_config(config, args):
    import yaml
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    return args


def create_smpl(pose, betas, trans):
    from lib.smpl_paths import SmplPaths
    sp = SmplPaths(gender='male')
    smpl = sp.get_smpl()

    smpl.betas[:10] = betas
    smpl.pose[:] = pose[:72]
    smpl.trans[:] = trans

    return smpl


def create_cage(min=-25, max=25):
    from psbody.mesh.lines import Lines
    min_ = min
    max_ = max
    cage = Lines(v=[
        [min_, -0.1, min_],
        [max_, -0.1, min_],
        [max_, -0.1, max_],
        [min_, -0.1, max_],
        #
        [min_, 30, min_],
        [max_, 30, min_],
        [max_, 30, max_],
        [min_, 30, max_],
    ], e=[[0, 1], [1, 2], [2, 3], [0, 3],
          [4, 5], [5, 6], [6, 7], [4, 7],
          [0, 4], [1, 5], [2, 6], [3, 7]], vc='green')

    return [cage]


def play_smpl(poses, betas, trans, contacts=None, mv=None):
    from psbody.mesh import Mesh
    import time
    colours = ['pink', 'brown', 'orange', 'yellow', 'white', 'blue']

    if mv is None:
        mv = create_cage()

    smpl = create_smpl(poses[0], betas, trans[0])
    for n, (p, t) in enumerate(zip(poses, trans)):
        smpl.pose[:] = p[:72]
        smpl.trans[:] = t

        dyn_meshes = [Mesh(smpl.r, smpl.f)]
        if contacts is not None:
            c = contacts[n]
            for n, l in enumerate(c):
                dyn_meshes.append(Mesh(l, [], vc=colours[n]))

        mv.set_dynamic_meshes(dyn_meshes)
        time.sleep(0.05)


def add_floor_to_mesh(chair, centre=(0, 0, 0), min_=-0.9, max_=0.9):
    """
    adds floor to chair mesh (trimesh)
    """
    n_v = len(chair.vertices)
    vertices = np.vstack([chair.vertices, np.array([
        [centre[0] + min_, 0, centre[2] + max_],
        [centre[0] + max_, 0, centre[2] + min_],
        [centre[0] + max_, 0, centre[2] + max_],
        [centre[0] + min_, 0, centre[2] + min_]])])
    faces = np.vstack([chair.faces, np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3]]) + n_v])
    chair = trimesh.Trimesh(vertices, faces)
    return chair


def sample_pts_floor(centre=(0, 0, 0), min_=-0.9, max_=0.9, num_pts=10000):
    """
    adds floor to chair mesh (trimesh)
    """
    import trimesh
    chair = trimesh.Trimesh(vertices=np.array([[centre[0] + min_, 0, centre[2] + max_],
                                               [centre[0] + max_, 0, centre[2] + min_],
                                               [centre[0] + max_, 0, centre[2] + max_],
                                               [centre[0] + min_, 0, centre[2] + min_]]),
                            faces=np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3]]))
    return torch.from_numpy(trimesh.sample.sample_surface_even(chair, num_pts)[0]).float()


def onehot_numpy(idx, num_class=6):
    '''Convert labels to one-hot encoding'''
    nb_classes = 6
    shape = idx.shape
    targets = idx.reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets.reshape(list(shape) + [num_class])
