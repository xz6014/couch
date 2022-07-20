import numpy as np
import torch
from os.path import join
from lib.th_SMPL import th_batch_SMPL
from psbody.mesh.meshviewer import MeshViewers
from psbody.mesh.mesh import Mesh
from psbody.mesh.lines import Lines
from helper_functions import floor_mesh
import time
from visualize_processed import all_objects_static_params, xsens_to_goal_transformation, SCAN_PATH
from copy import deepcopy
import pickle as pkl



if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('-s', "--seq_folder", default='mocap_npz/subject2_npz/session2_subject2_chair2_sit04')
    args = parser.parse_args()

    path = join(args.seq_folder + '_{}_full.npz'.format(args.save_name)).replace(args.seq_folder.split('/')[-2], 'mocap')

    seq_name = path.split('/')[-1]
    date = seq_name.split('_')[0]
    subject = seq_name.split('_')[1]
    chair_name = seq_name.split('_')[2]

    betas = torch.from_numpy(pkl.load(open('betas/{}.pkl'.format(subject), 'rb'))['beta']).float()
    smpl = th_batch_SMPL('male', False)
    smpl_hands = th_batch_SMPL('male', True)
    threshold = 2
    high = 2
    low = -0.35
    mv = MeshViewers([1, 1])
    color = 'white'

    J = smpl(pose=torch.zeros((1, 72)), trans=torch.zeros((1, 3)), betas=betas, scale=1)[1].numpy()[0]


    obj_scan_to_camera_params = all_objects_static_params[date][subject][chair_name]
    to_goal_y_angle = xsens_to_goal_transformation[date][chair_name]['angle']
    to_goal_y_height = xsens_to_goal_transformation[date][chair_name]['height']

    if 'chair1' in path:
        key = 'chairblack'
    elif 'chair2' in path:
        key = 'chairwood'
    elif 'sofa' in path:
        key = 'sofa'
    elif 'chair3' in path:
        key = 'chair3'

    # loading chair
    chair_mesh = Mesh()
    chair_mesh.load_from_ply(SCAN_PATH[key])
    chair_mesh_centre = 0.5 * (chair_mesh.v.max(0) + chair_mesh.v.min(0))
    chair_mesh.v -= chair_mesh_centre

    # loading chair params
    angle = all_objects_static_params[date][subject][chair_name]['angle']
    trans = all_objects_static_params[date][subject][chair_name]['trans']

    data = np.load(path, allow_pickle=True)


    pose_params_init = torch.from_numpy(data['pose']).float()
    trans_params_init = torch.from_numpy(data['trans']).float()
    chair_to_xsens_R = data['R'][0]
    chair_to_xsens_t = data['t'][0]

    R_chair = np.matmul(chair_to_xsens_R, axis_angle_to_matrix(torch.from_numpy(angle)).numpy())
    t_chair = chair_to_xsens_t

    chair_mesh.v = np.matmul(R_chair, chair_mesh.v.T).T + t_chair

    # chair to goal
    R_chair_y = axis_angle_to_matrix(
        torch.tensor([0, matrix_to_axis_angle(torch.from_numpy(R_chair))[1].item() + to_goal_y_angle, 0]))
    chair_mesh.v = np.matmul(R_chair_y.numpy().T, (chair_mesh.v.T - np.array([[t_chair[0]], [ 0], [t_chair[2]]]))).T

    # smpl to goal
    pose_params_init[:, :3] = matrix_to_axis_angle(
        torch.bmm(R_chair_y.T.unsqueeze(0).repeat(pose_params_init.shape[0], 1, 1),
                  axis_angle_to_matrix(pose_params_init[:, :3])))

    trans_params_init = torch.bmm(R_chair_y.T.unsqueeze(0).repeat(pose_params_init.shape[0], 1, 1),
                                  trans_params_init.unsqueeze(-1) - torch.tensor(
                                      [t_chair[0], 0, t_chair[2]]).reshape(1, 3, 1).float()).squeeze(-1)


    # SMPL Forward Kinematics
    verts, J_transformed = smpl_hands(pose=pose_params_init.float(),
                                      trans=trans_params_init.float(),
                                      betas=betas,
                                      scale=1)[:2]
    verts = verts.detach().cpu().numpy()

    height = deepcopy(J_transformed[:, [7, 8], 1].numpy()).min(1).mean() - 0.1
    trans_params_init[:, 1] -= deepcopy(verts[:, :, 1]).min(1)
    J_transformed[:, :, 1] -= deepcopy(verts[:, :, 1]).min(1)[:, np.newaxis]
    verts[:, :, 1] -= deepcopy(verts[:, :, 1]).min(1)[:, np.newaxis]

    # To visualise
    J_transformed = J_transformed.detach().cpu().numpy()[:, list(range(0, 23)) + [37]]

    for i in range(verts.shape[0]):

        mv[0][0].static_meshes = [Mesh(verts[i], smpl_hands.faces.numpy(), vc=color)] + [chair_mesh] + \
                                 [floor_mesh(scale=(3, 3, 3))] + \
                                 [Mesh(np.array(
                                     [[-threshold, high, -threshold], [threshold, high, -threshold],
                                      [-threshold, high, threshold],
                                      [threshold, high, threshold], [-threshold, low, -threshold],
                                      [threshold, low, -threshold],  # Todo change sit_pos to obj_offset
                                      [-threshold, low, threshold], [threshold, low, threshold]]), [])]
        mv[0][0].static_lines = [Lines(J_transformed[i], smpl.smpl.kintree_table.T[1:22], vc='green')]

        time.sleep(0.02)
