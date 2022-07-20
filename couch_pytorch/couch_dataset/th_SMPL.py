'''
Takes in smpl parms and initialises a smpl object with optimizable params.
class th_SMPL currently does not take batch dim.
Author: Bharat
Edit: Xiaohan
'''
import torch
import torch.nn as nn
# from smpl_layer import SMPL_Layer
import sys
sys.path.append('/BS/XZ_project1/work/PycharmProjects/chair/pose_regression/lib')
from smpl_layer import SMPL_Layer

class th_batch_SMPL(nn.Module):
    def __init__(self, gender='male', hands=False):
        super(th_batch_SMPL, self).__init__()

        self.gender = gender
        self.hands = hands
        ## pytorch smpl
        self.smpl = SMPL_Layer(center_idx=0,
                               gender=self.gender,
                               model_root='/BS/bharat/work/installation/smplpytorch/smplpytorch/native/models',
                               num_betas=10,
                               hands=self.hands)
        self.faces = self.smpl.th_faces


    def forward(self, betas, pose, trans, scale):
        betas = betas.repeat(pose.shape[0], 1)
        verts, jtr, tposed, naked = self.smpl(pose,
                                              th_betas=betas,
                                              th_trans=trans,
                                              scale=scale)
        return verts, jtr, tposed, naked


class th_SMPL(nn.Module):
    def __init__(self, betas=None, pose=None, trans=None, offsets=None, tailor=False):
        super(th_SMPL, self).__init__()
        if betas is None:
            self.betas = nn.Parameter(torch.zeros(300,))
        else:
            self.betas = nn.Parameter(betas)
        if pose is None:
            self.pose = nn.Parameter(torch.zeros(72,))
        else:
            self.pose = nn.Parameter(pose)
        if trans is None:
            self.trans = nn.Parameter(torch.zeros(3,))
        else:
            self.trans = nn.Parameter(trans)
        if offsets is None:

            if tailor:
                self.offsets = torch.zeros(6890, 3).cuda()
            else:
                self.offsets = nn.Parameter(torch.zeros(6890, 3))
        else:

            if tailor:
                self.offsets = offsets.cuda() #todo:hack for tailornt, should be tensor
            else:
                self.offsets = nn.Parameter(offsets)
        # self.update_betas = nn.Parameter(torch.zeros(10,))
        # self.update_pose = nn.Parameter(torch.zeros(72,))
        # self.update_trans = nn.Parameter(torch.zeros(3,))

        ## pytorch smpl
        self.smpl = SMPL_Layer(center_idx=0, gender=self.gender,
                          model_root='/BS/bharat/work/installation/smplpytorch/smplpytorch/native/common')

    def forward(self):
        verts, Jtr, tposed, naked, a_global = self.smpl(self.pose.unsqueeze(axis=0),
                                              th_betas=self.betas.unsqueeze(axis=0),
                                              th_trans=self.trans.unsqueeze(axis=0),
                                              th_offsets=self.offsets.unsqueeze(axis=0))
        return verts[0]
