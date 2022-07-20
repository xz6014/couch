import torch
import torch.nn as nn
import torch.nn.functional  as F
import numpy as np

rng = np.random.RandomState(23456)

class PredictionNet(nn.Module):
    def __init__(self, n_input, n_output, n_expert_weights, h, drop_prob=0.3, rng=rng):
        super(PredictionNet, self).__init__()
        self.n_expert_weights = n_expert_weights
        self.n_input = n_input
        self.n_output = n_output
        self.h = h

        self.register_parameter(name='expert_weights_fc0',
                                param=self.initial_alpha((n_expert_weights, h, n_input), rng))
        self.register_parameter(name='expert_weights_fc1', param=self.initial_alpha((n_expert_weights, h, h), rng))
        self.register_parameter(name='expert_weights_fc2',
                                param=self.initial_alpha((n_expert_weights, n_output, h), rng))
        self.register_parameter(name='expert_bias_fc0', param=nn.Parameter(torch.zeros((n_expert_weights, h))))
        self.register_parameter(name='expert_bias_fc1', param=nn.Parameter(torch.zeros((n_expert_weights, h))))
        self.register_parameter(name='expert_bias_fc2', param=nn.Parameter(torch.zeros((n_expert_weights, n_output))))

        self.drop1 = nn.Dropout(drop_prob)
        self.drop2 = nn.Dropout(drop_prob)
        self.drop3 = nn.Dropout(drop_prob)

        self.drop_prob = drop_prob

    def initial_alpha(self, shape, rng):
        alpha_bound = np.sqrt(6. / np.prod(shape[-2:]))
        alpha = np.asarray(
            rng.uniform(low=-alpha_bound, high=alpha_bound, size=shape),
            dtype=np.float32)
        return nn.Parameter(torch.from_numpy(alpha))

    def forward(self, x, BC):
        W0, B0, W1, B1, W2, B2 = self.blend(BC)

        x = self.drop1(x)
        x = torch.baddbmm(B0.unsqueeze(2), W0, x.unsqueeze(2))
        x = F.elu(x)
        x = self.drop2(x)
        x = torch.baddbmm(B1.unsqueeze(2), W1, x)
        x = F.elu(x)
        x = self.drop3(x)
        x = torch.baddbmm(B2.unsqueeze(2), W2, x)
        x = x.squeeze(2)
        return x

    def blend(self, BC):
        BC_w = BC.unsqueeze(2).unsqueeze(2)
        BC_b = BC.unsqueeze(2)

        W0 = torch.sum(BC_w * self.expert_weights_fc0.unsqueeze(0), dim=1)
        B0 = torch.sum(BC_b * self.expert_bias_fc0.unsqueeze(0), dim=1)
        W1 = torch.sum(BC_w * self.expert_weights_fc1.unsqueeze(0), dim=1)
        B1 = torch.sum(BC_b * self.expert_bias_fc1.unsqueeze(0), dim=1)
        W2 = torch.sum(BC_w * self.expert_weights_fc2.unsqueeze(0), dim=1)
        B2 = torch.sum(BC_b * self.expert_bias_fc2.unsqueeze(0), dim=1)
        return W0, B0, W1, B1, W2, B2


class GatingNN(nn.Module):
    def __init__(self, n_input, n_expert_weights, hg, drop_prob=0.0):
        super(GatingNN, self).__init__()
        self.fc0 = nn.Linear(n_input, hg)
        self.fc1 = nn.Linear(hg, hg)
        self.fc2 = nn.Linear(hg, n_expert_weights)

        self.drop1 = nn.Dropout(drop_prob)
        self.drop2 = nn.Dropout(drop_prob)
        self.drop3 = nn.Dropout(drop_prob)

        self.drop_prob = drop_prob

    def forward(self, x):
        x = self.drop1(x)
        x = F.elu(self.fc0(x))
        x = self.drop2(x)
        x = F.elu(self.fc1(x))
        x = self.drop3(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x


class MotionNet(nn.Module):
    def __init__(self, output_dim, args, drop_prob=0.3):
        super(MotionNet, self).__init__()
        self.rng = np.random.RandomState(1234)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.past_length = args.traj_length
        self.future_length = args.future
        self.num_outputs = output_dim
        self.num_experts = args.num_experts

        self.start_pose = 0
        
        ## smpl sit
        # if 'smpl' in args.data_path:
            # self.start_goal = 342
            # self.start_environment = 446
            # self.start_interaction = 2480
            # self.start_gating = 4528
            # self.dim_gating = 208
        self.start_goal = args.start_goal
        self.start_environment = args.start_environment
        self.start_interaction = args.start_interaction
        self.start_hand = args.start_hand
        self.start_gating = args.start_gating
        self.dim_gating = args.dim_gating


        self.gatingNN = GatingNN(self.dim_gating, 10, 512, drop_prob)
        self.predNN = PredictionNet(512+128+512+512+128, self.num_outputs, 10, 512,
                                           drop_prob)

        self.contact_layer = nn.Sigmoid()
        #
        self.encoder0 = nn.Sequential(nn.Dropout(p=drop_prob),
                                      nn.Linear(self.start_goal, 512),
                                      nn.ELU(),
                                      nn.Dropout(p=drop_prob),
                                      nn.Linear(512, 512),
                                      nn.ELU())

        self.encoder1 = nn.Sequential(nn.Dropout(p=drop_prob),
                                      nn.Linear(self.start_environment - self.start_goal, 128),
                                      nn.ELU(),
                                      nn.Dropout(p=drop_prob),
                                      nn.Linear(128, 128),
                                      nn.ELU())

        self.encoder2 = nn.Sequential(nn.Dropout(p=drop_prob),
                                      nn.Linear(self.start_interaction - self.start_environment, 512),
                                      nn.ELU(),
                                      nn.Dropout(p=drop_prob),
                                      nn.Linear(512, 512),
                                      nn.ELU())

        self.encoder3 = nn.Sequential(nn.Dropout(p=drop_prob),
                                      nn.Linear(self.start_hand - self.start_interaction, 512),
                                      nn.ELU(),
                                      nn.Dropout(p=drop_prob),
                                      nn.Linear(512, 512),
                                      nn.ELU())

        self.encoder4 = nn.Sequential(nn.Dropout(p=drop_prob),
                                      nn.Linear(self.start_gating - self.start_hand, 128),
                                      nn.ELU(),
                                      nn.Dropout(p=drop_prob),
                                      nn.Linear(128, 128),
                                      nn.ELU())
    def forward(self, x):
        BC = self.gatingNN(x[:, -self.dim_gating:])
        out = self.predNN(torch.cat([self.encoder0(x[:, :self.start_goal]),
                                       self.encoder1(x[:, self.start_goal:self.start_environment]),
                                       self.encoder2(x[:, self.start_environment:self.start_interaction]),
                                       self.encoder3(x[:, self.start_interaction:self.start_hand]),
                                       self.encoder4(x[:, self.start_hand:self.start_gating])], -1), BC)
        return out

    def save_weights_unity(self):

        sd = self.state_dict()

        # save gating network
        for i in range(3):
            sd['gatingNN.fc{}.weight'.format(i)].numpy().tofile(self.save_path + '/wc%0i%0i%0i_w.bin' % (0, i, 0))
            sd['gatingNN.fc{}.bias'.format(i)].numpy().tofile(self.save_path + '/wc%0i%0i%0i_b.bin' % (0, i, 0))

        # save MLPs
        encoder_idx = [1, 4]
        for i in range(4): # num encoder
            for j in range(2): # layer
                sd['encoder{}.{}.weight'.format(i, encoder_idx[j])].numpy().tofile(self.save_path + '/encoder%0i_w%0i.bin' % (i, j))
                sd['encoder{}.{}.bias'.format(i, encoder_idx[j])].numpy().tofile(self.save_path + '/encoder%0i_b%0i.bin' % (i, j))

        # save Motion Network:
        for i in range(3): # layer
            for j in range(self.num_experts):
                sd['motionNN.expert_weights_fc{}'.format(i)][j].numpy().tofile(self.save_path + '/wc%0i%0i%0i_w.bin' % (1, i, j))
                sd['motionNN.expert_bias_fc{}'.format(i)][j].numpy().tofile(self.save_path + '/wc%0i%0i%0i_b.bin' % (1, i, j))



