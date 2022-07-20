import torch
import torch.nn as nn


class controlnet(nn.Module):
    def __init__(self,  output_dim, args):
        super(controlnet, self).__init__()
        self.keep_prob = 1
        self.input_dim_pose = args.start_control
        self.input_dim_control = args.dim_control
        self.output_dim = output_dim
        self.hidden_dim = 128
        self.traj_length = args.traj_length

        self.encoder1 = nn.Sequential(nn.Linear(self.input_dim_pose, self.hidden_dim),
                                      nn.LeakyReLU(0.05, inplace=True),
                                      nn.Linear(self.hidden_dim, self.hidden_dim),
                                      nn.LeakyReLU(0.05, inplace=True),
                                      )
        self.encoder2 = nn.Sequential(nn.Linear(self.input_dim_control, self.hidden_dim),
                                      nn.LeakyReLU(0.05, inplace=True),
                                      nn.Linear(self.hidden_dim, self.hidden_dim),
                                      nn.LeakyReLU(0.05, inplace=True),
                                      )

        self.rnn = nn.LSTM(input_size=self.hidden_dim * 2, hidden_size=self.hidden_dim * 2, num_layers=2,
                           batch_first=True,
                           bidirectional=False)
        self.h0 = nn.Parameter(torch.zeros(2, 1, self.hidden_dim * 2).normal_(std=0.01), requires_grad=True)
        self.c0 = nn.Parameter(torch.zeros(2, 1, self.hidden_dim * 2).normal_(std=0.01), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(in_features=self.hidden_dim * 2, out_features=self.output_dim)

    def forward(self, x, h=None, c=None):
        x = x.unsqueeze(1)
        if h is None:
            h = self.h0.expand(-1, x.shape[0], -1).contiguous()
            c = self.c0.expand(-1, x.shape[0], -1).contiguous()
        x, (h, c) = self.rnn(torch.cat([self.encoder1(x[:, :, :self.input_dim_pose]),
                                        self.encoder2(x[:, :, self.input_dim_pose:self.input_dim_pose + self.input_dim_control])], -1),
                             (h, c))
        x = self.fc(x.squeeze(1))
        return x, h, c
        # return x, h, c


if __name__ == '__main__':
    controlnet = controlnet(20, 30)
    x = torch.randn(1, 122)
    print(controlnet(x))
