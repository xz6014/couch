import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
from os.path import exists, join
from os import makedirs
import sys
import argparse
from models.controlnet import controlnet

parser = argparse.ArgumentParser(description='Train ControlNet')
# Config
parser.add_argument('--config', default='configs/controlnet.yaml')
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

for key in config:
    for k, v in config[key].items():
        setattr(args, k, v)

torch.manual_seed(1234)
np.random.seed(1234)


class Trainer(object):
    def __init__(self, dataloader, args, output_size):
        self.save_path = 'ckpts/{}'.format(args.ckpt_name)
        self.hidden_dim = args.hidden_dim
        self.latent_dim = args.latent_dim
        self.traj_length = args.traj_length
        self.num_contacts = args.num_contacts
        self.dim_control = args.dim_control
        self.match_threshold = args.match_threshold
        self.mix_contact = args.mix_contact

        self.weight_mse = args.weight_mse
        self.weight_reg = args.weight_reg
        self.weight_match = args.weight_match
        self.weight_init = args.weight_init
        self.weight_entropy = args.weight_entropy
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = controlnet(output_dim=output_size, args=args).to(self.device)


        self.dataloader = dataloader
        self.ckpt_name = args.ckpt_name
        self.batch_size = args.batch_size

        self.epoch = args.epoch
        self.Te = 10
        self.Tmult = 2
        self.learning_rate_ini = 1e-4
        self.weightDecay_ini = 0.0025


    def train(self):

        # Print Training Information
        total_batch = len(self.dataloader)
        print('Training information')
        print('Total Batch', '->', total_batch)
        print('--------------------')

        # Start training
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate_ini,
                                      weight_decay=self.weightDecay_ini)
        iters = len(self.dataloader) * self.epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=5e-6)
        train_loss = []
        P = 1
        for e in range(self.epoch):

            total_loss = 0
            autoregressive_count = 0

            if args.scheduled_sampling:
                if e <= args.E1:
                    P = 1
                elif args.E1 < e <= args.E2:
                    P = 1 - (e - args.E1) / float(args.E2 - args.E1)
                else:
                    P = 0
            Bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor(1 - P, dtype=torch.float))

            for i, data in enumerate(self.dataloader):

                batch_X = data['x'].to(self.device)
                batch_Y = data['y'].to(self.device)
                batch_contact = data['contact'].to(self.device)
                batch_motion_vec = data['motion_vec'].to(self.device)

                h, c = None, None

                total_loss_mse = 0
                total_loss_entropy = 0
                total_loss_reg = 0
                total_loss_match = 0
                total_loss_init = 0.

                for j in range(args.L):
                    optimizer.zero_grad()

                    x = batch_X[:, j]
                    y = batch_Y[:, j]

                    if j != 0 and Bernoulli.sample().int() == 1:
                        # only use GT Phase Update
                        if self.dim_control == 42:
                            x = torch.cat((batch_X[:, j, :args.start_control],
                                           x_hat[:, :self.traj_length * self.num_contacts * 3]), dim=-1)
                        else:
                            x = torch.cat((batch_X[:, j, :args.start_control],
                                           x_hat[:, :self.traj_length * self.num_contacts * 3],
                                           batch_X[:, j,
                                           args.start_control + self.traj_length * self.num_contacts * 3:]),
                                          dim=-1)
                        autoregressive_count += 1
                    if args.arch == 'lstm':
                        y_hat, h, c = self.model(x, h, c)
                    else:
                        y_hat = self.model(x)

                    loss_reg = torch.nn.functional.mse_loss(
                        x[:, args.start_control:args.start_control + self.num_contacts * 3 * self.traj_length].ravel(),
                        y_hat[:, :self.num_contacts * 3 * self.traj_length].ravel())

                    loss_init = torch.nn.functional.mse_loss(
                        x[:, args.start_control:args.start_control + self.num_contacts * 3 * self.traj_length].reshape(-1, 2, self.traj_length, 3)[:, :, 0].ravel(),
                        y_hat[:, :self.num_contacts * 3 * self.traj_length].reshape(-1, 2, self.traj_length, 3)[:, :, 0].ravel())

                    ratio = 2
                    magnitude = y_hat[:, :self.num_contacts * 3 * self.traj_length].detach(). \
                                    reshape(-1, self.num_contacts, self.traj_length, 3).norm(p=2, dim=-1) * ratio


                    loss_match = (((magnitude < 1) & (magnitude > 3e-2)).int() * \
                                  (1 - magnitude - y_hat[:, self.num_contacts * 3 * self.traj_length:].reshape(-1, self.num_contacts,
                                                                     self.traj_length)).abs()).mean()

                    loss_reg = loss_reg * self.weight_reg
                    loss_mse, loss_entropy = self.loss_fn(y_hat, y)
                    loss_mse = loss_mse * self.weight_mse
                    loss_entropy = loss_entropy * self.weight_entropy
                    loss_init = loss_init * self.weight_init
                    loss_match = loss_match * self.weight_match

                    loss = loss_mse + loss_entropy + loss_reg + loss_match + loss_init

                    total_loss_mse += loss_mse.item()
                    total_loss_entropy += loss_entropy.item()
                    total_loss_reg += loss_reg.item()
                    total_loss_match += loss_match.item()
                    total_loss_init += loss_init.item()

                    total_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                    y_hat = y_hat.detach()

                    if args.arch == 'lstm':
                        h = h.detach()
                        c = c.detach()
                    if args.scheduled_sampling:
                        x_hat, _ = self.transform_traj_to_world(y_hat, batch_contact[:, j])

                scheduler.step()
            print('Epoch: {} | LR: {:.8f} | P: {:.2f} | Loss MSE {:.4f} | Loss CE {:.4f} '
                  '| Loss Match {:.4f}  | Loss Reg {:.4f} | Loss Init {:.4f}'.format(e,
                                                                  scheduler.get_lr()[0],
                                                                  P,
                                                                  total_loss_mse,
                                                                  total_loss_entropy,
                                                                  total_loss_match,
                                                                  total_loss_reg,
                                                                  total_loss_init))
            if e > 0 and (e + 1) % 30 == 0:
                self.save_weights(self.save_path, e + 1)

            if e == self.epoch - 1:
                self.save_onnx()

    def transform_traj_to_world(self, y, contacts):

        c = contacts.reshape(-1, self.num_contacts, self.traj_length, 3)
        traj = c + y[:, :self.traj_length * self.num_contacts * 3].reshape(-1, self.num_contacts,
                                                                           self.traj_length, 3)
        traj_world = traj.reshape(-1, self.num_contacts, self.traj_length, 3)
        traj_interp = (traj_world[:, :, 0].unsqueeze(2) - c) * torch.arange(self.traj_length).flip(
            0).float().to(self.device).reshape(1, 1, -1, 1) / 6
        return torch.cat([traj_interp.reshape(-1, self.traj_length * self.num_contacts * 3),
                          y[:, self.traj_length * self.num_contacts * 3:]], -1), \
               traj_world

    def loss_fn(self, output, y):
        loss_mse = torch.nn.functional.mse_loss(output[:, :self.traj_length * self.num_contacts * 3].ravel(),
                                                y[:, :self.traj_length * self.num_contacts * 3].ravel())
        loss_entropy = torch.nn.functional.mse_loss(output[:, self.traj_length * self.num_contacts * 3:].ravel(),
                                                y[:, self.traj_length * self.num_contacts * 3:].ravel())
        return loss_mse, loss_entropy

    def update_lr_wd(self, optimizer, AP, epoch):
        clr, wdc = AP.getParameter(epoch)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = clr
            param_group['weight_decay'] = wdc

    def save_weights(self, model_file, epoch):
        print('Saving weights to', join(model_file, '{}.bin'.format(epoch)))
        if not exists(model_file):
            makedirs(model_file)
        torch.save(self.model.state_dict(), join(model_file, '{}.bin'.format(epoch)))


    def save_onnx(self):
        """
        Exporting ONNX Model. Integrating Data Statistics into the model graph
        """

        # Dummy input
        dummy_input = torch.randn(1, args.dim_control + args.start_control).cuda()
        dummy_h = torch.zeros(2, 1, self.hidden_dim * 2).normal_(std=0.01).expand(-1, 1, -1).contiguous().cuda()
        dummy_c = torch.zeros(2, 1, self.hidden_dim * 2).normal_(std=0.01).expand(-1, 1, -1).contiguous().cuda()


        output_names = ["output1"]
        torch.onnx.export(self.model, (dummy_input, dummy_h, dummy_c),
                          "ckpts/{}/{}.onnx".format(self.ckpt_name, self.ckpt_name), export_params=True,
                          input_names=['inputs', 'h', 'c'],
                          verbose=True,
                          output_names=output_names)




if __name__ == '__main__':
    from dataset import ControlData
    dataset = ControlData(args)

    batch_size = args.batch_size if args.train else 1
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    trainer = Trainer(dataloader=train_loader, args=args, output_size=args.dim_control)
    if args.train:
        trainer.train()
