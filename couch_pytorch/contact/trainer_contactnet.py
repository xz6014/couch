import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import yaml
from os.path import exists, join
from os import makedirs
import argparse

from dataset import ContactVAEData

parser = argparse.ArgumentParser(description='Train ContactNet')
# Config
parser.add_argument('--config', default='configs/contactvae.yaml')
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

for key in config:
    for k, v in config[key].items():
        setattr(args, k, v)

from contactnet import ContactVAE

torch.manual_seed(1234)
np.random.seed(1234)


class Trainer(object):
    def __init__(self, dataloader, args):
        self.save_path = 'ckpts/{}'.format(args.ckpt_name)
        self.hidden_dim = args.hidden_dim
        self.latent_dim = args.latent_dim
        self.z_dim = args.z_dim

        self.model = ContactVAE(input_dim_goalnet=6, interaction_dim=2048, pose_dim=132, h_dim_goalnet=512,
                                z_dim_goalnet=self.z_dim).cuda()
        print(self.model)

        if torch.cuda.is_available():
            self.model.cuda()

        self.dataloader = dataloader
        if args.normalize:
            self.x_mean = torch.from_numpy(self.dataloader.dataset.x_mean).float().cuda()
            self.x_std = torch.from_numpy(self.dataloader.dataset.x_std).float().cuda()
            self.y_mean = torch.from_numpy(self.dataloader.dataset.y_mean).float().cuda()
            self.y_std = torch.from_numpy(self.dataloader.dataset.y_std).float().cuda()

        self.ckpt_name = args.ckpt_name
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.learning_rate_ini = args.lr
        self.weightDecay_ini = args.lr_decay

        self.loss_fn = torch.nn.MSELoss()

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

        for e in range(self.epoch):

            total_loss = 0
            total_recon_loss = 0
            total_kld_loss = 0
            for i, data in enumerate(self.dataloader):
                batch_X = data['x'].cuda()
                batch_Y = data['y'].cuda()

                optimizer.zero_grad()
                y_hat, mu, logvar = self.model(batch_Y, batch_X)
                recon_loss = (y_hat.reshape(-1, 2, 3) - batch_Y.reshape(-1, 2, 3)).norm(p=2, dim=-1).mean()
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + args.kl_w * kld
                loss.backward()
                optimizer.step()

                total_recon_loss += recon_loss.item()
                total_kld_loss += kld.item()
                total_loss += loss.item()

                scheduler.step()

            print('====> Epoch: {}, Total_train_loss: {:.4f}, recon_loss: {:.4f}, kld_loss: {:.4f}'.format(e + 1,
                                                                                                           total_loss,
                                                                                                           total_recon_loss,
                                                                                                           total_kld_loss))

            if e == self.epoch - 1:
                self.save_weights(self.save_path, e + 1)
                self.save_onnx()


    def save_weights(self, model_file, epoch):
        print('Saving weights to', join(model_file, '{}.bin'.format(epoch)))
        if not exists(model_file):
            makedirs(model_file)
        torch.save(self.model.state_dict(), join(model_file, '{}.bin'.format(epoch)))

    def save_onnx(self):
        """
        Exporting ONNX Model. Integrating Data Statistics into the model graph
        """
        import numpy
        import onnxruntime
        from copy import deepcopy
        # Dummy input
        dummy_input = torch.randn(1, 2180 + self.z_dim).cuda()


        output_names = ["output1"]
        torch.onnx.export(self.model.decoder, dummy_input, "ckpts/{}/{}.onnx".format(self.ckpt_name, self.ckpt_name),
                          export_params=True,
                          input_names=['x'],
                          verbose=True,
                          output_names=output_names)

        pt_outputs = self.model.decoder(deepcopy(dummy_input))

        # Run the exported model with ONNX Runtime
        ort_sess = onnxruntime.InferenceSession("ckpts/{}/{}.onnx".format(self.ckpt_name, self.ckpt_name))
        ort_inputs = {ort_sess.get_inputs()[0].name: deepcopy(dummy_input).cpu().numpy()}
        ort_outputs = ort_sess.run(None, ort_inputs)

        # Validate PyTorch and ONNX Runtime results
        numpy.testing.assert_allclose(pt_outputs.detach().cpu().numpy(), ort_outputs[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    def load_weights(self, model_file, epoch):
        print('Loading weights from', join(model_file, '{}.bin'.format(epoch)))
        self.model.load_state_dict(
            torch.load(join(model_file, '{}.bin'.format(epoch)), map_location=lambda storage, loc: storage))

    def test(self):
        import time
        from lib.th_SMPL import th_batch_SMPL
        from lib.helper_functions import floor_mesh
        from psbody.mesh.meshviewer import MeshViewers
        from psbody.mesh.lines import Lines
        from psbody.mesh.mesh import Mesh
        from psbody.mesh.sphere import Sphere
        from psbody.mesh.colors import name_to_rgb

        mv = MeshViewers([1, 1], keepalive=True, window_width=800, window_height=600)
        threshold = 1.5

        self.model.load_state_dict(torch.load('ckpts/{}/150.bin'.format(args.ckpt_name)))
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(self.dataloader):
                batch_X = data['x'].cuda()
                batch_Y = data['y'].cuda()
                chair_name = data['chair_name']
                chair_centre = data['chair_centre'].numpy()

                for j in range(200):
                    print(chair_name)
                    z = torch.randn((1, self.z_dim)).cuda() * 2

                    y_hat = self.model.decoder(torch.cat([z, batch_X], -1))
                    # y_hat = batch_Y
                    contacts = y_hat.reshape(-1, 3).detach().cpu().numpy() + chair_centre

                    meshes = []
                    chair = Mesh()
                    chair.load_from_obj('{}.obj'.format(chair_name[0]))
                    if 'mirror' in chair_name:
                        chair.v[:, 0] *= -1
                    # meshes += [chair]
                    # meshes += [Sphere(chair_centre, 0.05).to_mesh()]
                    meshes += [Sphere(C, 0.05).to_mesh() for C in contacts]
                    meshes += [floor_mesh(scale=(3, 3, 3))]
                    meshes += [Mesh(np.array(
                        [[-threshold, 2, -threshold], [threshold, 2, -threshold],
                         [-threshold, 2, threshold],
                         [threshold, 2, threshold], [-threshold, -0.1, -threshold],
                         [threshold, -0.1, -threshold],
                         [-threshold, -0.1, threshold], [threshold, -0.1, threshold]]), [])]
                    mv[0][0].static_meshes = meshes

                    J = batch_X[:, :132].reshape(-1, 22, 2, 3)[:, :, 0].detach().cpu().numpy().reshape(-1, 3) + chair_centre
                    rents = np.array([[0, 1], [0, 5], [0, 9],
                                      [1, 2], [2, 3], [3, 4],
                                      [5, 6], [6, 7], [7, 8],
                                      [9, 10], [10, 11],
                                      [11, 12], [11, 14], [11, 18],
                                      [12, 13], [14, 15], [15, 16], [16, 17],
                                      [18, 19], [19, 20], [20, 21]])
                    mv[0][0].static_lines = [Lines(J, rents, vc='green')]
                    time.sleep(1)


if __name__ == '__main__':
    if args.train:
        train_dataset = ContactVAEData(args)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        trainer = Trainer(dataloader=train_loader, args=args)
        trainer.train()
    else:
        test_dataset = ContactVAEData(args)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
        trainer = Trainer(dataloader=test_loader, args=args)
        trainer.test()
