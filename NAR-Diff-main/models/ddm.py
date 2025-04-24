import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from piq import SSIMLoss
import utils
from models.unet import DiffusionUNet
from models.decom import Retinex_decom
from models.enhance import Illumination_Enhance_Net


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if name not in self.shadow:
                self.shadow[name] = param.clone().detach().to(param.device)
            self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def total_variation_loss(x):
    return torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])) + \
           torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=[2, 3], keepdim=True)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)
        x_se = self.fc1(avg_pool)
        x_se = self.fc2(x_se)
        x_se = self.sigmoid(x_se).view(x.size(0), x.size(1), 1, 1)
        return x * x_se

class NoiseEstimator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, reduction=16):
        super(NoiseEstimator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.multi_scale_encoder = nn.ModuleList([
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.Conv2d(512, 256, kernel_size=5, padding=2),
        ])
        self.se_block = None
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        device = x.device
        x = self.encoder(x)
        multi_scale_features = [scale_conv(x) for scale_conv in self.multi_scale_encoder]
        x = torch.cat(multi_scale_features, dim=1)
        self.se_block = SEBlock(x.size(1), reduction=16).to(device)
        self.add_module("se_block", self.se_block)
        x = self.se_block(x)
        x = self.decoder(x)
        return x

class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()
        self.args = args
        self.config = config
        self.device = config.device
        self.Unet = DiffusionUNet(config)
        self.decom = self.load_stage1(Retinex_decom(), 'ckpt/stage1')
        self.enhance_L = Illumination_Enhance_Net(config.model.in_channels)
        self.noise_estimator = NoiseEstimator(in_channels=3)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    @staticmethod
    def load_stage1(model, model_dir):
        checkpoint = utils.logging.load_checkpoint(os.path.join(model_dir, 'Decom.pth'), 'cuda')
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        return model

    def sample_training(self, x_cond, b, eta=0.):
        noise_map = self.noise_estimator(x_cond)
        skip = self.config.diffusion.num_diffusion_timesteps // self.config.diffusion.num_sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            c1 = c1 * noise_map
            c2 = c2 * noise_map

            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1]

    def forward(self, Ilow_high):
        data_dict = {}
        b = self.betas.to(Ilow_high.device)

        Ilow = Ilow_high[:, :3, :, :]
        Inormal = Ilow_high[:, 3:, :, :]
        low_R, low_L = self.decom(Ilow)
        high_R, high_L = self.decom(Inormal)
        low_L_norm = utils.data_transform(low_L)
        low_R_norm = utils.data_transform(low_R)
        high_L_norm = utils.data_transform(high_L)
        high_R_norm = utils.data_transform(high_R)

        t = torch.randint(low=0, high=self.num_timesteps, size=(low_R_norm.shape[0] // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:low_R_norm.shape[0]].to(Ilow_high.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        noise = torch.randn_like(high_R_norm)

        if self.training:
            noised_R = high_R_norm * a.sqrt() + noise * (1.0 - a).sqrt()
            pred_noise = self.Unet(torch.cat([low_R_norm, noised_R], dim=1), t.float())
            pred_R = self.sample_training(low_R_norm, b)
            pred_R = utils.inverse_data_transform(pred_R)
            enhanced_image = low_L * pred_R
            data_dict["noise_output"] = pred_noise
            data_dict["e"] = noise
            data_dict["enhanced_image"] = enhanced_image
            data_dict["pred_R"] = pred_R
            data_dict["low_R"] = low_R
            data_dict["low_L"] = low_L
            data_dict["high_R"] = high_R
            data_dict["high_L"] = high_L
        else:
            pred_R = self.sample_training(low_R_norm, b)
            pred_R = utils.inverse_data_transform(pred_R)
            enhanced_image = low_L * pred_R
            data_dict["low_R"] = low_R
            data_dict["low_L"] = low_L
            data_dict["high_R"] = high_R
            data_dict["high_L"] = high_L
            data_dict["pred_R"] = pred_R
            data_dict["enhanced_image"] = enhanced_image
        return data_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device
        self.model = Net(args, config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)
        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {}".format(load_path, self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for name, param in self.model.named_parameters():
            if "decom" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                self.model.train()
                self.step += 1
                x = x.to(self.device)
                output = self.model(x)
                Inormal = x[:, 3:, :, :].to(self.device)

                noise_loss, photo_loss, scc_loss = self.noise_estimation_loss(output, Inormal)
                loss = noise_loss + photo_loss + scc_loss
                data_time += time.time() - data_start

                if self.step % 10 == 0:
                    print(f"Step: {self.step}, Noise Loss: {noise_loss:.5f}, "
                          f"Photo Loss: {photo_loss:.5f}, "
                          f"SCC Loss: {scc_loss:.5f}, "
                          f"Time: {data_time / (i + 1):.5f}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)
                    utils.logging.save_checkpoint({'step': self.step,
                                                   'epoch': epoch + 1,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                  filename=os.path.join(self.config.data.ckpt_dir,
                                                                        f'model_step_{self.step}'))
    def noise_estimation_loss(self, output, y):
        enhanced_image = output["enhanced_image"]
        pred_R = output["pred_R"]
        low_R = output["low_R"]
        low_L = output["low_L"]
        high_R = output["high_R"]
        high_L = output["high_L"]
        noise_output = output["noise_output"]
        e = output["e"]

        noise_loss = self.l2_loss(noise_output, e)
        photo_loss = self.l1_loss(enhanced_image, y)
        scc_loss = self.l1_loss(pred_R, high_R)

        return noise_loss, photo_loss, scc_loss

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder,
                                    self.config.data.type + str(self.config.data.patch_size))
        self.model.eval()
        with torch.no_grad():
            print(f'Performing validation at step: {step}')
            for i, (x, y) in enumerate(val_loader):
                b, _, img_h, img_w = x.shape
                img_h_64 = int(64 * np.ceil(img_h / 64.0))
                img_w_64 = int(64 * np.ceil(img_w / 64.0))
                x = F.pad(x, (0, img_w_64 - img_w, 0, img_h_64 - img_h), 'reflect')
                output_dict = self.model(x.to(self.device))
                pred_x = output_dict["enhanced_image"][:, :, :img_h, :img_w]
                utils.logging.save_image(pred_x, os.path.join(image_folder, str(step), f'{y[0]}_enhanced_image.png'))