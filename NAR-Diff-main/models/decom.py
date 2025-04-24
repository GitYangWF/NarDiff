import torch
import torch.nn as nn

class Retinex_decom(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(Retinex_decom, self).__init__()
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())

        self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
                                    padding=1, padding_mode='replicate')
        self.net1_recon1 = nn.Conv2d(4, 3, kernel_size,
                                     padding=1, padding_mode='replicate')
        self.net1_recon2 = nn.Conv2d(4, 1, kernel_size,
                                     padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)
        feats0 = self.net1_conv0(input_img)
        featss = self.net1_convs(feats0)
        outs1 = self.net1_recon(featss)
        R = torch.sigmoid(outs1[:, 0:3, :, :])
        L = torch.sigmoid(outs1[:, 3:4, :, :])
        L = torch.cat([L for i in range(3)], dim=1)

        return R, L
