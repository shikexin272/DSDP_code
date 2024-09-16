from NSAB import *
import torch
import torch.nn as nn


class Downsample(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=2, stride=2, padding=0, bias=True):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class DSDP_Net(nn.Module):
    def __init__(self, rank, in_c, n_feat, window_size, act=nn.LeakyReLU(1e-3), bias=True):
        super(DSDP_Net, self).__init__()
        self.device = 'cuda'
        self.aap = nn.AdaptiveAvgPool1d(rank)
        self.window_size = window_size

        # NSAB block
        self.layers = nn.Sequential(
            *[NSAB(dim=n_feat//4,
                   input_resolution=[32,32], num_heads=4, window_size=window_size,
                   mlp_ratio=4,
                   qkv_bias=True, qk_scale=None)
              for _ in range(5)])

        # U-branch
        self.NetU = nn.Sequential(
            nn.Conv2d(in_c, n_feat, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat), act,
            nn.Conv2d(n_feat, n_feat // 4, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat // 4), act,
            self.layers[0],
            nn.Conv2d(n_feat // 4, n_feat, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat), act,
            nn.Conv2d(n_feat, n_feat // 4, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat // 4), act,
            self.layers[1],
            nn.Conv2d(n_feat // 4, n_feat, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat), act,
            nn.Conv2d(n_feat, n_feat // 4, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat // 4), act,
            self.layers[2],
            nn.Conv2d(n_feat // 4, n_feat, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat), act,
            nn.Conv2d(n_feat, n_feat // 4, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat // 4), act,
            self.layers[3],
            nn.Conv2d(n_feat // 4, n_feat, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat), act,
            nn.Conv2d(n_feat, n_feat // 4, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat // 4), act,
            self.layers[4],
            nn.Conv2d(n_feat // 4, n_feat, 3, padding=1, bias=bias), nn.BatchNorm2d(n_feat), act,
            nn.Conv2d(n_feat, rank, 3, padding=1, bias=bias),
        ).to(self.device)

        # V-branch
        self.NetV = nn.Sequential(
            nn.Conv2d(in_c, n_feat, 4, stride=2, padding=1, bias=bias),
            nn.BatchNorm2d(n_feat), act,
            Downsample(n_feat, n_feat, 2, stride=2, padding=0, bias=bias),
            nn.BatchNorm2d(n_feat), act,
            Downsample(n_feat, n_feat, 2, stride=2, padding=0, bias=bias),
            nn.BatchNorm2d(n_feat), act,
            Downsample(n_feat, n_feat, 2, stride=2, padding=0, bias=bias),
            nn.BatchNorm2d(n_feat), act,
            nn.Conv2d(n_feat, in_c, 2, stride=2, padding=0, bias=bias),
        ).to(self.device)

    # estimated U
    def getU(self, x):
        _, DD, HH, WW = x.shape
        U = torch.sigmoid(self.NetU(x)).to(self.device)
        self.U = U
        U = self.U.reshape(1, -1, HH * WW)
        return U

    # estimated V
    def getV(self, x):
        _, DD, HH, WW = x.shape
        V = self.NetV(x).to(self.device)
        V = V.reshape(1, DD, -1)
        V = self.aap(V)
        V = torch.softmax(V, dim=-1)
        self.V = V
        return V


    def TV_Loss(self):
        gradient_u_x = self.U[:, :, :-1, :] - self.U[:, :, 1:, :]
        gradient_u_y = self.U[:, :, :, :-1] - self.U[:, :, :, 1:]
        gradient_v_x = self.V[:, :-1,:] - self.V[:, 1:,:]
        return gradient_u_x, gradient_u_y, gradient_v_x

    def forward(self, x):
        _, DD, HH, WW = x.shape
        U = self.getU(x).to(self.device)# [bz,r,H*W]
        V = self.getV(x).to(self.device)# [bz,B,r]
        xhat = torch.bmm(V,U)
        xhat = xhat.reshape(1, DD, HH, WW)
        Gu_x, Gu_y, Gv_x = self.TV_Loss()
        return xhat, Gu_x, Gu_y, Gv_x