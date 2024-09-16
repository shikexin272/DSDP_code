# -*- coding: utf-8 -*-

from skimage.metrics import peak_signal_noise_ratio as PSNR
import scipy.io as scio
import numpy as np
from DSDP_Net import *



class BasicModel:
    def __init__(self, model, img_degraded, gt, window_size,  num_epoch, lr, smooth_coef):
        self.img_degraded = img_degraded
        self.gt = gt
        self.xhat_old = self.img_degraded
        self.window_size = window_size
        self.num_epoch = num_epoch
        self.lr = lr
        self.smooth_coef = smooth_coef
        self.model = model

        self.device = 'cuda'
        self.best_psnr = 0.
        self.best_recover = None
        self.data_type = torch.cuda.FloatTensor
        self.output = None
        self.best_error = float('inf')

    def train(self):
        # the input and the label of the networks
        img_degraded_padded = self.pre_padding(self.img_degraded,self.window_size)
        Input = self.array2tensor(img_degraded_padded, self.device)
        label =self.array2tensor(img_degraded_padded, self.device)

        # loss function
        self.fn_loss = nn.L1Loss()

        # the parameters of the networks
        n_params = self.get_parameter_number()
        print('Network parameters = %d' % (n_params['Total']))
        print('Trainable network parameters = %d' % (n_params['Trainable']))

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)


        for epoch in range(self.num_epoch):
            self.epoch = epoch

            # train the network
            xhat, Gu_x, Gu_y, Gv_x = self.model(Input)

            Loss = self.fn_loss(xhat, label)
            Loss += tau * torch.mean(torch.abs(Gu_x))
            Loss += tau * torch.mean(torch.abs(Gu_y))
            Loss += mu * torch.mean(torch.abs(Gv_x))

            self.optimizer.zero_grad()
            Loss.backward()
            self.optimizer.step()

            # convert the torch tensor as a numpy array and smooth it
            xhat_np = self.tensor2array(xhat, clip=True)
            self.xhat_np = self.crop_image(xhat_np)
            self.output = self.postprocess(self.xhat_np)


            # converge or not
            if epoch % 10 == 0:
                rel_err = np.linalg.norm(self.output.flatten() - self.xhat_old.flatten(),ord=2) / np.linalg.norm(self.xhat_old.flatten(), ord=2)
                self.xhat_old = self.output

                if rel_err < self.best_error:
                    self.best_error = rel_err
                    self.best_recover = self.output
                if self.gt is None:
                    print('it = %04d, rel_error = %.5f\tbest_rel_error = %.5f' % (self.epoch, rel_err, self.best_error))
                else:
                        self.obtain_psnr()
                        if self.psnr_avg > self.best_psnr:
                            self.best_psnr = self.psnr_avg
                            self.best_recover = self.output
                        print('it = %04d, mpsnr = %.4f,smooth_mpsnr = %.4f\tbest_mpsnr = %.4f'%(self.epoch, self.psnr,self.psnr_avg,self.best_psnr))

        self.img_recon = self.output


    def postprocess(self, xhat_np):
        # smooth the output
        if self.output is None:
            self.output = xhat_np.copy()
        else:
            self.output = self.smooth_coef * self.output + (1 - self.smooth_coef) * xhat_np
        return self.output

    def get_psnr(self, imagery1, imagery2):
        M, N, p = imagery1.shape
        psnrvector = np.zeros(p)
        for i in range(p):
            J = 255 * imagery1[:, :, i]
            I = 255 * imagery2[:, :, i]
            psnrvector[i] = PSNR(J, I, data_range=np.max(J))
        mpsnr = np.mean(psnrvector)
        return mpsnr

    def obtain_psnr(self):
        self.psnr = self.get_psnr(self.gt, self.xhat_np)
        self.psnr_avg = self.get_psnr(self.gt, self.output)


    def pre_padding(self,img,win):
        height = img.shape[0]
        width = img.shape[1]
        if height % win != 0 or width %  win != 0:
            new_height = int(np.ceil(height /  win) *  win)
            new_width = int(np.ceil(width /  win) *  win)

            pad_top = (new_height - height) // 2
            pad_bottom = new_height - height - pad_top
            pad_left = (new_width - width) // 2
            pad_right = new_width - width - pad_left
            img_padded = np.pad(img, ((pad_left, pad_right), (pad_top, pad_bottom), (0, 0)),'reflect')
        else:
            img_padded =img
        return img_padded

    def crop_image(self, padded_image):
        original_height, original_width, _ = self.img_degraded.shape
        padded_height, padded_width, _ = padded_image.shape

        crop_top = (padded_height - original_height) // 2
        crop_bottom = crop_top + original_height
        crop_left = (padded_width - original_width) // 2
        crop_right = crop_left + original_width

        cropped_image = padded_image[crop_top:crop_bottom, crop_left:crop_right, :]
        return cropped_image

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def array2tensor(self,x, device):
        x = torch.tensor(x, dtype=torch.float32, device=device)
        x = x.permute(2, 0, 1)[None, ...]
        return x

    def tensor2array(self,x, clip=True):
        if x.ndim == 4:  # an image [1,C,H,W]
            x = np.array(x.detach().squeeze(0).permute(1, 2, 0).cpu())  # [H,W,C]
        elif x.ndim == 3:  # an signal [1,C,N]
            x = np.array(x.detach().squeeze(0).permute(1, 0).cpu())  # [N,C]
        if clip:
            x = np.clip(x, 0., 1.)
        return x

# 自己加的地方
if __name__ == "__main__":
    global tau,mu
    data_name = 'gf5_demo'
    cln_hsi = None
    num_epoch = 2000
    lr = 1e-3
    smooth_coef = 0.8
    n_feat = 256

    # the parameters need to tune.
    rank = 5# the rank of the subspace.
    window_size = 5# the window size of the WSA block.
    tau = 4 * 1e-2# the parameter before U_TV.
    mu = 5 * 1e-3# the parameter before V_TV.

    # laod the data
    filepath = 'data/' + data_name + '.mat'
    mat = scio.loadmat(filepath)
    noi_hsi = mat["noi"]#noisy mat
    if 'cln' in mat:
        cln_hsi = mat["cln"]#clean mat
    print('the shape of the data is', noi_hsi.shape)

    # define the DSDP-network
    model = DSDP_Net(rank=rank, in_c=noi_hsi.shape[-1], n_feat=n_feat, window_size=window_size, act=nn.LeakyReLU(1e-3), bias=True)

    # train the DSDP-network
    method = BasicModel(model=model, img_degraded=noi_hsi, gt=cln_hsi, window_size=window_size,
                        num_epoch=num_epoch, lr=lr, smooth_coef=smooth_coef
                        )
    method.train()

