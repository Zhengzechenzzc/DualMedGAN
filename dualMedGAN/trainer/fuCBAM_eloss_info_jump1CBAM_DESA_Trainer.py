#!/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os


from .utils import LambdaLR, Logger, ReplayBuffer
from .utils import weights_init_normal, get_config
from .datasets import ImageDataset, ValDataset
from Model.fuCBAM_eloss_info_jump1CBAM_DESA import *
# from Model.fu_eloss_info import Compute_InfomaxLoss
from .utils import Resize, ToTensor, smooothing_loss
from .utils import Logger
from .reg import Reg
from torchvision.transforms import RandomAffine, ToPILImage
from .transformer import Transformer_2D
from skimage import measure
from skimage.metrics import structural_similarity as compare_ssim
from torchvision.utils import save_image
import numpy as np
import cv2
from PIL import Image
import DCloss


class fuCBAM_eloss_info_jump1CBAM_DESA_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks
        self.netG_A2B = fuGenerator(config['input_nc'], config['output_nc']).cuda()
        #打印参数
        # print(self.netG_A2B)
        total_params_o = sum(p.numel() for p in self.netG_A2B.parameters())
        print("G模型的参数量：", total_params_o)

        self.netD_B = Discriminator(input_nc=config['input_nc'], nrkhs=1024, ndf=1024).cuda()
        # 打印参数
        # print(self.netD_B)
        total_params_o = sum(p.numel() for p in self.netD_B.parameters())
        print("D模型的参数量：", total_params_o)

        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        if config['regist']:
            self.R_A = Reg(config['size'], config['size'], config['input_nc'], config['input_nc']).cuda()
            self.spatial_transform = Transformer_2D().cuda()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if config['bidirect']:
            self.netG_B2A = fuGenerator(config['input_nc'], config['output_nc']).cuda()
            self.netD_A = Discriminator(config['input_nc']).cuda()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                                lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        else:
            self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()
        self.HEA_loss = None
        self.info_loss = Compute_InfoLoss()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.target_real = Variable(Tensor(1, 1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1, 1).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Dataset loader
        level = config['noise_level']  # set noise level

        transforms_1 = [ToPILImage(),
                        RandomAffine(degrees=level, translate=[0.02 * level, 0.02 * level],
                                     scale=[1 - 0.02 * level, 1 + 0.02 * level], fill=-1),
                        ToTensor(),
                        Resize(size_tuple=(config['size'], config['size']))]

        transforms_2 = [ToPILImage(),
                        RandomAffine(degrees=1, translate=[0.02, 0.02], scale=[0.98, 1.02], fill=-1),
                        ToTensor(),
                        Resize(size_tuple=(config['size'], config['size']))]

        self.dataloader = DataLoader(
            ImageDataset(config['dataroot'], level, transforms_1=transforms_1, transforms_2=transforms_2,
                         unaligned=False, ),
            batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])

        val_transforms = [ToTensor(),
                          Resize(size_tuple=(config['size'], config['size']))]

        self.val_data = torch.utils.data.DataLoader(
            ValDataset(config['val_dataroot'], transforms_=val_transforms, unaligned=False),
            batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])
        # print(len(self.val_data))

        # Loss plot
        self.logger = Logger(config['name'], config['port'], config['n_epochs'], len(self.dataloader))

    def train(self):
        ###### Training ######
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A'])).cuda()
                real_B = Variable(self.input_B.copy_(batch['B'])).cuda()
                if self.config['bidirect']:  # C dir
                    if self.config['regist']:  # C + R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B = self.netG_A2B(real_A)
                        pred_fake, _, _ = self.netD_B(fake_B)
                        HEA_loss = self.config['hea_lamda'] * DCLoss.DCLoss.loss(real_B, fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        + self.config['hea_lamda'] * HEA_loss

                        fake_A = self.netG_B2A(real_B)
                        pred_fake,_,_ = self.netD_A(fake_A)
                        HEA_loss = self.config['hea_lamda'] * DCLoss.DCLoss.loss(real_B, fake_B)
                        loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        + self.config['hea_lamda'] * HEA_loss

                        Trans = self.R_A(fake_B, real_B)
                        SysRegist_A2B = self.spatial_transform(fake_B, Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B, real_B)  ###SR
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)

                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B = self.netG_A2B(fake_A)
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + SR_loss + SM_loss
                        loss_Total.backward()
                        self.optimizer_G.step()
                        self.optimizer_R_A.step()

                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real, _, _ = self.netD_A(real_A)
                        _, local_feat_A, global_feat_A = self.netD_B(real_A)
                        info_loss_A = self.config['scale'] * self.info_loss.compute_infomax_loss(local_feat_A, global_feat_A)

                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real,
                                                                               self.target_real) + info_loss_A
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake, _, _ = self.netD_A(fake_A.detach())
                        _, local_feat_A, global_feat_A = self.netD_B(fake_A)
                        info_loss_A = self.config['scale'] * self.info_loss.compute_infomax_loss(local_feat_A, global_feat_A)
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_real,
                                                                               self.target_real) + info_loss_A

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real, _, _ = self.netD_B(real_B)
                        _, local_feat_B, global_feat_B = self.netD_B(real_B)
                        info_loss_B = self.config['scale'] * self.info_loss.compute_infomax_loss(local_feat_B, global_feat_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real,
                                                                               self.target_real) + info_loss_B

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake, _, _ = self.netD_B(fake_B.detach())
                        _, local_feat_B, global_feat_B = self.netD_B(fake_B)
                        info_loss_B = self.config['scale'] * self.info_loss.compute_infomax_loss(local_feat_B, global_feat_B)
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake,
                                                                               self.target_fake) + info_loss_B

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ###################################


                    else:  # only  dir:  C
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B = self.netG_A2B(real_A)
                        pred_fake, _, _ = self.netD_B(fake_B)
                        self.HEA_loss = DCLoss.DCLoss.loss(real_B, fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        + self.config['hea_lamda'] * self.HEA_loss

                        fake_A = self.netG_B2A(real_B)
                        pred_fake, _, _ = self.netD_A(fake_A)
                        self.HEA_loss = DCLoss.DCLoss.loss(real_A, fake_A)
                        loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        + self.config['hea_lamda'] * self.HEA_loss

                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)
                        recovered_B = self.netG_A2B(fake_A).cuda()
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                        loss_Total.backward()
                        self.optimizer_G.step()

                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real, _, _ = self.netD_A(real_A)
                        _, local_feat_A, global_feat_A = self.netD_B(real_A)
                        info_loss_A = self.config['scale'] * self.info_loss.compute_infomax_loss(local_feat_A, global_feat_A)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real,
                                                                               self.target_real) + info_loss_A
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake, _, _ = self.netD_A(fake_A.detach())
                        _, local_feat_A, global_feat_A = self.netD_B(fake_A)
                        info_loss_A = self.config['scale'] * self.info_loss.compute_infomax_loss(local_feat_A, global_feat_A)
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake,
                                                                               self.target_fake) + info_loss_A

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real, _, _ = self.netD_B(real_B)
                        _, local_feat_B, global_feat_B = self.netD_B(real_B)
                        info_loss_B = self.config['scale'] * self.info_loss.compute_infomax_loss(local_feat_B, global_feat_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real,
                                                                               self.target_real) + info_loss_B

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake, _, _ = self.netD_B(fake_B.detach())
                        _, local_feat_B, global_feat_B = self.netD_B(fake_B)
                        info_loss_B = self.config['scale'] * self.info_loss.compute_infomax_loss(local_feat_B, global_feat_B)
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake,
                                                                               self.target_fake) + info_loss_B

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ###################################



                else:  # s dir :NC
                    if self.config['regist']:  # NC+R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        #### regist sys loss
                        fake_B = self.netG_A2B(real_A)
                        Trans = self.R_A(fake_B, real_B)
                        self.HEA_loss = self.config['hea_lamda'] * DCLoss.DCLoss.loss(real_B, fake_B)
                        SysRegist_A2B = self.spatial_transform(fake_B, Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B, real_B)  ###SR
                        pred_fake, _, _ = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        ####smooth loss
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        toal_loss = SM_loss + adv_loss + SR_loss + self.HEA_loss
                        toal_loss.backward()
                        self.optimizer_R_A.step()
                        self.optimizer_G.step()

                        self.optimizer_D_B.zero_grad()
                        with torch.no_grad():
                            fake_B = self.netG_A2B(real_A)
                            # Real loss
                            pred_real, local_feat_B, global_feat_B = self.netD_B(real_B)
                            # _, local_feat_B, global_feat_B = self.netD_B(real_B)
                            info_loss_B = self.config['scale'] * self.info_loss.compute_infomax_loss(local_feat_B, global_feat_B)
                            loss_DB_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real,
                                                                                   self.target_real) + info_loss_B

                            # Fake loss
                            fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                            pred_fake, _, _ = self.netD_B(fake_B.detach())
                            _, local_feat_B, global_feat_B = self.netD_B(fake_B)
                            info_loss_B = self.config['scale'] * self.info_loss.compute_infomax_loss(local_feat_B, global_feat_B)
                            loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake,
                                                                                   self.target_fake) + info_loss_B

                            # Total loss
                            loss_D_B = (loss_DB_real + loss_D_fake)
                            loss_D_B.backward()

                            self.optimizer_D_B.step()



                    else:  # only NC
                        self.optimizer_G.zero_grad()
                        fake_B = self.netG_A2B(real_A)
                        #### GAN aligin loss
                        pred_fake, _, _ = self.netD_B(fake_B)
                        HEA_loss = self.config['hea_lamda'] * DCLoss.DCLoss.loss(real_B, fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real) + HEA_loss
                        adv_loss.backward()
                        self.optimizer_G.step()
                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()
                        # Real loss
                        pred_real, _, _ = self.netD_B(real_B)
                        _, local_feat_B, global_feat_B = self.netD_B(real_B)
                        info_loss_B = self.config['scale'] * self.info_loss.compute_infomax_loss(local_feat_B, global_feat_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real,
                                                                               self.target_real) + info_loss_B

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake, _, _ = self.netD_B(fake_B.detach())
                        _, local_feat_B, global_feat_B = self.netD_B(fake_B)
                        info_loss_B = self.config['scale'] * self.info_loss.compute_infomax_loss(local_feat_B, global_feat_B)
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake,
                                                                               self.target_fake) + info_loss_B

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ###################################

                self.logger.log({'loss_D_B': loss_D_B, 'info_loss_B':info_loss_B,'HEA_loss':HEA_loss, 'loss_total':loss_Total, 'loss_G_A2B': loss_GAN_A2B },
                                images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})  # ,'SR':SysRegist_A2B

            #         # Save models checkpoints
            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])

            # 每隔10epoch保存一次模型
            if epoch % 5 == 0:
                torch.save(self.netG_A2B.state_dict(), self.config['save_root'] +str(epoch)+ 'netG_A2B.pth')

            # torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')
            # torch.save(self.R_A.state_dict(), self.config['save_root'] + 'Regist.pth')
            # torch.save(netD_A.state_dict(), 'output/netD_A_3D.pth')
            # torch.save(netD_B.state_dict(), 'output/netD_B_3D.pth')

            #############val###############
            with torch.no_grad():
                MAE = 0
                num = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    fake_B = self.netG_A2B(real_A).detach().cpu().numpy().squeeze()
                    mae = self.MAE(fake_B, real_B)
                    MAE += mae
                    num += 1

                print('Val MAE:', MAE / num)

    def test(self, ):
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B.pth'))
        # self.R_A.load_state_dict(torch.load(self.config['save_root'] + 'Regist.pth'))
        with torch.no_grad():
            MAE = 0
            PSNR = 0
            SSIM = 0
            num = 0
            for i, batch in enumerate(self.val_data):
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()

                fake_B = self.netG_A2B(real_A)
                fake_B = fake_B.detach().cpu().numpy().squeeze()

                # 保存 fake_B 图像
                fake_B_img = np.clip((fake_B + 1) / 2.0, 0, 1)  # 将范围调整到 [0, 1]
                # 将 NumPy 数组转换为 PIL 图像
                fake_B_img = Image.fromarray((fake_B_img * 255).astype(np.uint8))
                save_path = os.path.join(self.config['image_save'], f"fake_B_batch_{i}.jpg")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fake_B_img.save(save_path)

                mae = self.MAE(fake_B, real_B)
                psnr = self.PSNR(fake_B, real_B)
                ssim = compare_ssim(fake_B, real_B, data_range=1)
                MAE += mae
                PSNR += psnr
                SSIM += ssim
                num += 1
            print('MAE:', MAE / num)
            print('PSNR:', PSNR / num)
            print('SSIM:', SSIM / num)

    def PSNR(self, fake, real):
        x, y = np.where(real != -1)  # Exclude background
        mse = np.mean(((fake[x][y] + 1) / 2. - (real[x][y] + 1) / 2.) ** 2)
        if mse < 1.0e-10:
            return 100
        else:
            PIXEL_MAX = 1
            return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    def MAE(self, fake, real):
        indices = np.where(real != -1)
        x = indices[0]
        y = indices[1]
        # Exclude background
        mae = np.abs(fake[x, y] - real[x, y]).mean()
        return mae / 2  # from (-1,1) normaliz  to (0,1)

    def save_deformation(self, defms, root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max, x_min = dir_x.max(), dir_x.min()
        y_max, y_min = dir_y.max(), dir_y.min()
        dir_x = ((dir_x - x_min) / (x_max - x_min)) * 255
        dir_y = ((dir_y - y_min) / (y_max - y_min)) * 255
        tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # tans_x[tans_x<=150] = 0
        tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
        tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # tans_y[tans_y<=150] = 0
        tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
        gradxy = cv2.addWeighted(tans_x, 0.5, tans_y, 0.5, 0)

        cv2.imwrite(root, gradxy)
