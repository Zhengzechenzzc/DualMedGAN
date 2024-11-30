import numpy as np
import pandas as pd
import skimage
import torch

# gray = color.rgb2gray(img)

# GLCM properties
def contrast_feature(matrix_coocurrence):
   contrast = skimage.feature.graycoprops(matrix_coocurrence, 'contrast')
   return contrast

def dissimilarity_feature(matrix_coocurrence):
   dissimilarity = skimage.feature.graycoprops(matrix_coocurrence, 'dissimilarity')
   return dissimilarity

def tuple2df(input):
   glcm_df = pd.DataFrame(list(input))
   return glcm_df

def dfdiff(y, yhat):
   return (y - yhat).transpose()

def diff_describe(input):
   des = input.describe().transpose()
   return des

def norm(x, des):
   # return (x - des['min']) / (des['max']-des['min'])
   return (x - des['mean']) / des['std']


def scale_data(data):
   # 将数据从 [-1, 1] 缩放到 [0, 2]
   scaled_data = (data + 1) * 2
   # 将数据从 [0, 2] 缩放到 [0, 255]
   scaled_data = (scaled_data * 255).astype(np.uint8)

   return scaled_data


class DCLoss():

   def glcmsix(img):
      image = img
      # image = img_as_ubyte(img)
      # bins = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 88, 96, 104, 112, 120, 128,
      #                  132, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 255])  # 16-bit

      bins = np.array(range(0, 255, 8))
      inds = np.digitize(image, bins)
      # print(inds)
      # print(len(image))
      # if inds.all() == 0:
      #    inds[inds <= 0] = 1 #TODO 元组inds填充为1
      max_value = inds.max() + 1
      matrix_coocurrence = skimage.feature.graycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                             levels=max_value,
                                             normed=False, symmetric=False)
      conf = tuple2df(contrast_feature(matrix_coocurrence)) # 对比度
      disf = tuple2df(dissimilarity_feature(matrix_coocurrence)) # 差异性
      # df_merge = pd.concat([homf , enef, asmf]) # , homf , enef, asmf
      df_merge = pd.concat([conf, disf])

      return df_merge

   def loss(y_real, y_fake):
      # real = io.imread(y_real)
      # fake = io.imread(y_fake)
      batch_loss = []
      for i in range(y_real.shape[0]):
         # real4d = y_real[i:i + 1, 1:2, :, :]
         # feak4d = y_fake[i:i + 1, 1:2, :, :]
         real4d = y_real[i:i + 1, :, :, :]
         feak4d = y_fake[i:i + 1, :, :, :]
         real2d = real4d.squeeze()
         feak2d = feak4d.squeeze()
         real2d_cpu = real2d.cpu()
         feak2d_cpu = feak2d.cpu()
         real2d_arr = real2d_cpu.detach().numpy()
         feak2d_arr = feak2d_cpu.detach().numpy()#TODO 将矩阵[1,1,128,128]转为[128,128]

         real2d_arr2 = scale_data(real2d_arr)
         feak2d_arr2 = scale_data(feak2d_arr)

         real_glcm = DCLoss.glcmsix(real2d_arr2)
         fake_glcm = DCLoss.glcmsix(feak2d_arr2)

         diff = dfdiff(real_glcm, fake_glcm)
         diff_mean = diff.mean()

         diff_loss = abs(diff_mean[0:1]) + abs(diff_mean[1:2]) * 10 + abs(diff_mean[2:3]) * 100
         batch_loss.append(diff_loss)

      loss = sum(batch_loss).values
      loss_cpu = torch.from_numpy(loss)
      loss_gpu = loss_cpu.cuda()
      return loss_gpu



