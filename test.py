import argparse
import torch
import cv2
from torch.autograd import Variable
import numpy as np
import time, math
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.transform import resize
import torch.nn as nn
from PIL import Image
from scipy.signal import convolve2d

parser = argparse.ArgumentParser(description="PyTorch LapSRN Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_100.pth", type=str, help="model path")
parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))

opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = torch.load(opt.model)["model"]
im_gt_y = sio.loadmat(opt.image)['im_gt_y']
im_b_y = sio.loadmat(opt.image)['im_b_y']
im_l_y = sio.loadmat(opt.image)['im_l_y']

#im_gt_y = sio.loadmat("Set5/" + opt.image + ".mat")['im_gt_y']
#im_b_y = sio.loadmat("Set5/" + opt.image + ".mat")['im_b_y']
#im_l_y = sio.loadmat("Set5/" + opt.image + ".mat")['im_l_y']

im_gt_y = im_gt_y.astype(float)
im_b_y = im_b_y.astype(float)
im_l_y = im_l_y.astype(float)

psnr_bicubic = PSNR(im_gt_y, im_b_y,shave_border=opt.scale)
#(score_bicubic, diff) = compare_ssim(im_gt_y, im_b_y,win_size=25, full=True)
score_bicubic = compute_ssim(np.array(im_gt_y),np.array(im_b_y))

im_input = im_l_y/255.

im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

if cuda:
    model = model.cuda()
    im_input = im_input.cuda()
else:
    model = model.cpu()
    
start_time = time.time()
HR_2x, HR_4x = model(im_input)
elapsed_time = time.time() - start_time

HR_4x = HR_4x.cpu()

im_h_y = HR_4x.data[0].numpy().astype(np.float)

im_h_y = im_h_y*255.
im_h_y[im_h_y<0] = 0
im_h_y[im_h_y>255.] = 255.
im_h_y = im_h_y[0,:,:]

psnr_predicted = PSNR(im_gt_y, im_h_y,shave_border=opt.scale)
#(score_predicted, diff) = compare_ssim(im_gt_y, im_h_y, win_size=25, full=True)
score_predicted = compute_ssim(np.array(im_gt_y),np.array(im_h_y))

print("Scale=",opt.scale)
print("PSNR_predicted=", psnr_predicted)
print("SSIM_predicted=", score_predicted)
print("PSNR_bicubic=", psnr_bicubic)
print("SSIM_bicubic=", score_bicubic)
print("It takes {}s for processing".format(elapsed_time))

fig = plt.figure()
ax = plt.subplot("131")
ax.imshow(im_gt_y, cmap='gray')
ax.set_title("GT")

ax = plt.subplot("132")
ax.imshow(im_b_y, cmap='gray')
ax.set_title("Input(Bicubic)")

ax = plt.subplot("133")
ax.imshow(im_h_y, cmap='gray')
ax.set_title("Output(LapSRN)")
plt.savefig("./test-out-x4/"+opt.image+".png")
plt.show()
