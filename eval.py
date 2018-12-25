import argparse
import torch
import cv2
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.transform import resize
import torch.nn as nn
from PIL import Image 
from scipy.signal import convolve2d

parser = argparse.ArgumentParser(description="PyTorch LapSRN Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_100.pth", type=str, help="model path")
parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")
parser.add_argument("--scale", default=1, type=int, help="scale factor, Default: 2")

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

image_list = glob.glob(opt.dataset+"/*.*") 

avg_psnr_predicted = 0.0
avg_psnr_bicubic = 0.0
avg_elapsed_time = 0.0
avg_ssim_predicted = 0.0
avg_ssim_bicubic = 0.0
for image_name in image_list:
    print("Processing ", image_name)
    im_gt_y = sio.loadmat(image_name)['im_gt_y']
    im_b_y = sio.loadmat(image_name)['im_b_y']
    im_l_y = sio.loadmat(image_name)['im_l_y']

    im_gt_y = im_gt_y.astype(np.float64)
    im_b_y = im_b_y.astype(np.float64)
    im_l_y = im_l_y.astype(np.float64)
    #psnr_bicubic = compare_psnr(im_gt_y, im_b_y,255,255)
    psnr_bicubic = PSNR(im_gt_y, im_b_y,shave_border=opt.scale)
    avg_psnr_bicubic += psnr_bicubic

    im_input = im_l_y/255.

    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

    if cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        im_input = im_input.cuda()
        #im_input = torch.nn.DataParallel(im_input)
    else:
        model = model.cpu()

    start_time = time.time()
    #HR_2x, HR_4x = model(im_input)
    HR_2x = model(im_input)
    elapsed_time = time.time() - start_time
    avg_elapsed_time += elapsed_time

    HR_2x = HR_2x.cpu()

    im_h_y = HR_2x.data[0].numpy().astype(np.float64)

    im_h_y = im_h_y*255.
    im_h_y[im_h_y<0] = 0
    im_h_y[im_h_y>255.] = 255.
    im_h_y = im_h_y[0,:,:]


    psnr_predicted = PSNR(im_gt_y,im_h_y,shave_border=opt.scale)


    #im_gt_y = im_gt_y.astype(np.float32)
    #im_h_y = im_h_y.astype(np.float)
    #(score_predicted, diff) = compare_ssim(im_gt_y, im_h_y,win_size=15, full=True)
    score_predicted = compute_ssim(np.array(im_gt_y),np.array(im_h_y))
    #psnr_predicted = compare_psnr(im_h_y,im_gt_y,255,255)
    avg_psnr_predicted += psnr_predicted
    avg_ssim_predicted += score_predicted
    #print ("ssim_predicted=", score_predicted)
    #(score_bicubic, diff) = compare_ssim(im_gt_y, im_b_y,win_size=15, full=True)
    score_bicubic = compute_ssim(np.array(im_gt_y),np.array(im_b_y))
    avg_ssim_bicubic += score_bicubic
    #print ("ssim_bicubic=", score_bicubic)

print("Scale=", opt.scale)
print("Dataset=", opt.dataset)
print("PSNR_predicted=", avg_psnr_predicted/len(image_list))
print("PSNR_bicubic=", avg_psnr_bicubic/len(image_list))
print("SSIM_predicted=", avg_ssim_predicted/len(image_list))
print("SSIM_bicubic=", avg_ssim_bicubic/len(image_list))
print("It takes average {}s for processing".format(avg_elapsed_time/len(image_list)))
