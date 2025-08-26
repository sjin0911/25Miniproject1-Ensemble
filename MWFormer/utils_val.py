import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# This script is adapted from the following repository: https://github.com/JingyunLiang/SwinIR


def calculate_psnr(img1, img2, test_y_channel=True):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    assert img1.shape[2] == 3
    img1 = img1.cpu().numpy().astype(np.float64)
    img2 = img2.cpu().numpy().astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
        
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    #return compare_psnr(img1, img2, data_range=255)
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, test_y_channel=True):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    assert img1.shape[2] == 3
    img1 = img1.cpu().numpy().astype(np.float64)
    img2 = img2.cpu().numpy().astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, ' f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, ' f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=True):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def validation_stylevec(style_filter, net:torch.nn.Module, val_data_loader1, device):
    print("Start Val")
    psnr_list = []
    ssim_list = []
    name_list = []
    for batch_id, test_data in  enumerate(tqdm(val_data_loader1)):

        input_image, gt, imgname = test_data
        input_image = input_image.to(device)
        gt = gt.to(device)


        # --- Forward + Backward + Optimize --- #
        style_filter.eval()
        net.eval()
        feature_vec = style_filter(input_image)
        pred_image = net(input_image,feature_vec)

        gt = gt[0].permute(1,2,0)
        pred_image = pred_image[0].permute(1,2,0)
        psnr = calculate_psnr(255*gt, 255*pred_image.clamp(0,1)).item()
        ssim = calculate_ssim(255*gt, 255*pred_image.clamp(0,1)).item()
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        name_list.append(imgname)

        # #save
        #pred_image = pred_image.squeeze(0)
        #pred_image = 255*pred_image.clamp(0,255).cpu().numpy()
        #pred_image = cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR)
        #output_path = './results/snow/pred' + str(batch_id) + '.png'
        #cv2.imwrite(output_path, pred_image)
        #if batch_id==60:
        #    break
    return sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list)


#test unpaired data using the proposed model (without ground truth)
def validation_unpaired(style_filter, net:torch.nn.Module, val_data_loader1, device):
    print("Start Val")
    for batch_id, test_data in  enumerate(tqdm(val_data_loader1)):

        input_image, imgname = test_data
        input_image = input_image.to(device)


        # --- Forward + Backward + Optimize --- #
        style_filter.eval()
        net.eval()
        feature_vec = style_filter(input_image)
        pred_image = net(input_image,feature_vec)

        pred_image = pred_image[0].permute(1,2,0)

        # #save
        # pred_image = pred_image.squeeze(0)
        # pred_image = 255*pred_image.clamp(0,1).cpu().numpy()
        # pred_image = cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR)
        # output_path = './results/my_real/pred' + imgname + '.png'
        # #print(output_path)
        # cv2.imwrite(output_path, pred_image)

        # if batch_id==99:
        #     break


def validation_hybrid(style_filter, feature_vec1, feature_vec2, net:torch.nn.Module, val_data_loader1, device):
    print("Start Val")
    style_filter.eval()

    psnr_list = []
    ssim_list = []
    for batch_id, test_data in  enumerate(tqdm(val_data_loader1)):

        input_image, gt, imgname = test_data
        input_image = input_image.to(device)


        # --- Forward + Backward + Optimize --- #
        pred_image = net(input_image,feature_vec1)
        pred_image = pred_image.clamp(0,1)

        # #write stage1 result
        # output_path = './results/hybrid/' + str(imgname[0][:-4]) + 'stage1.png'
        # pred_image1 = pred_image
        # pred_image1 = pred_image1[0].permute(1,2,0)
        # pred_image1 = pred_image1.squeeze(0)
        # pred_image1 = 255*pred_image1.cpu().numpy()
        # pred_image1 = cv2.cvtColor(pred_image1, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(output_path, pred_image1)

        #calc
        pred_image = pred_image * 2 - 1
        feature_vec = style_filter(pred_image)
        pred_image = net(pred_image, feature_vec)

        gt = gt[0].permute(1,2,0)
        pred_image = pred_image[0].permute(1,2,0)
        psnr_list.append(calculate_psnr(255*gt, 255*pred_image.clamp(0,1)).item()), ssim_list.append(calculate_ssim(255*gt, 255*pred_image.clamp(0,1)).item())

        # #save stage2 result
        # pred_image = pred_image.squeeze(0)
        # pred_image = 255*pred_image.clamp(0,255).cpu().numpy()
        # pred_image = cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR)
        # output_path = './results/hybrid/' + str(imgname[0][:-4]) + 'pred.png'
        # #print(output_path)
        # cv2.imwrite(output_path, pred_image)
    return sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list)


def validation_hybrid_unpaired(style_filter, feature_vec1, feature_vec2, net:torch.nn.Module, val_data_loader1, device):
    print("Start Val")
    style_filter.eval()

    for batch_id, test_data in  enumerate(tqdm(val_data_loader1)):

        input_image, imgname = test_data
        input_image = input_image.to(device)


        # --- Forward + Backward + Optimize --- #
        pred_image = net(input_image,feature_vec1)
        pred_image = pred_image.clamp(0,1)

        # #write stage1 result
        # output_path = './results/hybrid/' + str(imgname[0][:-4]) + 'stage1.png'
        # pred_image1 = pred_image
        # pred_image1 = pred_image1[0].permute(1,2,0)
        # pred_image1 = pred_image1.squeeze(0)
        # pred_image1 = 255*pred_image1.cpu().numpy()
        # pred_image1 = cv2.cvtColor(pred_image1, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(output_path, pred_image1)

        #calc
        pred_image = pred_image * 2 - 1
        feature_vec2 = style_filter(pred_image)
        pred_image = net(pred_image, feature_vec2)
        pred_image = pred_image[0].permute(1,2,0)

        # #save stage2 result
        # pred_image = pred_image.squeeze(0)
        # pred_image = 255*pred_image.clamp(0,255).cpu().numpy()
        # pred_image = cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR)
        # output_path = './results/hybrid/' + str(imgname[0][:-4]) + 'pred.png'
        # #print(output_path)
        # cv2.imwrite(output_path, pred_image)

        # if batch_id==99:
        #     break