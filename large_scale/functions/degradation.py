import numpy as np
import torch
from munch import Munch

import functions.svd_operators as svd_op
from functions import measurements
from utils.inpaint_util import MaskGenerator

__DEGRADATION__ = {}

def register_degradation(name: str):
    def wrapper(fn):
        if __DEGRADATION__.get(name) is not None:
            raise NameError(f'DEGRADATION {name} is already registered')
        __DEGRADATION__[name]=fn
        return fn
    return wrapper

def get_degradation(name: str,
                    deg_config: Munch,
                    device:torch.device):
    if __DEGRADATION__.get(name) is None:
        raise NameError(f'DEGRADATION {name} does not exist.')
    return __DEGRADATION__[name](deg_config, device)

@register_degradation(name='cs_walshhadamard')
def deg_cs_walshhadamard(deg_config, device):
    compressed_size = round(1/deg_config.deg_scale)
    A_funcs = svd_op.WalshHadamardCS(deg_config.channels,
                                     deg_config.image_size,
                                     compressed_size,
                                     torch.randperm(deg_config.image_size**2),
                                     device)
    return A_funcs

@register_degradation(name='cs_blockbased')
def deg_cs_blockbased(deg_config, device):
    cs_ratio = deg_config.deg_scale
    A_funcs = svd_op.CS(deg_config.channels,
                        deg_config.image_size,
                        cs_ratio,
                        device)
    return A_funcs

@register_degradation(name='inpainting')
def deg_inpainting(deg_config, device):
    """动态生成中心矩形mask的图像修复"""

    # 使用 deg_scale 作为 mask_size
    mask_size = deg_config.deg_scale

    image_size = deg_config.image_size

    # 确保 mask_size 合理
    mask_size = max(16, min(mask_size, image_size))  # 限制在16到图像尺寸之间

    print(f"图像修复 - 图像尺寸: {image_size}x{image_size}, 修复区域: {mask_size}x{mask_size}")

    # 生成中心矩形区域
    center = image_size // 2
    half_mask = mask_size // 2

    start_row = max(0, center - half_mask)
    end_row = min(image_size, start_row + mask_size)
    start_col = max(0, center - half_mask)
    end_col = min(image_size, start_col + mask_size)

    # 调整以确保大小为 mask_size
    if (end_row - start_row) < mask_size:
        if end_row == image_size:
            start_row = end_row - mask_size

    if (end_col - start_col) < mask_size:
        if end_col == image_size:
            start_col = end_col - mask_size

    # 生成缺失像素索引
    total_pixels = image_size * image_size
    missing_list = []

    for i in range(start_row, end_row):
        for j in range(start_col, end_col):
            idx = i * image_size + j
            if 0 <= idx < total_pixels:
                missing_list.append(idx)

    missing = torch.tensor(missing_list, device=device, dtype=torch.long)

    # 复制到三个通道
    missing_r = missing * 3
    missing_g = missing_r + 1
    missing_b = missing_g + 1
    missing_all = torch.cat([missing_r, missing_g, missing_b], dim=0)

    print(f"修复区域: 左上角({start_row},{start_col}), 右下角({end_row},{end_col})")
    print(f"缺失像素数: {len(missing)} 个像素点")

    A_funcs = svd_op.Inpainting(deg_config.channels,
                                deg_config.image_size,
                                missing_all,
                                device)
    return A_funcs



# @register_degradation(name='inpainting')
# def deg_inpainting(deg_config, device):
#     # TODO: generate mask rather than load
#     loaded = np.load("exp/inp_masks/mask_768_half.npy")  # block
#     # loaded = np.load("lip_mask_4.npy")
#     mask = torch.from_numpy(loaded).to(device).reshape(-1)
#     missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
#     missing_g = missing_r + 1
#     missing_b = missing_g + 1
#     missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
#     A_funcs = svd_op.Inpainting(deg_config.channels,
#                                 deg_config.image_size,
#                                 missing,
#                                 device)
#     return A_funcs

@register_degradation(name='denoising')
def deg_denoise(deg_config, device):
    A_funcs = svd_op.Denoising(deg_config.channels,
                               deg_config.image_size,
                               device)
    return A_funcs

@register_degradation(name='colorization')
def deg_colorization(deg_config, device):
    A_funcs = svd_op.Colorization(deg_config.image_size,
                                  device)
    return A_funcs


@register_degradation(name='sr_avgpool')
def deg_sr_avgpool(deg_config, device):
    blur_by = int(deg_config.deg_scale)
    A_funcs = svd_op.SuperResolution(deg_config.channels,
                                     deg_config.image_size,
                                     blur_by,
                                     device)
    return A_funcs

@register_degradation(name='sr_bicubic')
def deg_sr_bicubic(deg_config, device):
    def bicubic_kernel(x, a=-0.5):
        if abs(x) <= 1:
            return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
        elif 1 < abs(x) and abs(x) < 2:
            return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
        else:
            return 0

    factor = int(deg_config.deg_scale)
    k = np.zeros((factor * 4))
    for i in range(factor * 4):
        x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
        k[i] = bicubic_kernel(x)
    k = k / np.sum(k)
    kernel = torch.from_numpy(k).float().to(device)
    A_funcs = svd_op.SRConv(kernel / kernel.sum(),
                            deg_config.channels,
                            deg_config.image_size,
                            device,
                            stride=factor)
    return A_funcs

@register_degradation(name='deblur_uni')
def deg_deblur_uni(deg_config, device):
    A_funcs = svd_op.Deblurring(torch.tensor([1/deg_config.deg_scale]*deg_config.deg_scale).to(device),
                                deg_config.channels,
                                deg_config.image_size,
                                device)
    return A_funcs

@register_degradation(name='deblur_gauss')
def deg_deblur_gauss(deg_config, device):
    sigma = 3.0
    pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
    size = deg_config.deg_scale
    ker = []
    for k in range(-size//2, size//2):
        ker.append(pdf(k))
    kernel = torch.tensor(ker, dtype=torch.float32).to(device)
    A_funcs = svd_op.Deblurring(kernel / kernel.sum(),
                                deg_config.channels,
                                deg_config.image_size,
                                device)
    return A_funcs

@register_degradation(name='deblur_aniso')
def deg_deblur_aniso(deg_config, device):
    sigma = 20
    pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
    kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)

    sigma = 1
    pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
    kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)

    A_funcs = svd_op.Deblurring2D(kernel1 / kernel1.sum(),
                                  kernel2 / kernel2.sum(),
                                  deg_config.channels,
                                  deg_config.image_size,
                                  device)
    return A_funcs

@register_degradation(name='deblur_motion')
def deg_deblur_motion(deg_config, device):
    A_funcs = measurements.MotionBlurOperator(
        kernel_size=deg_config.deg_scale,
        intensity=0.5,
        device=device
    )
    return A_funcs

@register_degradation(name='deblur_nonuniform')
def deg_deblur_motion(deg_config, device, kernels=None, masks=None):
    A_funcs = measurements.NonuniformBlurOperator(
        deg_config.image_size,
        deg_config.deg_scale,
        device,
        kernels=kernels,
        masks=masks,
    )
    return A_funcs


# ======= FOR arbitraty image size =======
@register_degradation(name='sr_avgpool_gen')
def deg_sr_avgpool_general(deg_config, device):
    blur_by = int(deg_config.deg_scale)
    A_funcs = svd_op.SuperResolutionGeneral(deg_config.channels,
                                            deg_config.imgH,
                                            deg_config.imgW,
                                            blur_by,
                                            device)
    return A_funcs

@register_degradation(name='deblur_gauss_gen')
def deg_deblur_guass_general(deg_config, device):
    A_funcs = measurements.GaussialBlurOperator(
        kernel_size=deg_config.deg_scale,
        intensity=3.0,
        device=device
    )
    return A_funcs


from functions.jpeg import jpeg_encode, jpeg_decode

class JPEGOperator():
    def __init__(self, qf: int, device):
        self.qf = qf
        self.device = device

    def A(self, img):
        x_luma, x_chroma = jpeg_encode(img, self.qf)
        return x_luma, x_chroma

    def At(self, encoded):
        return jpeg_decode(encoded, self.qf)


@register_degradation(name='jpeg')
def deg_jpeg(deg_config, device):
    A_funcs = JPEGOperator(
        qf = deg_config.deg_scale,
        device=device
    )
    return A_funcs
