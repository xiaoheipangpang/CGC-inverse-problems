import torch
import torch.nn.functional as F
import numpy as np
import lpips
import time
import pandas as pd
from scipy import linalg
from torchvision import models
from torch.nn.functional import adaptive_avg_pool2d
import warnings

warnings.filterwarnings('ignore')


class InceptionV3(torch.nn.Module):
    """InceptionV3模型用于FID计算"""

    def __init__(self):
        super(InceptionV3, self).__init__()
        inception = models.inception_v3(pretrained=True)
        inception.eval()
        self.block1 = torch.nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block2 = torch.nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block3 = torch.nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
        )

        # self.block4 = torch.nn.Sequential(
        #     inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
        #     adaptive_avg_pool2d
        # )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = adaptive_avg_pool2d(x, output_size=(1, 1))
        return x.view(x.size(0), -1)


class MetricsCalculator:
    def __init__(self, device='cuda'):
        self.device = device

        # 初始化LPIPS模型
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.lpips_model.eval()

        # 初始化FID模型
        self.inception_model = InceptionV3().to(device)
        self.inception_model.eval()

    def calculate_psnr(self, img1, img2):
        """计算PSNR - 纯PyTorch实现"""
        # 确保图像在[0,1]范围内
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)

        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

    def calculate_ssim(self, img1, img2, window_size=11, size_average=True):
        """计算SSIM - 纯PyTorch实现"""

        # 基于: https://github.com/Po-Hsun-Su/pytorch-ssim

        def gaussian(window_size, sigma):
            gauss = torch.Tensor(
                [np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
            return gauss / gauss.sum()

        def create_window(window_size, channel):
            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
            return window

        def _ssim(img1, img2, window, window_size, channel, size_average=True):
            mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
            mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            if size_average:
                return ssim_map.mean()
            else:
                return ssim_map.mean(1).mean(1).mean(1)

        (_, channel, _, _) = img1.size()
        window = create_window(window_size, channel).to(img1.device)

        return _ssim(img1, img2, window, window_size, channel, size_average).item()

    def calculate_lpips(self, img1, img2):
        """计算LPIPS"""
        # 确保图像在[-1,1]范围内（LPIPS的要求）
        if img1.max() <= 1.0 and img1.min() >= 0:
            img1 = img1 * 2 - 1
        if img2.max() <= 1.0 and img2.min() >= 0:
            img2 = img2 * 2 - 1

        with torch.no_grad():
            return self.lpips_model(img1, img2).item()

    def calculate_fid(self, real_images, generated_images):
        """计算FID"""

        def get_activations(images, model, batch_size=50):
            """提取特征"""
            model.eval()
            activations = []
            device = next(model.parameters()).device  # 获取模型所在的设备

            with torch.no_grad():
                for i in range(0, len(images), batch_size):
                    batch = images[i:i + batch_size].to(device)  # 将batch移动到模型所在的设备
                    # 调整图像大小到299x299
                    batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                    pred = model(batch)
                    activations.append(pred.cpu().numpy())

            return np.concatenate(activations, axis=0)
        # 提取特征
        act1 = get_activations(real_images, self.inception_model)
        act2 = get_activations(generated_images, self.inception_model)

        # 计算统计量
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

        # 计算FID
        diff = mu1 - mu2
        covmean = linalg.sqrtm(sigma1.dot(sigma2))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def calculate_single_image_metrics(self, original, reconstructed):
        """计算单张图像的PSNR, SSIM, LPIPS"""
        # 确保图像在[0,1]范围内
        original = torch.clamp(original, 0, 1)
        reconstructed = torch.clamp(reconstructed, 0, 1)

        psnr_value = self.calculate_psnr(original, reconstructed)
        ssim_value = self.calculate_ssim(original, reconstructed)
        lpips_value = self.calculate_lpips(original, reconstructed)

        return {
            'psnr': psnr_value,
            'ssim': ssim_value,
            'lpips': lpips_value
        }

    def save_results_to_excel(self, results, filename):
        """保存结果到Excel文件"""
        df = pd.DataFrame(results)

        # 计算平均值（排除FID，因为FID是整体指标）
        avg_results = {
            'image_name': 'Average',
            'psnr': df['psnr'].mean(),
            'ssim': df['ssim'].mean(),
            'lpips': df['lpips'].mean(),
            'time': df['time'].mean(),  # 确保平均时间被计算
            'vram': df['vram'].mean() if 'vram' in df.columns else 0,
            'nfes': df['nfes'].mean() if 'nfes' in df.columns else 0
        }
        if 'Times (s)' in df.columns:
            avg_results['Times (s)'] = df['Times (s)'].mean()
        if 'VRAM (GB)' in df.columns:
            avg_results['VRAM (GB)'] = df['VRAM (GB)'].mean()

        # 如果有FID，添加到平均值行
        if 'fid' in df.columns and not df['fid'].isna().all():
            # FID对所有图像相同，取第一个非空值
            fid_values = df['fid'].dropna()
            if len(fid_values) > 0:
                avg_results['fid'] = fid_values.iloc[0]

        df_avg = pd.DataFrame([avg_results])
        final_df = pd.concat([df, df_avg], ignore_index=True)

        # 格式化数值显示
        for col in ['psnr', 'ssim', 'lpips', 'time', 'vram', 'Times (s)', 'VRAM (GB)']:
            if col in final_df.columns:
                if col == 'psnr':
                    final_df[col] = final_df[col].round(2)
                elif col in ['ssim', 'lpips']:
                    final_df[col] = final_df[col].round(4)
                elif col in ['time', 'Times (s)']:
                    final_df[col] = final_df[col].round(3)
                elif col in ['vram', 'VRAM (GB)']:
                    final_df[col] = final_df[col].round(2)

        final_df.to_excel(filename, index=False)
        print(f"结果已保存到: {filename}")

        # 打印平均指标
        print("\n=== 平均指标 ===")
        print(f"PSNR: {avg_results['psnr']:.2f} dB")
        print(f"SSIM: {avg_results['ssim']:.4f}")
        print(f"LPIPS: {avg_results['lpips']:.4f}")
        print(f"平均时间: {avg_results['time']:.3f} 秒/张")
        if 'vram' in avg_results and avg_results['vram'] > 0:
            print(f"平均显存: {avg_results['vram']:.2f} GB")
        if 'fid' in avg_results and not pd.isna(avg_results['fid']):
            print(f"FID: {avg_results['fid']:.2f}")

def create_excel_filename(task, method, dataset, output_dir="results"):
    """创建Excel文件名"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{task}_{method}_{dataset}_metrics.xlsx"
    return os.path.join(output_dir, filename)