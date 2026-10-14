import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms
from torchvision.utils import save_image
from util import straightness, parse_config, save_traj
import argparse
from tqdm import tqdm
from network_edm import SongUNet, DhariwalUNet, EDMPrecondVel
from torch.nn import DataParallel
import json
import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy as np
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from inverse_operators import *
from metrics_calculator import MetricsCalculator, create_excel_filename

# ========== 预设配置 ==========
PRESET_CONFIGS = {
    # AFHQ 64x64 配置
    "rfoc_afhq64_box": {
        "gpu": "0",
        "ckpt": "./checkpoints/afhq-configF.pth",
        "config": "configs_unet/afhq64_ve_aug.json",
        "input_dir": "./data/AFHQ64",
        "inverse_problem": "box_inpaint",
        "mask_size": 32,
        "N": 100,
        "batchsize": 1,
        "gradient_scale": 500
    },
    "rfoc_afhq64_sr_4": {
        "gpu": "0",
        "ckpt": "./checkpoints/afhq-configF.pth",
        "config": "configs_unet/afhq64_ve_aug.json",
        "input_dir": "./data/AFHQ64",
        "inverse_problem": "super_resolution",
        "scale_factor": 4,
        "N": 100,
        "batchsize": 1,
        "gradient_scale": 500
    },
    "rfoc_afhq64_deblur": {
        "gpu": "0",
        "ckpt": "./checkpoints/afhq-configF.pth",
        "config": "configs_unet/afhq64_ve_aug.json",
        "input_dir": "./data/AFHQ64",
        "inverse_problem": "deblur",
        "kernel_size": 11,
        "blur_sigma": 5.0,
        "noise_sigma": 0.0,
        "N": 100,
        "batchsize": 1,
        "gradient_scale": 500
    },
    "rfoc_afhq64_motion_deblur": {
        "gpu": "0",
        "ckpt": "./checkpoints/afhq-configF.pth",
        "config": "configs_unet/afhq64_ve_aug.json",
        "input_dir": "./data/AFHQ64",
        "inverse_problem": "motion_deblur",
        "kernel_size": 15,
        "intensity": 0.5,
        "noise_sigma": 0.05,
        "N": 100,
        "batchsize": 1,
        "gradient_scale": 500,
        "solver": "euler",
        "sampler": "new"
    },
    # Celeba64 配置
    "rfoc_celeba64_box": {
        "gpu": "0",
        "ckpt": "./checkpoints/ffhq-configE.pth",
        "config": "configs_unet/ffhq64_ve_aug.json",
        "input_dir": "./data/Celeba64",
        "inverse_problem": "box_inpaint",
        "mask_size": 32,
        "N": 100,
        "batchsize": 1,
        "gradient_scale": 500
    },
    "rfoc_celeba64_sr_4": {
        "gpu": "0",
        "ckpt": "./checkpoints/ffhq-configE.pth",
        "config": "configs_unet/ffhq64_ve_aug.json",
        "input_dir": "./data/Celeba64",
        "inverse_problem": "super_resolution",
        "scale_factor": 4,
        "N": 100,
        "batchsize": 1,
        "gradient_scale": 500
    },
    "rfoc_celeba64_deblur": {
        "gpu": "0",
        "ckpt": "./checkpoints/ffhq-configE.pth",
        "config": "configs_unet/ffhq64_ve_aug.json",
        "input_dir": "./data/Celeba64",
        "inverse_problem": "deblur",
        "kernel_size": 11,
        "blur_sigma": 5.0,
        "noise_sigma": 0.0,
        "N": 100,
        "batchsize": 1,
        "gradient_scale": 500
    },
    "rfoc_celeba64_motion_deblur": {
        "gpu": "0",
        "ckpt": "./checkpoints/ffhq-configE.pth",
        "config": "configs_unet/ffhq64_ve_aug.json",
        "input_dir": "./data/Celeba64",
        "inverse_problem": "motion_deblur",
        "kernel_size": 15,
        "intensity": 0.5,
        "noise_sigma": 0.05,
        "N": 100,
        "batchsize": 1,
        "gradient_scale": 500,
        "solver": "euler",
        "sampler": "new"
    },
    "rfoc_imagenet_0.05denoise": {
        "gpu": "0",
        "ckpt": "./checkpoints/imagenet-configE.pth",
        "config": "configs_unet/imagenet64.json",
        "input_dir": "./data/ImageNet64",
        "inverse_problem": "denoise",
        "noise_sigma": 0.05,
        "N": 100,
        "batchsize": 1,
        "gradient_scale": 500
    },
    "rfoc_imagenet_0.1denoise": {
        "gpu": "0",
        "ckpt": "./checkpoints/imagenet-configE.pth",
        "config": "configs_unet/imagenet64.json",
        "input_dir": "./data/ImageNet64",
        "inverse_problem": "denoise",
        "noise_sigma": 0.1,
        "N": 100,
        "batchsize": 1,
        "gradient_scale": 500
    }
}


def get_args():
    parser = argparse.ArgumentParser(description='Configs')
    # 预设配置参数
    parser.add_argument('--preset', type=str, default='rfoc_afhq64_box',
                        choices=[
                            # AFHQ64 配置
                            'rfoc_afhq64_box',
                            'rfoc_afhq64_sr_4',
                            'rfoc_afhq64_deblur',
                            'rfoc_afhq64_motion_deblur',
                            # CelebA64 配置
                            'rfoc_celeba64_box',
                            'rfoc_celeba64_sr_4',
                            'rfoc_celeba64_deblur',
                            'rfoc_celeba64_motion_deblur',
                            # ImageNet64 配置
                            'rfoc_imagenet_0.05denoise',
                            'rfoc_imagenet_0.1denoise',
                        ])
    # 原有参数保持不变
    parser.add_argument('--gpu', type=str, help='gpu index')
    parser.add_argument('--dir', type=str, help='Saving directory name')
    parser.add_argument('--ckpt', type=str, default=None, help='Flow network checkpoint')
    parser.add_argument('--batchsize', type=int, default=1, help='Batch size')
    parser.add_argument('--N', type=int, default=20, help='Number of sampling steps')
    parser.add_argument('--save_traj', action='store_true', help='Save the trajectories')
    parser.add_argument('--save_z', action='store_true', help='Save zs for distillation')
    parser.add_argument('--save_data', action='store_true', help='Save data')
    parser.add_argument('--solver', type=str, default='euler', help='ODE solvers')
    parser.add_argument('--config', type=str, default=None, help='Decoder config path, must be .json file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--im_dir', type=str, help='Image dir')
    parser.add_argument('--action', type=str, default='sample', help='sample or interpolate')
    parser.add_argument('--compile', action='store_true', help='Compile the model')
    parser.add_argument('--t_steps', type=str, default=None, help='t_steps, e.g. 1,0.75,0.5,0.25')
    parser.add_argument('--sampler', type=str, default='new', help='default / new')

    # Inversion
    parser.add_argument('--data_path', type=str, default=None, help='Image path for inversion')
    parser.add_argument('--label_inv', type=int, help='Label for inversion')
    parser.add_argument('--label_rec', type=int, help='Label for reconstruction')
    parser.add_argument('--N_decode', type=int, default=5, help='Number of decoding sampling steps')

    # Inverse problem arguments
    parser.add_argument('--inverse_problem', type=str, default='none',
                        help='Type of inverse problem (deblur/box_inpaint/super_resolution)')
    parser.add_argument('--noise_sigma', type=float, default=0.0, help='Noise sigma for degradation')
    parser.add_argument('--kernel_size', type=int, default=11, help='Kernel size for deblurring')
    parser.add_argument('--blur_sigma', type=float, default=5.0, help='Blur sigma for deblurring')
    parser.add_argument('--mask_size', type=int, default=20, help='Box size for inpainting')
    parser.add_argument('--scale_factor', type=int, default=4, help='Scale factor for super resolution')
    parser.add_argument('--input_dir', type=str, help='Directory containing input images')

    parser.add_argument('--gradient_scale', type=float, default=100.0, help='Scale factor for gradients')
    parser.add_argument('--likebaseline', action='store_true', help='Like baseline disable mean loss')

    # iLQR相关参数
    parser.add_argument('--ilqr_iterations', type=int, default=1, help='Number of iLQR iterations')
    parser.add_argument('--show_ilqr', action='store_true', help='显示iLQR迭代信息')

    # 指标计算参数
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for metrics results')
    parser.add_argument('--skip_fid', action='store_true', help='Skip FID calculation to save time')

    args = parser.parse_args()

    # 如果指定了预设配置，覆盖相应参数
    if args.preset:
        preset_config = PRESET_CONFIGS[args.preset]
        for key, value in preset_config.items():
            if hasattr(args, key):
                setattr(args, key, value)

        # 自动设置输出目录
        if args.dir is None:
            args.dir = f"outputs/{args.preset}"

    return args


def _terminal_gradient(x0hat, downsampled, operator, likebaseline=False):

    with torch.enable_grad():
        x0hat_g = x0hat.detach().requires_grad_()
        if likebaseline:
            loss = 0.5 * torch.mean((operator.degradation(x0hat_g) - downsampled) ** 2)
        else:
            loss = torch.mean(
                F.mse_loss(operator.degradation(x0hat_g), downsampled, reduction="none") *
                operator.get_mask(shape=downsampled.shape)
            )
        g = torch.autograd.grad(loss, x0hat_g)[0]
    return g.detach()


@torch.no_grad()
def sample_ode_generative(model, z1, N, use_tqdm=True, solver='euler', label=None, inversion=False,
                          time_schedule=None, sampler='default', operator=None, ref_img_path="", output_dir="",
                          gradient_scale=500, likebaseline=False, show_ilqr=False, ilqr_iterations=1):


    assert solver in ['euler', 'heun']
    assert len(z1.shape) == 4
    assert operator is not None
    assert ref_img_path != ""

    downsample = operator.degradation
    upsample = operator.degradation_transpose

    if inversion:
        assert sampler == 'default'
    tq = tqdm if use_tqdm else lambda x: x

    if solver == 'heun':
        if N % 2 == 0:
            raise ValueError("N must be odd when using Heun's method.")
        N = (N + 1) // 2

    z1 = z1.detach()
    batchsize = z1.shape[0]
    device = z1.device

    # ---------- 时间调度 ----------
    if time_schedule is not None:
        time_schedule = time_schedule + [0]
        sigma_schedule = [t_ / (1 - t_ + 1e-6) for t_ in time_schedule]
        print(f"sigma_schedule: {sigma_schedule}")
    else:
        t_func = lambda i: i / N
        if inversion:
            time_schedule = [t_func(i) for i in range(0, N)] + [1]
            time_schedule[0] = 1e-3
        else:
            time_schedule = [t_func(i) for i in reversed(range(1, N + 1))] + [0]
            time_schedule[0] = 1 - 1e-5

    # ---------- 图像预处理 ----------
    image = Image.open(ref_img_path).convert("RGB").resize((64, 64))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    preprocessed = transform(image).unsqueeze(0).to(device)

    downsampled = downsample(preprocessed)
    upsampled = upsample(downsampled)
    filename = os.path.splitext(os.path.basename(ref_img_path))[0]
    save_image(upsampled * 0.5 + 0.5, os.path.join(output_dir, f"{filename}_degraded.png"))

    # ---------- 标签 ----------
    config = model.module.config if hasattr(model, 'module') else model.config
    if config["label_dim"] > 0 and label is None:
        label = torch.randint(0, config["label_dim"], (batchsize,)).to(device)
        label = F.one_hot(label, num_classes=config["label_dim"]).type(torch.float32)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    # ==================================================================
    # CGC 初始化
    # ==================================================================
    num_steps = len(time_schedule) - 1
    u_k = [torch.zeros_like(z1) for _ in range(num_steps)]  # u_k^(0) = 0
    z_init = z1.clone()

    alpha = 1.0 / gradient_scale  # 控制权重 α
    max_iters = max(1, ilqr_iterations)
    delta_u_threshold = 1e-3

    if show_ilqr:
        print(f"Starting CGC-iLQR: max_iters={max_iters}, alpha={alpha:.6f}, eps={delta_u_threshold}")

    traj, x0hat_list = [], []

    # ==================================================================
    # 外层 iLQR 迭代
    # ==================================================================
    converged = False
    for iter_idx in range(max_iters):
        if show_ilqr:
            print(f"[iLQR] Iteration {iter_idx + 1}/{max_iters}")

        # ---------- Step 1: Forward Rollout ----------
        z = z_init.clone()
        traj_iter = [z.detach().clone()]
        x0hat_iter = []
        u_k_new = [None] * num_steps

        for i in tq(range(num_steps)):
            t = torch.ones((batchsize), device=device) * time_schedule[i]
            t_next = torch.ones((batchsize), device=device) * time_schedule[i + 1]
            dt = t_next[0] - t[0]

            # x_k^(i) = h_RF++(x_k^(i) + u_k^(i))
            z_controlled = z + u_k[i]

            vt = model(z_controlled, t, label)
            x0hat = z_controlled - vt * t.view(-1, 1, 1, 1)

            if solver == 'heun' and i < N - 1:
                z_next = z_controlled.detach().clone() + vt * dt
                vt_next = model(z_next, t_next, label)
                vt = (vt + vt_next) / 2
                x0hat = z_next - vt_next * t_next.view(-1, 1, 1, 1)

            x0hat_iter.append(x0hat.detach().clone())

            if i < num_steps - 1:
                # 计算终端梯度 g̃_0^(k)
                g_tilde = _terminal_gradient(x0hat, downsampled, operator, likebaseline)
                control_signal = gradient_scale * g_tilde

                # 施加数据一致性控制
                z_after_ctrl = z_controlled.detach() - control_signal

                # 记录控制序列 u_k^(i+1) = u_k^(i) - (1/α)·g̃_0^(k)
                u_k_new[i] = u_k[i] - alpha * g_tilde
            else:
                z_after_ctrl = z_controlled
                u_k_new[i] = u_k[i]

            # 状态前进
            z = z_after_ctrl.detach().clone() + vt * dt
            traj_iter.append(z.detach().clone())

        # ---------- Step 2: Update Controls ----------
        delta_u_sq = 0.0
        for i in range(num_steps):
            delta_u_i = u_k_new[i] - u_k[i]
            delta_u_sq += torch.sum(delta_u_i ** 2).item()
            u_k[i] = u_k_new[i].detach().clone()

        delta_u = delta_u_sq ** 0.5 if delta_u_sq > 0 else 0.0

        # 保存本次迭代结果
        traj = traj_iter
        x0hat_list = x0hat_iter

        if show_ilqr:
            print(f"[iLQR]   Δu = {delta_u:.6e}")

        # ---------- 收敛判据 ----------
        if iter_idx == 0:
            converged = True
            if show_ilqr:
                print(f"[iLQR] Converged at iteration {iter_idx + 1} (Δu={delta_u:.6e})")
            break
        if delta_u < delta_u_threshold:
            converged = True
            if show_ilqr:
                print(f"[iLQR] Converged at iteration {iter_idx + 1} (Δu={delta_u:.6e})")
            break

    if show_ilqr and not converged:
        print(f"[iLQR] Reached max iterations ({max_iters})")

    if show_ilqr:
        final_loss = F.mse_loss(operator.degradation(traj[-1]), downsampled).item()
        print(f"[iLQR] Final data consistency loss: {final_loss:.6f}")

    end_time = time.time()
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    total_time = end_time - start_time

    return traj, x0hat_list, max_memory, total_time, preprocessed


def main(arg):

    if not os.path.exists(arg.dir):
        os.makedirs(arg.dir)
    assert arg.config is not None
    config = parse_config(arg.config)
    arg.res = config['img_resolution']
    arg.input_nc = config['in_channels']
    arg.label_dim = config['label_dim']

    # 初始化指标计算器
    metrics_calculator = MetricsCalculator(device=torch.device("cuda"))

    # 存储所有结果
    all_results = []
    all_original_images = []
    all_reconstructed_images = []

    # 直接创建目录，不检查是否存在
    os.makedirs(os.path.join(arg.dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(arg.dir, "zs"), exist_ok=True)
    os.makedirs(os.path.join(arg.dir, "trajs"), exist_ok=True)
    os.makedirs(os.path.join(arg.dir, 'tmp'), exist_ok=True)
    os.environ['TMPDIR'] = os.path.join(arg.dir, 'tmp')

    # Initialize model
    model_class = DhariwalUNet if config['unet_type'] == 'adm' else SongUNet
    flow_model = model_class(**config)

    # Setup device
    device_ids = arg.gpu.split(',')
    device = torch.device("cuda")
    print(f"Using {'multiple' if len(device_ids) > 1 else ''} GPU {arg.gpu}!")

    pytorch_total_params = sum(p.numel() for p in flow_model.parameters()) / 1000000
    print(f"Total parameters: {pytorch_total_params}M")

    flow_model = EDMPrecondVel(flow_model, use_fp16=config.get('use_fp16', False))

    if arg.ckpt is None:
        raise ValueError("Model checkpoint must be provided")
    flow_model.load_state_dict(torch.load(arg.ckpt, map_location="cpu"))

    if len(device_ids) > 1:
        flow_model = DataParallel(flow_model)
    flow_model = flow_model.to(device).eval()

    if arg.compile:
        flow_model = torch.compile(flow_model, mode="reduce-overhead", fullgraph=True)

    # Save configs
    with open(os.path.join(arg.dir, 'config_sampling.json'), 'w') as f:
        json.dump(vars(arg), f, indent=4)

    if arg.action == 'sample':
        sample(arg, flow_model, device, metrics_calculator, all_results, all_original_images, all_reconstructed_images)
    else:
        raise NotImplementedError(f"Action {arg.action} not implemented")


@torch.no_grad()
def sample(arg, model, device, metrics_calculator, all_results, all_original_images, all_reconstructed_images):
    # 原有代码结构保持不变，但内部调用修改后的sample_ode_generative
    output_dir = os.path.join(arg.dir, "samples")
    os.makedirs(output_dir, exist_ok=True)

    # Setup inverse problem
    sampling_config = SamplingConfig()
    sampling_config.inverse_problem = arg.inverse_problem
    sampling_config.noise_sigma = arg.noise_sigma
    sampling_config.kernel_size = arg.kernel_size
    sampling_config.blur_sigma = arg.blur_sigma
    sampling_config.mask_size = arg.mask_size
    sampling_config.scale_factor = arg.scale_factor

    operator = get_inverse_operator(sampling_config, device=device)

    input_images = sorted(glob.glob(os.path.join(arg.input_dir, "*.[pj][np][g]")))
    if not input_images:
        raise ValueError(f"No images found in {arg.input_dir}")
    print(f"Found {len(input_images)} input images")

    straightness_list = []
    nfes = []
    total_times = []
    max_memories = []

    # 从预设名中提取数据集名和方法名
    dataset_name = arg.preset.split('_')[1] if '_' in arg.preset else 'unknown'
    task_name = arg.inverse_problem
    method_name = 'rfoc'

    for i, img_path in enumerate(tqdm(input_images)):
        # 记录开始时间和内存
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()

        z = torch.randn(arg.batchsize, arg.input_nc, arg.res, arg.res).to(device)

        if arg.label_dim > 0:
            label_onehot = torch.eye(arg.label_dim, device=device)[
                torch.randint(0, arg.label_dim, (z.shape[0],), device=device)]
        else:
            label_onehot = None

        if arg.solver in ['euler', 'heun']:
            t_steps = [float(t) for t in arg.t_steps.split(",")] if arg.t_steps else None
            if t_steps:
                t_steps[0] = 1 - 1e-5
            z = z * (1 - 1e-5)

            # 调用修改后的函数，添加CGC-iLQR参数
            traj_uncond, traj_uncond_x0, max_memory, total_time, original_img = sample_ode_generative(
                model, z1=z, N=arg.N, use_tqdm=False, solver=arg.solver,
                label=label_onehot, time_schedule=t_steps, sampler=arg.sampler,
                operator=operator, ref_img_path=img_path, output_dir=output_dir,
                gradient_scale=arg.gradient_scale,
                likebaseline=arg.likebaseline,
                show_ilqr=arg.show_ilqr,
                ilqr_iterations=arg.ilqr_iterations
            )
            x0 = traj_uncond[-1]
            uncond_straightness = straightness(traj_uncond, mean=False)
            straightness_list.append(uncond_straightness)
            total_times.append(total_time)
            max_memories.append(max_memory)
        else:
            raise NotImplementedError(f"Solver {arg.solver} not implemented")

        # 记录结束时间和内存使用
        end_time = time.time()
        processing_time = end_time - start_time
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() - initial_memory
            vram_gb = peak_memory / (1024 ** 3)
        else:
            vram_gb = 0

        # 计算图像质量指标
        # 将图像从[-1,1]转换到[0,1]范围
        original_normalized = (original_img + 1) / 2
        reconstructed_normalized = (x0 + 1) / 2

        image_metrics = metrics_calculator.calculate_single_image_metrics(
            original_normalized, reconstructed_normalized
        )

        # 存储图像用于FID计算
        all_original_images.append(original_normalized.cpu())
        all_reconstructed_images.append(reconstructed_normalized.cpu())

        # 收集结果
        image_name = os.path.basename(img_path)
        result = {
            'image_name': image_name,
            'psnr': image_metrics['psnr'],
            'ssim': image_metrics['ssim'],
            'lpips': image_metrics['lpips'],
            'time': processing_time,
            'vram': vram_gb,
            'nfes': arg.N
        }
        all_results.append(result)

        print(f"Processed {image_name}: PSNR={image_metrics['psnr']:.2f}, SSIM={image_metrics['ssim']:.4f}, "
              f"LPIPS={image_metrics['lpips']:.4f}, Time={processing_time:.2f}s")

        if arg.save_traj:
            save_traj(traj_uncond, os.path.join(arg.dir, "trajs", f"{i:05d}_traj.png"))
            save_traj(traj_uncond_x0, os.path.join(arg.dir, "trajs", f"{i:05d}_traj_x0.png"))

        for idx in range(len(x0)):
            input_filename = os.path.basename(img_path)
            path_img = os.path.join(arg.dir, "samples", input_filename)
            path_z = os.path.join(arg.dir, "zs", input_filename.replace('.png', '.npy'))
            save_image(x0[idx] * 0.5 + 0.5, path_img)

            if arg.save_z:
                np.save(path_z, z[idx].cpu().numpy())

    # 计算FID（如果有多张图像且未跳过）
    if len(all_original_images) > 1 and not arg.skip_fid:
        print("计算FID...")
        original_tensor = torch.cat(all_original_images)
        reconstructed_tensor = torch.cat(all_reconstructed_images)

        fid_score = metrics_calculator.calculate_fid(original_tensor, reconstructed_tensor)

        # 将FID添加到所有结果中
        for result in all_results:
            result['fid'] = fid_score
    else:
        print("跳过FID计算")
        for result in all_results:
            result['fid'] = float('nan')

    # 保存指标结果到Excel
    excel_filename = create_excel_filename(task_name, method_name, dataset_name, arg.output_dir)
    metrics_calculator.save_results_to_excel(all_results, excel_filename)

    # 原有的其他指标保存
    straightness_list = torch.stack(straightness_list).view(-1).cpu().numpy()
    straightness_mean = np.mean(straightness_list).item()
    straightness_std = np.std(straightness_list).item()
    print(f"straightness.shape: {straightness_list.shape}")
    print(f"straightness_mean: {straightness_mean}, straightness_std: {straightness_std}")
    nfes_mean = np.mean(nfes) if len(nfes) > 0 else arg.N
    print(f"nfes_mean: {nfes_mean}")
    avg_time = np.mean(total_times)
    avg_memory = np.mean(max_memories)
    print(f"average_time: {avg_time:.2f}s")
    print(f"average_max_memory: {avg_memory:.2f}GB")

    result_dict = {
        "straightness_mean": straightness_mean,
        "straightness_std": straightness_std,
        "nfes_mean": nfes_mean,
        "average_time": avg_time,
        "average_max_memory": avg_memory,
        "method": "CGC"
    }
    with open(os.path.join(arg.dir, 'result_sampling.json'), 'w') as f:
        json.dump(result_dict, f, indent=4)

    # Plot straightness distribution
    plt.figure()
    plt.hist(straightness_list, bins=20)
    plt.savefig(os.path.join(arg.dir, "straightness.png"))
    plt.close()

    print(f"图像质量指标已保存到: {excel_filename}")
    print(f"Excel文件保存路径: {excel_filename}")
    print(f"绝对路径: {os.path.abspath(excel_filename)}")


if __name__ == "__main__":
    arg = get_args()
    main(arg)