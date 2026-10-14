import argparse
from pathlib import Path
from typing import List
from munch import munchify
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from torchvision import transforms
import time
import os

from util import set_seed, get_img_list, process_text
from sd3_sampler import get_solver
from functions.degradation import get_degradation

# 导入预设配置和指标计算
from preset_config import PRESET_CONFIGS, TASK_CATEGORIES, QUICK_PRESETS, get_preset_by_components, \
    list_presets_by_category
from metrics_calculator import MetricsCalculator, create_excel_filename


@torch.no_grad
def precompute(args, prompts: List[str], solver) -> List[torch.Tensor]:
    prompt_emb_set = []
    pooled_emb_set = []

    num_samples = args.num_samples if args.num_samples > 0 else len(prompts)
    for prompt in prompts[:num_samples]:
        prompt_emb, pooled_emb = solver.encode_prompt(prompt, batch_size=1)
        prompt_emb_set.append(prompt_emb)
        pooled_emb_set.append(pooled_emb)

    return prompt_emb_set, pooled_emb_set


def create_organized_workdir(args, dataset_name, method_name, task_name):
    """创建有组织的输出目录结构"""
    if args.preset:
        workdir_name = f"{dataset_name}_{method_name}_{task_name}"
    else:
        workdir_name = f"custom_{method_name}_{task_name}"

    main_workdir = Path(args.workdir_base) / workdir_name
    main_workdir.mkdir(parents=True, exist_ok=True)

    subdirs = ['input', 'recon', 'label']
    for subdir in subdirs:
        (main_workdir / subdir).mkdir(exist_ok=True)

    return main_workdir


def apply_preset_config(args, preset_name):
    """应用预设配置到args"""
    if preset_name in QUICK_PRESETS:
        actual_preset = QUICK_PRESETS[preset_name]
        print(f"使用快捷预设 '{preset_name}' -> '{actual_preset}'")
        preset_name = actual_preset

    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"预设 '{preset_name}' 不存在。可用的预设: {list(PRESET_CONFIGS.keys())[:5]}...")

    preset = PRESET_CONFIGS[preset_name]

    args.task = preset["task"]
    args.deg_scale = preset["deg_scale"]
    args.method = preset["method"]
    args.NFE = preset["NFE"]
    args.step_size = preset["step_size"]
    args.img_size = preset["img_size"]

    args.img_path = Path(preset["img_path"])

    print(f"应用预设: {preset_name}")
    print(f"  描述: {preset['description']}")
    print(f"  数据集: {preset['dataset']}")
    print(f"  方法: {args.method}")
    print(f"  任务: {args.task}")
    print(f"  降质尺度: {args.deg_scale}")
    print(f"  图像尺寸: {args.img_size}")
    print(f"  采样步数: {args.NFE}")
    print(f"  步长: {args.step_size}")


def list_available_presets():
    print("可用的预设配置:")

    print("\n快捷预设:")
    for quick_name, full_name in QUICK_PRESETS.items():
        print(f"  {quick_name}: {full_name}")

    for category in TASK_CATEGORIES.keys():
        list_presets_by_category(category)


def run(args):
    # 设置随机种子
    set_seed(args.seed)

    # 应用预设配置
    if args.preset:
        apply_preset_config(args, args.preset)
        preset_config = PRESET_CONFIGS[args.preset]
        dataset_name = preset_config['dataset']
        method_name = preset_config['method']
        task_name = preset_config['task']
    elif args.components:
        dataset, method, task = args.components.split(',')
        preset_name = get_preset_by_components(dataset, method, task)
        apply_preset_config(args, preset_name)
        dataset_name = dataset
        method_name = method
        task_name = task
    else:
        dataset_name = "custom"
        method_name = args.method
        task_name = args.task

    # 创建输出目录
    args.workdir = create_organized_workdir(args, dataset_name, method_name, task_name)
    print(f"输出目录: {args.workdir}")

    # 初始化指标计算器
    metrics_calculator = MetricsCalculator(device=torch.device("cuda"))
    all_results = []
    all_original_images = []
    all_reconstructed_images = []

    # 加载求解器（只保留 rfoc / CGC）
    solver_kwargs = {}
    if hasattr(args, 'local_model_path') and args.local_model_path:
        solver_kwargs['local_model_path'] = args.local_model_path

    solver = get_solver(args.method, **solver_kwargs)

    # 加载文本提示
    prompts = process_text(prompt=args.prompt, prompt_file=args.prompt_file)

    # 文本编码器设置
    if hasattr(solver, 'text_enc_1') and solver.text_enc_1 is not None:
        solver.text_enc_1.to('cuda')
    if hasattr(solver, 'text_enc_2') and solver.text_enc_2 is not None:
        solver.text_enc_2.to('cuda')
    if hasattr(solver, 'text_enc_3') and solver.text_enc_3 is not None:
        solver.text_enc_3.to('cuda')

    # 内存优化
    if args.efficient_memory:
        print("使用内存优化模式...")
        with torch.no_grad():
            prompt_emb_set, pooled_emb_set = precompute(args, prompts, solver)
            null_emb, null_pooled_emb = solver.encode_prompt([''], batch_size=1)

        del solver.text_enc_1
        del solver.text_enc_2
        del solver.text_enc_3
        torch.cuda.empty_cache()

        prompt_embs = [[x, y] for x, y in zip(prompt_emb_set, pooled_emb_set)]
        null_embs = [null_emb, null_pooled_emb]
    else:
        prompt_embs = [[None, None]] * len(prompts)
        null_embs = [None, None]

    print("初始化完成。")

    solver.vae.to('cuda')
    solver.transformer.to('cuda')

    # 问题设置
    deg_config = munchify({
        'channels': 3,
        'image_size': args.img_size,
        'deg_scale': args.deg_scale
    })

    if args.task == 'inpainting':
        deg_config.update({
            'imgH': args.img_size,
            'imgW': args.img_size
        })

    operator = get_degradation(args.task, deg_config, solver.transformer.device)

    # 图像预处理
    tf = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor()
    ])

    # 获取图像列表
    img_files = list(get_img_list(args.img_path))
    if not img_files:
        raise ValueError(f"在 {args.img_path} 中未找到图像文件")

    print(f"找到 {len(img_files)} 张图像")

    pbar = tqdm(img_files, desc="求解中")
    for i, path in enumerate(pbar):
        start_time = time.time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()

        try:
            img = tf(Image.open(path).convert('RGB'))
            img = img.unsqueeze(0).to(solver.vae.device)
            img = img * 2 - 1

            y = operator.A(img)
            y = y + 0.03 * torch.randn_like(y)

            # 使用 CGC 求解器重建
            out = solver.sample(measurement=y,
                                operator=operator,
                                NFE=args.NFE,
                                img_shape=(args.img_size, args.img_size),
                                step_size=args.step_size,
                                task=args.task)
            out = out.float()

            end_time = time.time()
            processing_time = end_time - start_time
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() - initial_memory
                vram_gb = peak_memory / (1024 ** 3)
            else:
                vram_gb = 0

            # 计算图像质量指标
            original_normalized = ((img + 1) / 2).float()
            reconstructed_normalized = ((out + 1) / 2).float()

            image_metrics = metrics_calculator.calculate_single_image_metrics(
                original_normalized, reconstructed_normalized
            )

            all_original_images.append(original_normalized.cpu())
            all_reconstructed_images.append(reconstructed_normalized.cpu())

            image_name = Path(path).stem
            result = {
                'image_name': image_name,
                'psnr': image_metrics['psnr'],
                'ssim': image_metrics['ssim'],
                'lpips': image_metrics['lpips'],
                'time': processing_time,
                'vram': vram_gb,
                'nfes': args.NFE
            }
            all_results.append(result)

            pbar.set_description(f"求解中 PSNR:{image_metrics['psnr']:.2f} SSIM:{image_metrics['ssim']:.4f}")

            print(f"处理 {image_name}: "
                  f"PSNR={image_metrics['psnr']:.2f}dB, "
                  f"SSIM={image_metrics['ssim']:.4f}, "
                  f"LPIPS={image_metrics['lpips']:.4f}, "
                  f"时间={processing_time:.2f}s, "
                  f"显存={vram_gb:.2f}GB")

            # 保存结果
            save_image(operator.At(y).reshape(img.shape).float(),
                       args.workdir / 'input' / f"{image_name}_input.png",
                       normalize=True)
            save_image(out,
                       args.workdir / 'recon' / f"{image_name}_recon.png",
                       normalize=True)
            save_image(img,
                       args.workdir / 'label' / f"{image_name}_label.png",
                       normalize=True)

        except Exception as e:
            print(f"处理图像 {path} 时出错: {e}")
            continue

        if (i + 1) == args.num_samples:
            break

    # 计算FID
    if len(all_original_images) > 1 and not args.skip_fid:
        print("计算FID...")
        try:
            original_tensor = torch.cat(all_original_images)
            reconstructed_tensor = torch.cat(all_reconstructed_images)
            fid_score = metrics_calculator.calculate_fid(original_tensor, reconstructed_tensor)

            for result in all_results:
                result['fid'] = fid_score

            print(f"FID Score: {fid_score:.2f}")
        except Exception as e:
            print(f"FID计算失败: {e}")
            for result in all_results:
                result['fid'] = float('nan')
    else:
        print("跳过FID计算（图像数量不足或设置了--skip-fid）")
        for result in all_results:
            result['fid'] = float('nan')

    os.makedirs(args.output_dir, exist_ok=True)

    # 保存指标到 Excel
    excel_filename = create_excel_filename(task_name, method_name, dataset_name, args.output_dir)
    metrics_calculator.save_results_to_excel(all_results, excel_filename)

    # 打印总结
    print(f"\n{'=' * 60}")
    print(f"实验完成总结")
    print(f"{'=' * 60}")
    print(f"数据集: {dataset_name}")
    print(f"方法: {method_name}")
    print(f"任务: {task_name}")
    print(f"处理图像: {len(all_results)} 张")
    print(f"图像输出: {args.workdir}")
    print(f"指标文件: {excel_filename}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # ==================================================================
    # 硬编码的预设列表（只保留 rfoc / CGC 相关预设）
    # ==================================================================
    PRESET_LIST = [
        # 超分辨率
        "DIV2K1024_rfoc_sr_avgpool_8x",
        "FFHQ1024_rfoc_sr_avgpool_8x",
        "AFHQ512_rfoc_sr_avgpool_4x",
        "Celeba512_rfoc_sr_avgpool_4x",

        # 运动去模糊
        "DIV2K1024_rfoc_deblur_motion",
        "AFHQ512_rfoc_deblur_motion",
        "Celeba512_rfoc_deblur_motion",

        # 高斯去模糊
        "AFHQ512_rfoc_deblur_gauss_mild",
        "Celeba512_rfoc_deblur_gauss_mild",

        # 图像修复
        "DIV2K1024_rfoc_inpainting_half",
        "AFHQ512_rfoc_inpainting_half",
        "Celeba512_rfoc_inpainting_half",

        # 添加更多你需要测试的 rfoc 预设
    ]

    parser = argparse.ArgumentParser(description='SD3-CGC 图像复原系统')

    # 单个预设参数
    parser.add_argument('--preset', type=str, default='AFHQ512_rfoc_sr_avgpool_4x',
                        help='使用预设配置')

    # 批量运行标志
    parser.add_argument('--run-all', action='store_true', default=True,
                        help='运行所有预设列表中的配置')

    parser.add_argument('--components', type=str, default=None,
                        help='通过组件指定: 数据集,方法,任务，如: FFHQ512,rfoc,sr_avgpool_4x')
    parser.add_argument('--list-presets', action='store_true',
                        help='列出所有可用的预设配置')
    parser.add_argument('--list-category', type=str, default=None,
                        help='按类别列出预设')

    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--prompt_file', type=str, default=None)
    parser.add_argument('--local-model-path', type=str, default=None,
                        help='本地模型路径，如不指定则使用默认路径')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--img_path', type=Path, default=None,
                        help='输入图像路径（如不指定则使用预设的数据集）')
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--workdir_base', type=Path, default='workdir',
                        help='工作目录基础路径')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='指标结果输出目录')
    parser.add_argument('--efficient_memory', default=False, action='store_true')
    parser.add_argument('--skip_fid', action='store_true', default=False,
                        help='跳过FID计算')

    parser.add_argument('--NFE', type=int, default=None)
    parser.add_argument('--cfg_scale', type=float, default=1.0)
    parser.add_argument('--img_size', type=int, default=None)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--deg_scale', type=int, default=None)
    parser.add_argument('--step_size', type=float, default=None)

    args = parser.parse_args()

    # 列出预设
    if args.list_presets:
        list_available_presets()
        exit(0)

    if args.list_category:
        list_presets_by_category(args.list_category)
        exit(0)

    # 参数验证
    if not args.preset and not args.components and not args.run_all:
        print("警告: 未指定预设，将使用默认参数运行")
        print("建议使用以下方式之一:")
        print("  --preset 预设名           # 运行单个预设")
        print("  --run-all                # 运行所有预设列表")
        print("  --components 数据集,方法,任务")
        if not args.img_path:
            raise ValueError("未指定预设时必须提供 --img_path 参数")

    # 批量运行
    if args.run_all:
        total_presets = len(PRESET_LIST)
        print(f"批量运行 {total_presets} 个预设:")
        for i, preset_name in enumerate(PRESET_LIST, 1):
            print(f"  {i:2d}. {preset_name}")

        print("\n" + "=" * 60)

        completed = 0
        failed = 0

        pbar = tqdm(PRESET_LIST, desc="批量运行进度", unit="预设")

        for preset_name in pbar:
            pbar.set_description(f"正在运行: {preset_name}")

            try:
                current_num = completed + failed + 1
                pbar.set_postfix({
                    '进度': f'{current_num}/{total_presets}',
                    '成功': completed,
                    '失败': failed
                })

                args.preset = preset_name
                run(args)

                completed += 1
                pbar.set_postfix({
                    '进度': f'{current_num}/{total_presets}',
                    '成功': completed,
                    '失败': failed
                })

                print(f"预设 {preset_name} 完成")

                if current_num < total_presets:
                    time.sleep(2)

            except Exception as e:
                print(f"预设 {preset_name} 失败: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
                pbar.set_postfix({
                    '进度': f'{current_num}/{total_presets}',
                    '成功': completed,
                    '失败': failed
                })
                continue

        pbar.set_description("批量运行完成")
        pbar.close()

        print(f"\n{'=' * 60}")
        print(f"批量运行完成摘要")
        print(f"{'=' * 60}")
        print(f"预设总数: {total_presets}")
        print(f"成功完成: {completed}")
        print(f"失败: {failed}")
        print(f"成功率: {completed / total_presets * 100:.1f}%")
        print(f"{'=' * 60}")