
# 方法说明
METHOD_DESCRIPTIONS = {
    "rfoc": "CGC - Control-Generation Co-design 逆问题求解方法"
}

# 数据集配置
DATASET_CONFIGS = {
    "AFHQ512": {
        "path": "./data/AFHQ512",
        "img_size": 512,
        "description": "动物面部高质量数据集"
    },
    "Celeba512": {
        "path": "./data/Celeba512",
        "img_size": 512,
        "description": "名人面部数据集"
    },
    "DIV2K768": {
        "path": "./data/DIV2K768",
        "img_size": 768,
        "description": "超分辨率训练数据集"
    },
    "DIV2K1024": {
        "path": "./data/DIV2K1024",
        "img_size": 1024,
        "description": "超分辨率训练数据集"
    },
    "FFHQ128": {
        "path": "./data/FFHQ128",
        "img_size": 128,
        "description": "人脸高质量数据集128px"
    },
    "FFHQ512": {
        "path": "./data/FFHQ512",
        "img_size": 512,
        "description": "人脸高质量数据集512px"
    },
    "FFHQ1024": {
        "path": "./data/FFHQ1024",
        "img_size": 1024,
        "description": "人脸高质量数据集1024px"
    },
    "ImageNet256": {
        "path": "./data/ImageNet256",
        "img_size": 256,
        "description": "ImageNet数据集256px"
    },
    "WHU256": {
        "path": "./data/WHU256",
        "img_size": 256,
        "description": "WHU数据集256"
    }
}

# 任务配置
TASK_CONFIGS = {
    # 重叠任务
    "sr_avgpool_4x": {
        "task": "sr_avgpool",
        "deg_scale": 4,
        "NFE": 25,
        "step_size": 15.0,
        "description": "4倍平均池化超分辨率"
    },
    "box_inpaint_small": {
        "task": "inpainting",
        "deg_scale": 20,
        "NFE": 35,
        "step_size": 10.0,
        "description": "小区域盒状修复(20x20)"
    },
    "deblur_gauss_mild": {
        "task": "deblur_gauss",
        "deg_scale": 15,
        "NFE": 25,
        "step_size": 10.0,
        "description": "轻度高斯去模糊"
    },

    # 扩展重叠任务
    "sr_avgpool_2x": {
        "task": "sr_avgpool",
        "deg_scale": 2,
        "NFE": 20,
        "step_size": 12.0,
        "description": "2倍平均池化超分辨率"
    },
    "sr_avgpool_8x": {
        "task": "sr_avgpool",
        "deg_scale": 8,
        "NFE": 30,
        "step_size": 20.0,
        "description": "8倍平均池化超分辨率"
    },
    "box_inpaint_medium": {
        "task": "inpainting",
        "deg_scale": 32,
        "NFE": 40,
        "step_size": 12.0,
        "description": "中区域盒状修复(32x32)"
    },
    "box_inpaint_large": {
        "task": "inpainting",
        "deg_scale": 64,
        "NFE": 45,
        "step_size": 15.0,
        "description": "大区域盒状修复(64x64)"
    },
    "deblur_gauss_severe": {
        "task": "deblur_gauss",
        "deg_scale": 25,
        "NFE": 35,
        "step_size": 15.0,
        "description": "重度高斯去模糊"
    },

    "sr_bicubic_4x": {
        "task": "sr_bicubic",
        "deg_scale": 4,
        "NFE": 28,
        "step_size": 18.0,
        "description": "4倍双三次超分辨率"
    },
    "deblur_motion": {
        "task": "deblur_motion",
        "deg_scale": 21,
        "NFE": 30,
        "step_size": 25.0,
        "description": "运动模糊去除"
    },
    "inpainting_half": {
        "task": "inpainting",
        "deg_scale": 256,
        "NFE": 50,
        "step_size": 8,
        "description": "半区域图像修复"
    },
    "colorization": {
        "task": "colorization",
        "deg_scale": 1,
        "NFE": 25,
        "step_size": 12.0,
        "description": "灰度图像上色"
    },
    "cs_walsh_25p": {
        "task": "cs_walshhadamard",
        "deg_scale": 4,
        "NFE": 35,
        "step_size": 30.0,
        "description": "25%采样率压缩感知"
    },
    "denoising_mild": {
        "task": "denoising",
        "deg_scale": 1,
        "NFE": 15,
        "step_size": 5.0,
        "description": "轻度去噪"
    },
    "denoising_severe": {
        "task": "denoising",
        "deg_scale": 1,
        "NFE": 25,
        "step_size": 12.0,
        "description": "重度去噪"
    },
    "jpeg_qf20": {
        "task": "jpeg",
        "deg_scale": 20,
        "NFE": 20,
        "step_size": 8.0,
        "description": "质量因子20的JPEG复原"
    }
}

# 方法配置 - 只保留 rfoc
METHOD_CONFIGS = {
    "rfoc": {
        "method": "rfoc",
        "cfg_scale": 1.0,
        "description": "CGC 方法"
    }
}

# ========== 自动生成所有预设组合 ==========
PRESET_CONFIGS = {}

# 生成所有数据集-方法-任务的组合
for dataset_name, dataset_config in DATASET_CONFIGS.items():
    for method_name, method_config in METHOD_CONFIGS.items():
        for task_name, task_config in TASK_CONFIGS.items():
            # 创建预设名称
            preset_name = f"{dataset_name}_{method_name}_{task_name}"

            # 合并配置
            PRESET_CONFIGS[preset_name] = {
                "dataset": dataset_name,
                "method": method_config["method"],
                "task": task_config["task"],
                "img_path": dataset_config["path"],
                "img_size": dataset_config["img_size"],
                "deg_scale": task_config["deg_scale"],
                "NFE": task_config["NFE"],
                "step_size": task_config["step_size"],
                "cfg_scale": method_config["cfg_scale"],
                "description": f"{dataset_config['description']} - {method_config['description']} - {task_config['description']}"
            }

# ========== 任务分类 ==========
TASK_CATEGORIES = {
    "重叠-超分辨率": ["sr_avgpool_4x", "sr_avgpool_2x", "sr_avgpool_8x"],
    "重叠-盒状修复": ["box_inpaint_small", "box_inpaint_medium", "box_inpaint_large"],
    "重叠-去模糊": ["deblur_gauss_mild", "deblur_gauss_severe"],
    "新增-超分辨率": ["sr_bicubic_4x"],
    "新增-去模糊": ["deblur_motion"],
    "新增-图像修复": ["inpainting_half", "colorization"],
    "新增-压缩感知": ["cs_walsh_25p"],
    "新增-去噪": ["denoising_mild", "denoising_severe"],
    "新增-JPEG压缩": ["jpeg_qf20"]
}

# ========== 常用预设快捷方式 ==========
QUICK_PRESETS = {
    "quick_overlap_sr": "FFHQ512_rfoc_sr_avgpool_4x",
    "quick_overlap_inpaint": "AFHQ512_rfoc_box_inpaint_small",
    "quick_overlap_deblur": "Celeba512_rfoc_deblur_gauss_mild",
    "quick_new_sr": "FFHQ512_rfoc_sr_bicubic_4x",
    "quick_new_motion": "ImageNet256_rfoc_deblur_motion",
    "quick_new_color": "FFHQ512_rfoc_colorization",
    "quick_new_denoise": "FFHQ512_rfoc_denoising_severe"
}


def get_preset_by_components(dataset, method, task):
    """通过组件获取预设名称"""
    preset_name = f"{dataset}_{method}_{task}"
    if preset_name in PRESET_CONFIGS:
        return preset_name
    else:
        available = [k for k in PRESET_CONFIGS.keys()
                     if dataset in k and method in k and task in k]
        if available:
            return available[0]
        else:
            raise ValueError(f"找不到匹配的预设: {dataset}_{method}_{task}")


def list_presets_by_category(category=None):
    """按类别列出预设"""
    if category and category in TASK_CATEGORIES:
        tasks = TASK_CATEGORIES[category]
        presets = [k for k in PRESET_CONFIGS.keys()
                   if any(task in k for task in tasks)]
    else:
        presets = list(PRESET_CONFIGS.keys())

    print(f"\n{'预设名称':<50} {'数据集':<12} {'方法':<10} {'任务类型':<8} {'描述'}")
    print("-" * 120)

    for preset in sorted(presets):
        config = PRESET_CONFIGS[preset]

        desc_short = config['description']
        if len(desc_short) > 40:
            desc_short = desc_short[:40] + "..."

        print(
            f"{preset:<48} {config['dataset']:<12} {config['method']:<10} {preset.split('_')[-1]:<8} {desc_short}")


def get_overlap_presets():
    """获取所有与原文件重叠的预设"""
    overlap_tasks = ["sr_avgpool_4x", "box_inpaint_small", "deblur_gauss_mild"]
    overlap_presets = {}

    for preset_name, config in PRESET_CONFIGS.items():
        if any(task in preset_name for task in overlap_tasks):
            overlap_presets[preset_name] = config

    return overlap_presets


if __name__ == "__main__":
    print(f"总共生成 {len(PRESET_CONFIGS)} 个预设配置")

    # 统计重叠任务
    overlap_count = len(get_overlap_presets())
    print(f"与原文件重叠的预设: {overlap_count} 个")
    print(f"新增任务预设: {len(PRESET_CONFIGS) - overlap_count} 个")

    print("\n预设分类统计:")
    for category, tasks in TASK_CATEGORIES.items():
        print(f"  {category}: {len(tasks)} 个任务")

    print("\n快捷预设:")
    for quick_name, full_name in QUICK_PRESETS.items():
        print(f"  {quick_name}: {full_name}")