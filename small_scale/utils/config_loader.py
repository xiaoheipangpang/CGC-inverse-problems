# utils/config_loader.py
"""
配置文件加载器
"""

import sys
import os
from typing import Dict, Any

# 添加配置目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from configs.presets import get_preset, list_available_options

    HAS_CONFIGS = True
except ImportError:
    HAS_CONFIGS = False
    print("警告: 未找到统一配置文件，使用原有预设")


class ConfigLoader:
    """配置加载器"""

    @staticmethod
    def load_preset(preset_name: str) -> Dict[str, Any]:
        """加载预设配置"""
        if not HAS_CONFIGS:
            raise ImportError("统一配置文件未找到，请确保 configs/presets.py 存在")

        config = get_preset(preset_name)
        if not config:
            raise ValueError(f"未找到预设: {preset_name}")

        return config

    @staticmethod
    def merge_args_with_preset(args, preset_name: str) -> Dict[str, Any]:
        """合并命令行参数和预设配置"""
        if not HAS_CONFIGS:
            return vars(args) if hasattr(args, '__dict__') else args

        # 获取预设配置
        preset_config = ConfigLoader.load_preset(preset_name)

        # 合并配置（命令行参数优先级更高）
        merged_config = preset_config.copy()

        if hasattr(args, '__dict__'):
            for key, value in vars(args).items():
                if value is not None:
                    merged_config[key] = value

        return merged_config