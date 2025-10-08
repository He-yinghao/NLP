import os
import yaml
from typing import Dict, Any, List


class ConfigManager:
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir

    def load_config(self, config_paths: List[str]) -> Dict[str, Any]:
        """加载并合并多个配置文件"""
        config = {}

        for path in config_paths:
            full_path = os.path.join(self.config_dir, path)
            with open(full_path, "r") as f:
                partial_config = yaml.safe_load(f)
                config = self._deep_merge(config, partial_config)

        return config

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """深度合并字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def get_model_config(self, model_name: str, experiment: str = None) -> Dict:
        """获取特定模型的配置"""
        config_paths = [
            "base/data.yaml",
            "base/training.yaml",
            f"models/{model_name}.yaml",
        ]

        if experiment:
            config_paths.append(f"experiments/{experiment}.yaml")

        return self.load_config(config_paths)


# 使用示例
config_manager = ConfigManager()

# 获取BERT基础配置
bert_config = config_manager.get_model_config(ModelName.BERT.value)

# 获取特定实验配置
exp_config = config_manager.get_model_config(ModelName.BERT.value, "exp_001_bert_base")
