from dataclasses import dataclass

import yaml


@dataclass
class TrainConfig:
    model_name: str = "ViT-H-14"
    pretrained: str = ""
    compile: bool = False
    seed: int = 42

    device: str = "cuda"
    lora_text: bool = True
    lora_vision: bool = True
    vision_heads: int = 16  # This is true for ViT-L/14

    output_dir: str = "./clipora_output"

    wandb: bool = False
    wandb_project: str = ""

    train_dataset: str = "./data/train.csv"
    eval_dataset: str = "./data/val.csv"
    datatype: str = "csv"
    image_col: str = "image"
    text_col: str = "text"
    csv_separator: str = ","
    shuffle: bool = True
    workers: int = 0

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    precision: str = "fp32"
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False

    use_8bit_adam: bool = False

    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    epochs: int = 5
    warmup: int = 500
    save_interval: int = 1000
    eval_interval: int = 100
    eval_steps: int = 100


def parse_yaml_to_config(yaml_path: str) -> TrainConfig:
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return TrainConfig(**config_dict)
