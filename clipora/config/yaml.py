import argparse

import yaml


def parse_yaml_to_args(yaml_file):
    # Initialize the argument parser
    args = argparse.Namespace()

    # Define the default values
    defaults = {
        "model_name": "",  # The name of the pretrained OpenCLIP model to use
        "pretrained": "",  # The path to the pretrained OpenCLIP model to use
        "output_dir": "./clipora_output",  # The directory to save the model to
        "lora_rank": 16,  # The rank of the LoRA matrices
        "lora_alpha": 32,  # The alpha value for the LoRA matrices
        "lora_dropout": 0.0,  # The dropout rate for the LoRA matrices
        "precision": "fp32",
        "gradient_accumulation_steps": 1,  # The number of gradient accumulation steps
        "gradient_checkpointing": False,  # Whether to use gradient checkpointing
        "use_8bit_adam": False,  # Whether to use 8-bit Adam
        "learning_rate": 5e-6,  # The initial learning rate
        "adam_beta1": 0.9,  # The beta1 value for Adam
        "adam_beta2": 0.999,  # The beta2 value for Adam
        "adam_epsilon": 1e-8,  # The epsilon value for Adam
        "epochs": 5,  # The number of epochs to train for
        "warmup": 500,  # The number of warmup steps
        "save_steps": 1000,  # The number of steps between each save
        "datatype": "csv",  # The type of data to use
        "instance_data_dir": "./data.csv",  # The location of the training data
        "image_col": "image",  # The name of the column containing the image paths
        "text_col": "text",  # The name of the column containing the text
        "csv_separator": ",",  # The separator used in the CSV file
        "batch_size": 32,  # The batch size for training per GPU
        "shuffle": True,  # Whether to shuffle the data
        "workers": 1,  # Number of dataloader workers per GPU.
        "wandb": False,  # Whether to use wandb logging
        "wandb_project": "",  # The name of the wandb project
    }

    # Load the YAML file
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    # Assign each key-value pair in the YAML file to an attribute in args
    # If a key isn't in the YAML file, its default value is used
    for key, value in defaults.items():
        setattr(args, key, config.get(key, value))

    return args
