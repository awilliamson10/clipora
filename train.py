import argparse
import itertools
import logging
import os

import numpy as np
import open_clip
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPModel

from clipora.config import TrainConfig, parse_yaml_to_config
from clipora.data import get_dataloader
from clipora.lora.inject import inject_linear_attention
from clipora.scheduler.cosine import cosine_lr

logger = logging.getLogger(__name__)


def compute_clip_loss(model, input_ids, pixel_values):
    input_ids = input_ids.to(model.device)
    pixel_values = pixel_values.to(model.device)
    return model(input_ids=input_ids, pixel_values=pixel_values, return_loss=True).loss


@torch.no_grad()
def evaluate(model, dataloader, config):
    out = {}
    model.eval()
    losses = torch.zeros(config.eval_steps)
    for k in range(config.eval_steps):
        pixels, input_ids = next(iter(dataloader))
        loss = compute_clip_loss(model, input_ids, pixels)
        losses[k] = loss.item()
    out["eval_loss"] = losses.mean()
    model.train()
    return out


def init_model(config: TrainConfig):
    model = CLIPModel.from_pretrained(config.model)
    image_processor = CLIPImageProcessor.from_pretrained(config.model)
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["k_proj", "v_proj", "q_proj", "out_proj"],
    )
    model = get_peft_model(model, lora_config)
    if config.compile:
        model.compile()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"No. of parameters: {total_params:,}")
    print(f"No. of trainable parameters: {trainable_params:,}")
    return model, image_processor


def main(config: TrainConfig):
    logging.basicConfig(level=logging.INFO)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb" if config.wandb else None,
    )

    if accelerator.is_main_process:
        accelerator.print()
        if config.output_dir is not None:
            accelerator.print(f"Output directory: {config.output_dir}")
            os.makedirs(config.output_dir, exist_ok=True)

        if config.wandb:
            accelerator.init_trackers(
                project_name=config.wandb_project if config.wandb_project else None,
            )

    if config.seed is not None:
        seed = config.seed
        accelerator.print(f"Using seed {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model, image_preprocess = init_model(config)

    train_dataloader = get_dataloader(config, image_preprocess, "train")
    eval_dataloader = get_dataloader(config, image_preprocess, "val")
    assert len(train_dataloader), "No data found, please check your data location."

    if config.gradient_checkpointing:
        model.set_grad_checkpointing(True)

    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    if isinstance(config.learning_rate, str):
        config.learning_rate = float(config.learning_rate)

    params_to_optimize = [
        {
            "params": itertools.chain(model.parameters()),
            "lr": config.learning_rate,
        },
    ]

    optimizer = optimizer_class(
        params_to_optimize,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )

    # create scheduler if train
    total_steps = train_dataloader.num_batches * config.epochs
    # if args.warmup is float, it is a percentage of total_steps
    if isinstance(config.warmup, float):
        assert (
            0 <= config.warmup <= 1
        ), "Warmup must be between 0 and 1 if not a fixed number of steps."
        config.warmup = int(config.warmup * total_steps)

    scheduler = cosine_lr(optimizer, config.learning_rate, config.warmup, total_steps)

    model, optimizer, scheduler, train_dataloader, eval_dataloader = (
        accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, eval_dataloader
        )
    )

    print("***** Running training *****")
    print(f"  Num Iters = {len(train_dataloader)}")
    print(f"  Num Epochs = {config.epochs}")
    print(f"  Instantaneous batch size per device = {config.batch_size}")
    print(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(config.epochs * len(train_dataloader)),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if accelerator.is_local_main_process:
                if global_step % config.eval_interval == 0:
                    if accelerator.is_local_main_process:
                        eval_loss = evaluate(model, eval_dataloader, config)
                        accelerator.log(eval_loss, step=global_step)
                        progress_bar.write(
                            f"Step: {global_step}, Eval loss: {eval_loss['eval_loss']}"
                        )
                        if eval_loss["eval_loss"] < best_val_loss:
                            best_val_loss = eval_loss["eval_loss"]
                            save_path = os.path.join(config.output_dir, "best_val")
                            model.save_pretrained(save_path)

            pixels, input_ids = batch
            loss = compute_clip_loss(model, input_ids, pixels)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = model.parameters()
                accelerator.clip_grad_norm_(params_to_clip, 1.0)  # args.max_grad_norm)
            optimizer.step()
            scheduler(global_step)
            progress_bar.update(1)
            global_step += 1

            logs = {
                "loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "step": global_step,
                "epoch": epoch,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        merged = model.merge_and_unload()
        save_path = os.path.join(config.output_dir)
        merged.save_pretrained(save_path)

    accelerator.print("\n\nTraining completed.\n\n")
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="The path to the yaml file containing the training configuration.",
    )
    config = parse_yaml_to_config(parser.parse_args().config)
    main(config)
