import argparse
import itertools
import logging
import math
import os

import open_clip
import torch
import tqdm
from accelerate import Accelerator

from clipora.config.yaml import parse_yaml_to_args as parse_args
from clipora.data import get_dataloader
from clipora.lora.extract import extract_lora_ups_down, save_lora_weight
from clipora.lora.inject import inject_trainable_lora
from clipora.scheduler.cosine import cosine_lr
from clipora.utils import unwrap_model

logger = logging.getLogger(__name__)


def main(args):
    logging.basicConfig(level=logging.INFO)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.wandb,
    )

    if accelerator.is_main_process:
        accelerator.print(
            f"Training {args.model_name} with LoRA on {args.instance_data_dir}"
        )
        if args.output_dir is not None:
            accelerator.print(f"Output directory: {args.output_dir}")
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the model and tokenizer
    model, preprocess_train, _ = open_clip.create_model_and_transforms(
        f"hf-hub:{args.model_name}",
    )

    # Get the TextEncoder and ImageEncoder
    text_encoder = model.transformer
    text_encoder.requires_grad_(False)

    image_encoder = model.visual
    image_encoder.requires_grad_(False)

    text_encoder_lora_params, _ = inject_trainable_lora(
        text_encoder,
        target_replace_module=["MultiheadAttention"],
        r=args.lora_rank,
    )

    for _up, _down in extract_lora_ups_down(
        text_encoder, target_replace_module=["MultiheadAttention"]
    ):
        print("Before training: text encoder First Layer lora up", _up.weight.data)
        print("Before training: text encoder First Layer lora down", _down.weight.data)
        break

    image_encoder_lora_params, names = inject_trainable_lora(
        image_encoder.transformer,
        target_replace_module=["MultiheadAttention"],
        r=args.lora_rank,
    )

    for _up, _down in extract_lora_ups_down(
        image_encoder.transformer, target_replace_module=["MultiheadAttention"]
    ):
        print("Before training: image encoder First Layer lora up", _up.weight.data)
        print("Before training: image encoder First Layer lora down", _down.weight.data)
        break

    if args.gradient_checkpointing:
        model.set_gradient_checkpointing(True)

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = [
        {
            "params": itertools.chain(*text_encoder_lora_params),
            "lr": args.learning_rate,
        },
        {
            "params": itertools.chain(*image_encoder_lora_params),
            "lr": args.learning_rate,
        },
    ]

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )

    train_dataloader = get_dataloader(args, preprocess_train)
    assert len(train_dataloader), "No data found, please check your data location."

    # create scheduler if train
    total_steps = train_dataloader.num_batches * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    text_encoder, image_encoder, optimizer, scheduler = accelerator.prepare(
        text_encoder, image_encoder, optimizer, scheduler
    )

    model.to(accelerator.device)

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataloader.dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.epochs}")
    print(f"  Instantaneous batch size per device = {args.batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.epochs * len(train_dataloader)),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    global_step = 0
    last_save = 0

    for epoch in range(args.epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            loss = open_clip.ClipLoss()

            images, texts = batch
            images = images.to(device=accelerator.device, non_blocking=True)
            texts = texts.to(device=accelerator.device, non_blocking=True)

            image_features, text_features, logit_scale = model(images, texts)
            total_loss = loss(image_features, text_features, logit_scale)

            accelerator.backward(total_loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if accelerator.is_local_main_process:
                if args.save_steps and global_step - last_save >= args.save_steps:
                    last_save = global_step
                    lora_save = f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}.text_encoder.pt"
                    logger.info(f"Saving LoRA weights to {lora_save}")
                    with torch.no_grad():
                        save_lora_weight(
                            unwrap_model(model).logit_scale.clamp_(0, math.log(100)),
                            lora_save,
                        )

                    last_save = global_step

            logs = {"loss": loss.detach().item(), "lr": scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        lora_save = (
            f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}.text_encoder.pt"
        )
        save_lora_weight(
            unwrap_model(model).logit_scale.clamp_(0, math.log(100)), lora_save
        )
        logger.info(f"Saving LoRA weights to {lora_save}")

    accelerator.print("\n\nLoRA Training completed.\n\n")
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="The path to the yaml file containing the training configuration.",
    )
    args = parse_args(parser.parse_args().yaml)
    main(args)
