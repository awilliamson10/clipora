import argparse
import itertools
import logging
import os

import open_clip
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

from clipora.config.yaml import parse_yaml_to_args as parse_args
from clipora.data import get_dataloader
from clipora.lora.inject import inject_linear_attention
from clipora.scheduler.cosine import cosine_lr

logger = logging.getLogger(__name__)


def main(args):
    logging.basicConfig(level=logging.INFO)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if args.wandb else None,
    )

    if accelerator.is_main_process:
        accelerator.print(
            f"Training {args.model_name} with LoRA on {args.instance_data_dir}"
        )
        if args.output_dir is not None:
            accelerator.print(f"Output directory: {args.output_dir}")
            os.makedirs(args.output_dir, exist_ok=True)

        if args.wandb:
            accelerator.init_trackers(
                project_name=args.wandb_project,
                config={
                    "learning_rate": args.learning_rate,
                    "dataset": args.instance_data_dir,
                    "epochs": args.epochs,
                },
            )

    # Load the model and tokenizer
    model, preprocess_train, _ = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=accelerator.device,
    )
    model_config = open_clip.get_model_config(args.model_name)

    # Inject linear attention to TextTransformer
    model = inject_linear_attention(
        model=model,
        encoder={"transformer"},
        embed_dim=model_config["embed_dim"],
        num_heads=model_config["text_cfg"]["heads"],
    )
    model = inject_linear_attention(
        model=model,
        encoders={"visual.transformer"},
        embed_dim=model_config["vision_cfg"]["width"],
        num_heads=16,  # There may be a better way to find this, but this is true for ViT-L/14
    )
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["qkv", "proj"],
    )
    model = get_peft_model(model, config)
    # model.token_embedding.requires_grad_(True)
    accelerator.print("LoRA injected.")
    model.print_trainable_parameters()

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
            "params": itertools.chain(model.parameters()),
            "lr": args.learning_rate,
        },
    ]

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )

    train_dataloader = get_dataloader(args, preprocess_train, "train")
    val_dataloader = get_dataloader(args, preprocess_train, "val")
    assert len(train_dataloader), "No data found, please check your data location."

    # create scheduler if train
    total_steps = train_dataloader.num_batches * args.epochs
    # if args.warmup is float, it is a percentage of total_steps
    if isinstance(args.warmup, float):
        assert (
            0 <= args.warmup <= 1
        ), "Warmup must be between 0 and 1 if not a fixed number of steps."
        args.warmup = int(args.warmup * total_steps)

    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    model.to(device=accelerator.device)

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

            if global_step > 0 and global_step % args.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    for j, val_batch in enumerate(val_dataloader):
                        images, texts = val_batch
                        images = images.to(device=accelerator.device, non_blocking=True)
                        texts = texts.to(device=accelerator.device, non_blocking=True)

                        image_features, text_features, logit_scale = model(
                            images, texts
                        )
                        loss = open_clip.ClipLoss()
                        total_loss = loss(image_features, text_features, logit_scale)

                        logs = {
                            "val_loss": total_loss.item(),
                            "lr": optimizer.param_groups[0]["lr"],
                            "epoch": epoch,
                        }
                        accelerator.log(logs, step=global_step)

                        if j > args.eval_steps:
                            break

                model.train()

            loss = open_clip.ClipLoss()

            images, texts = batch
            images = images.to(device=accelerator.device, non_blocking=True)
            texts = texts.to(device=accelerator.device, non_blocking=True)

            image_features, text_features, logit_scale = model(images, texts)
            total_loss = loss(image_features, text_features, logit_scale)

            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                params_to_clip = model.parameters()
                accelerator.clip_grad_norm_(params_to_clip, 1.0)  # args.max_grad_norm)

            optimizer.step()
            scheduler(step=global_step)
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_local_main_process:
                if args.save_steps and global_step - last_save >= args.save_steps:
                    last_save = global_step
                    lora_save = f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}.text_encoder.pt"
                    logger.info(f"Saving LoRA weights to {lora_save}")
                    model.save_pretrained(lora_save)
                    last_save = global_step

            logs = {
                "loss": total_loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        lora_save = (
            f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}.text_encoder.pt"
        )
        logger.info(f"Saving LoRA weights to {lora_save}")
        model.save_pretrained(lora_save)

    accelerator.print("\n\nLoRA Training completed.\n\n")
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="The path to the yaml file containing the training configuration.",
    )
    yaml = parser.parse_args().config
    args = parse_args(yaml)
    main(args)
