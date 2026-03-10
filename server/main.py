import argparse
import re
import os

import torch
from torch.utils.data.dataloader import DataLoader

from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    DataCollatorForTokenClassification,
    BitsAndBytesConfig
)

# import wandb

from lora import LoraConfig, get_peft_model, print_trainable_parameters
from data_loader import load_data, encode_data
from trainer import Trainer
import signal



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="sst2", 
                        choices=["sst2", "rte", "boolq", "wsc", "wic", "multirc", "copa", "winogrande", "squad", "drop", "mrpc",
                                 "qqp", "qnli", "wnli", "arc_e", "arc_c", "hellaswag"
                                 ],
                        help="The name of the GLUE task to train on.")
    parser.add_argument("--model_name_or_path", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                        choices=["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-3B-Instruct" ,"microsoft/phi-2"],
                        help="Path to pretrained model.")
    parser.add_argument("--optimizer", type=str, default="zo", choices=["fo", "zo"], help="Optimizer.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum input sequence length after tokenization.")
    parser.add_argument("--n", type=int, default=1, help="Number of pertubation.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Training batch size per device.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Evaluation batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--eps", type=float, default=1e-2, help="Perturbation scale.")
    parser.add_argument("--lowrank", type=str2bool, default=True, help="Use low-rank noise (row/col outer product) for 2D parameters in ZO.")
    parser.add_argument("--num_train_epochs", type=int, default=1000, help="Total number of training epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training.")
    parser.add_argument("--quantization", type=str, default="no", choices=["no", "8bit", "4bit"], help="Quantization.")
    parser.add_argument("--enable_mixed_layer_quantization", type=str2bool, default=False,
                        help="Enable layer-wise mixed INT4/INT8 quantization before training.")
    parser.add_argument("--mixed_int4_ratio", type=float, default=0.01, help="Target ratio of parameters to quantize into INT4 (0~1).")
    parser.add_argument("--mixed_int8_ratio", type=float, default=0.2, help="Target ratio of parameters to quantize into INT8 (0~1).")
    parser.add_argument("--mixed_quant_calib_batches", type=int, default=2, help="Number of train batches used for sensitivity estimation.")
    parser.add_argument("--quant_alpha", type=float, default=0.0, help="Alpha weight in score for quantization gain.")
    parser.add_argument("--quant_beta", type=float, default=0.0, help="Beta weight in score for quantization gain.")
    parser.add_argument("--zo_mode", type=str, default="single", choices=["single", "dual"], help="Number of forward pass.")
    parser.add_argument("--enable_early_exit", type=str2bool, default=False, help="Enable SNR-based early exit for ZO single mode.")
    parser.add_argument("--gamma", type=float, default=0.5, help="SNR threshold for early exit.")
    parser.add_argument("--max_resample_k", type=int, default=3, help="Maximum resampling attempts on the same batch before skipping it.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--max_iterations", type=int, default=25000, help="Log every X epochs.")
    parser.add_argument("--split",  type=str2bool, default=False , help="Split batch between perturbations.")
    parser.add_argument("--mixed_precision",  type=str2bool, default=False , help="Split batch between perturbations.")
    parser.add_argument('--peft', type=str, choices=["no", "lora", "lora-fa"], default="lora-fa", help="PEFT method.")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--compute_dtype", type=str, default="fp16", choices=["bf16" ,"fp16", "fp32"], help="Compute dtype.")
    parser.add_argument('--total_batch_size', type=int, default=None, help='Total batch size for training')
    parser.add_argument('--lora_alpha', type=float, default=16, help='LoRA alpha')
    parser.add_argument("--torch_optimizer", type=str, default="adam", help='Torch optimizer')
    parser.add_argument('--zero_shot', type=str2bool, default=False, help='Zero shot training')
    parser.add_argument('--zo_sign', type=str2bool, default=False, help='Enable ZO-SignSGD (take the sign of the estimated gradient)')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if not 0.0 <= args.mixed_int4_ratio <= 1.0:
        raise ValueError("mixed_int4_ratio must be in [0, 1].")
    if not 0.0 <= args.mixed_int8_ratio <= 1.0:
        raise ValueError("mixed_int8_ratio must be in [0, 1].")
    if args.mixed_int4_ratio + args.mixed_int8_ratio > 1.0:
        raise ValueError("mixed_int4_ratio + mixed_int8_ratio must be <= 1.")
    if args.enable_mixed_layer_quantization and args.quantization != "no":
        raise ValueError("When enable_mixed_layer_quantization=True, please set --quantization no.")
    if args.mixed_quant_calib_batches <= 0:
        raise ValueError("mixed_quant_calib_batches must be > 0.")
    if args.gamma < 0:
        raise ValueError("gamma must be >= 0.")
    if args.max_resample_k < 1:
        raise ValueError("max_resample_k must be >= 1.")
    num_gpus = torch.cuda.device_count()
    if args.total_batch_size is not None:
        assert args.total_batch_size == args.per_device_train_batch_size * num_gpus, "Total batch size must be equal to per_device_train_batch_size * num_gpus"

    print(f"Run args: {args}")
    set_seed(args.seed)
    # compute_dtype = torch.float32 if args.optimizer == "fo" else torch.float16
    compute_types = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


    if args.optimizer == "fo":
        args.compute_dtype = "fp32"
    compute_dtype = compute_types[args.compute_dtype]

    flag = -1
    if args.lowrank and args.enable_mixed_layer_quantization and args.enable_early_exit and args.per_device_train_batch_size == 16 and not args.zo_sign:
        flag = 1
    if not args.lowrank and not args.enable_mixed_layer_quantization and not args.enable_early_exit and args.per_device_train_batch_size == 16 and not args.zo_sign:
        flag = 2
    if not args.lowrank and not args.enable_mixed_layer_quantization and not args.enable_early_exit and args.per_device_train_batch_size != 16 and not args.zo_sign:
        flag = 3
    if args.zo_sign:
        flag = 4
    if not args.lowrank and args.enable_mixed_layer_quantization and args.enable_early_exit:
        flag = 5
    if args.lowrank and not args.enable_mixed_layer_quantization and args.enable_early_exit:
        flag = 6
    if args.lowrank and args.enable_mixed_layer_quantization and not args.enable_early_exit:
        flag = 7
    if args.lowrank and not args.enable_mixed_layer_quantization and not args.enable_early_exit:
        flag = 8
    if not args.lowrank and args.enable_mixed_layer_quantization and not args.enable_early_exit:
        flag = 9
    if not args.lowrank and not args.enable_mixed_layer_quantization and args.enable_early_exit:
        flag = 10
    if args.optimizer == "fo":
        flag = 11
    if args.peft == "lora-fa" and args.n == 1:
        flag = 12
    if args.zero_shot:
        flag = 13
    if args.per_device_train_batch_size == 4 and args.lowrank and args.enable_mixed_layer_quantization and args.enable_early_exit and not args.zo_sign:
        flag = 14
    if args.per_device_train_batch_size == 4 and args.lowrank and args.enable_mixed_layer_quantization and not args.enable_early_exit and not args.zo_sign:
        flag = 15
    if args.per_device_train_batch_size == 2 and args.lowrank and args.enable_mixed_layer_quantization and args.enable_early_exit and not args.zo_sign:
        flag = 16
    if args.per_device_train_batch_size == 1 and args.lowrank and args.enable_mixed_layer_quantization and args.enable_early_exit and not args.zo_sign:
        flag = 17
    if flag == 12:
        run_name += f"-lora_rank{args.lora_rank}-lora_alpha{args.lora_alpha}"
    
    run_name = f"{flag}--{args.task_name}-{args.model_name_or_path}"
    args.run_name = run_name
    args.run_name_file = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name)
    print(f"Run name: {run_name}")
        

    if args.quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=compute_dtype, quantization_config=quantization_config, device_map="auto")
    print(f"Memory footprint (GB): {model.get_memory_footprint() / 1024**3}") 

    if args.peft != "no":
        lora_config = LoraConfig(
            r=args.lora_rank, 
            lora_alpha=args.lora_alpha, 
            n=args.n,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)

    if args.peft in ["lora", "lora-fa"]:
        for name, param in model.named_parameters():
            if (args.peft == "lora-fa" and "lora_B" in name) or (args.peft == "lora" and "lora" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif args.peft == "no":
        for name, param in model.named_parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown peft mode: {args.peft}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, padding_side="left", truncation_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"{args.model_name_or_path} model tokenizer: set pad token")

    raw_datasets = load_data(args)
    train_dataset, eval_dataset, cls_idx = encode_data(args, tokenizer, raw_datasets)

    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    optimizer = None

    if args.optimizer == "fo":
        if args.torch_optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        elif args.torch_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
            print("Using SGD optimizer")
        else:
            raise ValueError(f"Unknown torch optimizer: {args.torch_optimizer}")
        accelerator = Accelerator(mixed_precision=('bf16' if args.mixed_precision else 'no'))
        print(f"Using mixed precision: {args.mixed_precision} with compute dtype: bf16")
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model,
            optimizer,
            DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size,  drop_last=args.split),
            DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        )
    else:
        accelerator = Accelerator()
        model, train_dataloader, eval_dataloader = accelerator.prepare(
            model,
            DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size,  drop_last=args.split),
            DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        )

    trainer = Trainer(args, model, tokenizer, train_dataloader, eval_dataloader, accelerator, cls_idx, optimizer)
    if args.zero_shot:
        trainer.zero_shot_eval()
    else:
        trainer.train()

    # model.save_pretrained("output")
    # tokenizer.save_pretrained("output")

    # wandb.finish()


if __name__ == "__main__":
    main()
