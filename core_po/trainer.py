import gc
import os
import random
from typing import Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from datasets import Dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

from .arguments import ScriptArguments
from .data import get_training_data
from .generation import get_preference_annotated_responses
from .judge import Judge
from .models import import_llama3


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def setup_dpo_trainer(
    model,
    tokenizer,
    script_args: ScriptArguments,
    accelerator: Accelerator,
) -> Tuple[torch.nn.Module, DPOTrainer]:
    dpo_config = DPOConfig(
        output_dir="./dpo_output",
        max_grad_norm=script_args.max_grad_norm,
        num_train_epochs=1,
        remove_unused_columns=False,
        beta=0.1,
    )
    placeholder_dataset = Dataset.from_dict(
        {
            "prompt": ["Example prompt"],
            "chosen": ["Example chosen response"],
            "rejected": ["Example rejected response"],
        }
    )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=script_args.learning_rate,
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    dpo_trainer = DPOTrainer(
        model=accelerator.unwrap_model(model),
        args=dpo_config,
        processing_class=tokenizer,
        optimizers=(optimizer, None),
        train_dataset=placeholder_dataset,
    )
    dpo_trainer.accelerator = accelerator
    return model, dpo_trainer


def run_training(script_args: ScriptArguments) -> None:
    accelerator = Accelerator()
    process_id = accelerator.local_process_index
    print(f"process: {process_id}")

    lora_config = build_lora_config()
    model, tokenizer = import_llama3(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        lora_config,
        gpu_id=process_id,
        load_in_8bit=script_args.load_in_8bit,
    )
    model, dpo_trainer = setup_dpo_trainer(model, tokenizer, script_args, accelerator)

    judge = Judge(accelerator.unwrap_model(model), tokenizer)

    num_gpus = torch.cuda.device_count()
    question_set = get_training_data(process_id, num_gpus)

    step = 0
    while len(question_set.keys()) > 0:
        problem_type = random.choice(list(question_set.keys()))
        question = question_set[problem_type].pop(0)
        if len(question_set[problem_type]) < 1:
            del question_set[problem_type]

        step += 1

        local_results = get_preference_annotated_responses(
            question,
            accelerator.unwrap_model(model),
            tokenizer,
            judge,
        )

        accelerator.wait_for_everyone()
        gc.collect()
        torch.cuda.empty_cache()

        prompts = tokenizer(local_results[0], return_tensors="pt")
        chosen = tokenizer(
            local_results[1],
            padding=True,
            truncation=True,
            max_length=1560,
            return_tensors="pt",
        )
        rejected = tokenizer(
            local_results[2],
            padding=True,
            truncation=True,
            max_length=1560,
            return_tensors="pt",
        )
        lengths_ = prompts["input_ids"].to(model.device).shape[1]
        batch = {
            "prompt_input_ids": prompts["input_ids"].to(model.device)[:, :-1],
            "prompt_attention_mask": prompts["attention_mask"].to(model.device)[:, :-1],
            "chosen_input_ids": chosen["input_ids"].to(model.device)[:, lengths_-1:],
            "chosen_attention_mask": chosen["attention_mask"].to(model.device)[:, lengths_-1:],
            "rejected_input_ids": rejected["input_ids"].to(model.device)[:, lengths_-1:],
            "rejected_attention_mask": rejected["attention_mask"].to(model.device)[:, lengths_-1:],
        }

        dpo_trainer.training_step(model, batch)
        dpo_trainer.accelerator.gradient_state._set_sync_gradients(True)
        if script_args.max_grad_norm is not None and script_args.max_grad_norm > 0:
            _grad_norm = dpo_trainer.accelerator.clip_grad_norm_(
                model.parameters(),
                script_args.max_grad_norm,
            )
        if dpo_trainer.accelerator.distributed_type == DistributedType.DEEPSPEED:
            grad_norm = model.get_global_grad_norm()
            if hasattr(grad_norm, "item"):
                grad_norm = grad_norm.item()
        else:
            grad_norm = _grad_norm
        dpo_trainer.optimizer.step()
        model.zero_grad()

        accelerator.wait_for_everyone()
        if dpo_trainer.accelerator.is_main_process and step % 200 == 0 and step != 0:
            save_path = os.path.join(
                script_args.save_directory,
                script_args.save_name,
                f"step_{step}",
            )
            accelerator.unwrap_model(model).save_pretrained(save_path)
            print(f"step {step}: model saved")

        gc.collect()
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
