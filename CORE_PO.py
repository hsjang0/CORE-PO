import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
import random
from tqdm import tqdm
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
import numpy as np
import pandas as pd
import gc
tqdm.pandas()
from peft import LoraConfig
import re
from transformers import (
        LlamaForCausalLM,
        LlamaTokenizer,
    StoppingCriteria,
)
from peft import PeftModel, get_peft_model
import copy
from accelerate.utils import DistributedType
from utils import import_llama3, Judge, get_preference_annotated_responses
from training_data import get_training_data


@dataclass
class ScriptArguments:
    save_directory: Optional[str] = field(default='./dpo_saved/')
    learning_rate: Optional[float] = field(default=5e-6, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=4, metadata={"help": "the batch size"})
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "loading model in 8 bit or bfloat16"})
    early_stopping: Optional[bool] = field(default=True, metadata={"help": "whether to early stop"})
    max_grad_norm: Optional[float] = field(default=1.0, metadata={"help": "Maximum gradient norm for gradient clipping"})
    save_name: Optional[str] = field(default='OURS', metadata={"help": "Name for this experiment"})
    top_p: Optional[float] = field(default=0.9, metadata={'help':"the path to the sft model; need to merge if using lora"})
    top_k: Optional[int] = field(default=40, metadata={'help':"the path to the sft model; need to merge if using lora"})


accelerator = Accelerator()
process_id = accelerator.local_process_index 
gpu_id = process_id
current_device = process_id
print('process: {}'.format(process_id))
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
os.makedirs(os.path.join(script_args.save_directory, script_args.save_name), exist_ok=True)




# Prepare LLMs
lora_config = LoraConfig(
    r=128, 
    lora_alpha=256, 
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  
    bias="none",
    task_type="CAUSAL_LM",
)
model, tokenizer = import_llama3("meta-llama/Meta-Llama-3.1-8B-Instruct", lora_config, gpu_id, script_args.load_in_8bit)




# Import DPO class to utilize their loss functions
dpo_config = DPOConfig(
    output_dir="./dpo_output",             
    max_grad_norm=script_args.max_grad_norm,
    num_train_epochs=1,
    remove_unused_columns=False,
    beta=0.1,
)
# This is just a placeholder (we do not use this)
placeholder_dataset = Dataset.from_dict({
    "prompt": ["Example prompt"],
    "chosen": ["Example chosen response"],
    "rejected": ["Example rejected response"],
})
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=script_args.learning_rate)
accelerator = Accelerator()
model, optimizer = accelerator.prepare(model, optimizer)
dpo_trainer = DPOTrainer(
    model=accelerator.unwrap_model(model),
    args=dpo_config,
    processing_class=tokenizer,
    optimizers=(optimizer, None),
    train_dataset=placeholder_dataset
)
dpo_trainer.accelerator = accelerator


# Prepare a function for confidence estimation
judge = Judge(accelerator.unwrap_model(model), tokenizer)


# Prepare dataset
num_gpus = torch.cuda.device_count()
question_set = get_training_data(gpu_id, num_gpus)
    
        


# NOTE THAT FINE-TUNING USUALLY REQUIRES 400–800 STEPS TO YIELD BEST PERFORMANCE (AS WE USE LORA)
step = 0
while len(question_set.keys()) > 0:
    problem_type = random.choice(list(question_set.keys()))
    question = question_set[problem_type].pop(0)
    if len(question_set[problem_type]) < 1:
        del question_set[problem_type]
        
    step = step + 1


    # Sample outputs
    local_results = get_preference_annotated_responses(question, accelerator.unwrap_model(model), tokenizer, judge)  
      
    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()
    
    
    # Process data
    prompts = tokenizer(local_results[0], return_tensors="pt")
    chosen = tokenizer(local_results[1], padding=True, truncation=True, max_length=1560, return_tensors="pt")
    rejected = tokenizer(local_results[2], padding=True, truncation=True, max_length=1560, return_tensors="pt")
    lengths_ = prompts["input_ids"].to(model.device).shape[1]
    batch = {
        "prompt_input_ids": prompts["input_ids"].to(model.device)[:,:-1], "prompt_attention_mask": prompts["attention_mask"].to(model.device)[:,:-1],
        "chosen_input_ids": chosen["input_ids"].to(model.device)[:,lengths_-1:], "chosen_attention_mask": chosen["attention_mask"].to(model.device)[:,lengths_-1:],
        "rejected_input_ids": rejected["input_ids"].to(model.device)[:,lengths_-1:], "rejected_attention_mask": rejected["attention_mask"].to(model.device)[:,lengths_-1:],
    }


    # Optimization
    tr_loss_step = dpo_trainer.training_step(model, batch)
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

                    
    # Save
    accelerator.wait_for_everyone()
    if dpo_trainer.accelerator.is_main_process and step % 200 == 0 and step!=0:
        save_path = os.path.join(script_args.save_directory, script_args.save_name, 'step_{}'.format(step))
        accelerator.unwrap_model(model).save_pretrained(save_path)
        print("step {}: model saved".format(step))
        
                
    gc.collect()
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
