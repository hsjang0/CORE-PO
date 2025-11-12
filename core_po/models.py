from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, prepare_model_for_kbit_training
import torch


def import_llama3(model_name: str, lora_config, gpu_id: int, load_in_8bit: bool = True):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        device_map=gpu_id,
    )
    base_model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(base_model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eot = "<|eot_id|>"
    eot_id = tokenizer.convert_tokens_to_ids(eot)
    tokenizer.pad_token = "<|pad_id|>"
    model.config.pad_token = "<|pad_id|>"
    tokenizer.pad_token_id = 104001
    model.config.pad_token_id = 104001
    model.config.eos_token = eot
    model.config.eos_token_id = eot_id
    model.train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model, tokenizer
