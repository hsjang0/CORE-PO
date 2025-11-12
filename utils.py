from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
import random
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftConfig
import numpy as np
from peft import LoraConfig
import re
import copy



def prompt_for_multiple_choice(question: str, options: str) -> str:
    return f"""<|start_header_id|>system<|end_header_id|>

Be a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Answer the following question using **reasoning** before providing a final answer. Provide precise, structured, and well-reasoned responses.

**Question:** {question}

### Response Format

**Understanding the question:** <identify key details>

**Reasoning:** <perform chain-of-thought>

**Final answer:** \"The answer is <choose the most promising single answer from {options}> which is <copy the content>\"

Ensure correctness and clarity. Return a concise and definitive response to the question. DO NOT RETURN TWO OR MORE ANSWERS. STRICTLY FOLLOW THE RESPONSE FORMAT.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""

def prompt_for_numerical_response(question: str) -> str:
    return f"""<|start_header_id|>system<|end_header_id|>

Be a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Answer the following question using **reasoning** before providing a final answer. Provide precise, structured, and well-reasoned responses.

**Question:** {question}

### Response Format

**Understanding the question:** <identify key details>

**Reasoning:** <perform chain-of-thought>

**Final answer:** \"The answer is $<value>$\"

Ensure correctness and clarity. Return a concise and definitive response to the question. STRICTLY FOLLOW THE RESPONSE FORMAT.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""


class Judge:
    def __init__(self, model, tokenizer):
        self.model = model    
        self.tokenizer = tokenizer
        self.max_number = 4    
        self.reason_header = """<|start_header_id|>system<|end_header_id|>
    
Be a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Answer whether the **selected reasoning** is correct for the given **question**. Additionally, we provide randomly generated reasoning before presenting the selected reasoning."""

        self.answer_header = """<|start_header_id|>system<|end_header_id|>

Be a helpful assistant<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Answer whether the **selected answer** is correct for the given **question**, based on the provided **reasoning**."""


    def compute_likelihood_with_response(self, prompt):
        combined_prompts = [prompt + "A"] + [prompt + "B" ]
        inputs = self.tokenizer(combined_prompts, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs.input_ids, return_dict=True)
        labels = inputs.input_ids[:, 1:]
        logits = outputs.logits[:, :-1, :]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        response_log_probs = log_probs.gather(2, labels.unsqueeze(2)).squeeze(2)
        lengths = inputs.attention_mask[:, 1:].sum(dim=1)
        batch_indices = torch.arange(inputs.input_ids.size(0), device="cuda")
        last_log_prob = response_log_probs[batch_indices, lengths - 1]
        
        prob = torch.exp(last_log_prob)
        return prob[0].item() / (prob[0]+prob[1]).item()

    
    def judge(self, question, answer_set):
        conf_reasoning = []
        conf_answer = []
        reasoning_set = []
        final_answer_set = []
        
        # Extract reasoning and final answers
        flags = []
        for e, answer in enumerate(answer_set):
            try:
                if answer.count("<|eot_id|>") == 0:
                    flags.append(True)
                    continue
                extract_reasoning = answer.split('**Understanding the question:**')[1].split('**Final answer:**')[0].replace("\n**Reasoning:**","")
                extract_final_answer = answer.split('**Final answer:**')[1].split('<|eot_id|>')[0]
                reasoning_set.append(extract_reasoning)
                final_answer_set.append(extract_final_answer)
                flags.append(False)
            except:
                flags.append(True)
                continue    
                
        
        # Measure confidence in reasoning
        prompts = []
        prompt_header = self.reason_header + f"\n\n**Question:** {question}\n\n" 
        for _, reasoning in enumerate(reasoning_set):
            if len(reasoning_set)>=self.max_number: 
                support_set = random.sample(reasoning_set,self.max_number)
            else:
                support_set = random.sample(reasoning_set,len(reasoning_set))
            random.shuffle(support_set)
            
            in_context = ""
            for e, support_reasoning in enumerate(support_set):
                in_context += f"**Randomly generated reasoning {e+1} (this may be either correct or incorrect):** " + support_reasoning
            in_context += "\n\n**Selected reasoning:** " + reasoning
            in_context += "\n\nIs the **selected reasoning** correct?\nA) True\nB) False\nThe **selected reasoning** is: [A / B, depending on whether the **selected reasoning** is correct given the **question**]<|eot_id|>"
            prompt = prompt_header + in_context + "\n<|start_header_id|>assistant<|end_header_id|>\n\nThe **selected reasoning** is: "
            prompts.append(copy.deepcopy(prompt)) 
        for prompt in prompts:
            conf_reasoning.append(self.compute_likelihood_with_response(prompt))


        prompts = []        
        prompt_header = self.answer_header + f"\n\n**Question:** {question}\n\n**Reasoning:** "
        for e, reasoning in enumerate(reasoning_set):
            in_context = reasoning
            in_context += "\n\n**Selected answer:** "+ final_answer_set[e]
            in_context += "\n\nIs the **selected answer** correct?\nA) True\nB) False\nThe **selected answer** is: [A / B, depending on whether the **selected answer** is correct given the **question** and the **reasoning**]<|eot_id|>"
            prompt = prompt_header + in_context + "\n<|start_header_id|>assistant<|end_header_id|>\n\nThe **selected answer** is: "
            prompts.append(copy.deepcopy(prompt)) 
        for prompt in prompts:
            conf_answer.append(self.compute_likelihood_with_response(prompt))
        
        
        for e, flag in enumerate(flags):
            if flag:
                conf_answer.insert(e,0)
                conf_reasoning.insert(e,0)
        return np.array(conf_reasoning)*np.array(conf_answer)    



def import_llama3(model_name, lora_config, gpu_id, load_in_8bit=True):
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
    tokenzier_vocab_size = len(tokenizer)  
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model, tokenizer




def generate_answer(prompt, model, tokenizer, max_length=1560, temperature=1.0):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=5,
        )
    answers = [tokenizer.decode(output).split("STRICTLY FOLLOW THE RESPONSE FORMAT.<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n")[1] for output in output_ids]
    return answers



def get_preference_annotated_responses(question, model, tokenizer, judge): 
    if question.count('[I]') >=1 and question.count('[III]') >=1: 
        if question.count('[V]') >= 1:
            options = "[I] / [II] / [III] / [IV] / [V]"
        else:
            options = "[I] / [II] / [III] / [IV]"
        prompt = prompt_for_multiple_choice(question, options)
    else:
        prompt = prompt_for_numerical_response(question)
    answers = generate_answer(prompt, model, tokenizer)


    chosen_ind = []
    rejected_ind = []
    prompts_ind = []
    
    
    conf_set = judge.judge(question, answers)
    answers = [ans for _, ans in sorted(zip(conf_set, answers), reverse=True)]
    for answer in answers[1:]:
        prompts_ind.append(prompt)
        if answers[0].count('<|eot_id|>') > 0:
            chosen_ind.append(prompt+answers[0].split('<|eot_id|>')[0]+'<|eot_id|>')
        else:
            chosen_ind.append(prompt+answers[0])
        if answer.count('<|eot_id|>') > 0:
            rejected_ind.append(prompt+answer.split('<|eot_id|>')[0]+'<|eot_id|>')
        else:
            rejected_ind.append(prompt+answer)
    return prompts_ind, chosen_ind, rejected_ind