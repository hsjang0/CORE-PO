import copy
import random
from typing import List

import numpy as np
import torch


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

    def compute_likelihood_with_response(self, prompt: str) -> float:
        combined_prompts = [prompt + "A"] + [prompt + "B"]
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
        return prob[0].item() / (prob[0] + prob[1]).item()

    def judge(self, question: str, answer_set: List[str]) -> np.ndarray:
        conf_reasoning: List[float] = []
        conf_answer: List[float] = []
        reasoning_set: List[str] = []
        final_answer_set: List[str] = []
        flags = []

        for answer in answer_set:
            try:
                if answer.count("<|eot_id|>") == 0:
                    flags.append(True)
                    continue
                reasoning = (
                    answer.split("**Understanding the question:**")[1]
                    .split("**Final answer:**")[0]
                    .replace("\n**Reasoning:**", "")
                )
                final_answer = answer.split("**Final answer:**")[1].split("<|eot_id|>")[0]
                reasoning_set.append(reasoning)
                final_answer_set.append(final_answer)
                flags.append(False)
            except Exception:
                flags.append(True)
                continue

        prompts = []
        prompt_header = self.reason_header + f"\n\n**Question:** {question}\n\n"
        for reasoning in reasoning_set:
            if len(reasoning_set) >= self.max_number:
                support_set = random.sample(reasoning_set, self.max_number)
            else:
                support_set = random.sample(reasoning_set, len(reasoning_set))
            random.shuffle(support_set)

            in_context = ""
            for idx, support_reasoning in enumerate(support_set):
                in_context += (
                    f"**Randomly generated reasoning {idx + 1} "
                    f"(this may be either correct or incorrect):** "
                    + support_reasoning
                )
            in_context += "\n\n**Selected reasoning:** " + reasoning
            in_context += (
                "\n\nIs the **selected reasoning** correct?\nA) True\nB) False\n"
                "The **selected reasoning** is: [A / B, depending on whether "
                "the **selected reasoning** is correct given the **question**]"
                "<|eot_id|>"
            )
            prompt = (
                prompt_header
                + in_context
                + "\n<|start_header_id|>assistant<|end_header_id|>\n\n"
                "The **selected reasoning** is: "
            )
            prompts.append(copy.deepcopy(prompt))
        for prompt in prompts:
            conf_reasoning.append(self.compute_likelihood_with_response(prompt))

        prompts = []
        prompt_header = (
            self.answer_header + f"\n\n**Question:** {question}\n\n**Reasoning:** "
        )
        for reasoning, final_answer in zip(reasoning_set, final_answer_set):
            in_context = reasoning
            in_context += "\n\n**Selected answer:** " + final_answer
            in_context += (
                "\n\nIs the **selected answer** correct?\nA) True\nB) False\n"
                "The **selected answer** is: [A / B, depending on whether the "
                "**selected answer** is correct given the **question** and the "
                "**reasoning**]<|eot_id|>"
            )
            prompt = (
                prompt_header
                + in_context
                + "\n<|start_header_id|>assistant<|end_header_id|>\n\n"
                "The **selected answer** is: "
            )
            prompts.append(copy.deepcopy(prompt))
        for prompt in prompts:
            conf_answer.append(self.compute_likelihood_with_response(prompt))

        for idx, flag in enumerate(flags):
            if flag:
                conf_answer.insert(idx, 0)
                conf_reasoning.insert(idx, 0)

        return np.array(conf_reasoning) * np.array(conf_answer)
