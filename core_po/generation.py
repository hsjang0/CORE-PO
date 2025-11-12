from typing import List, Sequence, Tuple

import torch


def prompt_for_multiple_choice(question: str, options: str) -> str:
    return f"""<|start_header_id|>system<|end_header_id|>

Be a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Answer the following question using **reasoning** before providing a final answer. Provide precise, structured, and well-reasoned responses.

**Question:** {question}

### Response Format

**Understanding the question:** <identify key details>

**Reasoning:** <perform chain-of-thought>

**Final answer:** "The answer is <choose the most promising single answer from {options}> which is <copy the content>"

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

**Final answer:** "The answer is $<value>$"

Ensure correctness and clarity. Return a concise and definitive response to the question. STRICTLY FOLLOW THE RESPONSE FORMAT.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""


def generate_answer(prompt: str, model, tokenizer, max_length: int = 1560, temperature: float = 1.0) -> List[str]:
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
    return [
        tokenizer.decode(output).split(
            "STRICTLY FOLLOW THE RESPONSE FORMAT.<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"
        )[1]
        for output in output_ids
    ]


def get_preference_annotated_responses(question: str, model, tokenizer, judge) -> Tuple[Sequence[str], Sequence[str], Sequence[str]]:
    if question.count("[I]") >= 1 and question.count("[III]") >= 1:
        options = "[I] / [II] / [III] / [IV] / [V]" if question.count("[V]") >= 1 else "[I] / [II] / [III] / [IV]"
        prompt = prompt_for_multiple_choice(question, options)
    else:
        prompt = prompt_for_numerical_response(question)
    answers = generate_answer(prompt, model, tokenizer)

    conf_set = judge.judge(question, answers)
    answers = [ans for _, ans in sorted(zip(conf_set, answers), reverse=True)]

    prompts_ind = []
    chosen_ind = []
    rejected_ind = []

    for answer in answers[1:]:
        prompts_ind.append(prompt)
        best = answers[0]
        if best.count("<|eot_id|>") > 0:
            chosen_ind.append(prompt + best.split("<|eot_id|>")[0] + "<|eot_id|>")
        else:
            chosen_ind.append(prompt + best)
        if answer.count("<|eot_id|>") > 0:
            rejected_ind.append(prompt + answer.split("<|eot_id|>")[0] + "<|eot_id|>")
        else:
            rejected_ind.append(prompt + answer)

    return prompts_ind, chosen_ind, rejected_ind
