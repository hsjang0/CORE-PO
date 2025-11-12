import random
from typing import Dict, List

import datasets


TaskPool = Dict[str, List[str]]


def get_training_data(gpu_id: int, num_gpus: int) -> TaskPool:
    question_set: TaskPool = {
        "gsm8k": [],
        "arc": [],
        "gpqa": [],
        "math": [],
    }
    pick = ["I", "II", "III", "IV", "V"]

    train_dataset = datasets.load_dataset("openai/gsm8k", "main")["train"]
    for question in train_dataset["question"][gpu_id::num_gpus]:
        question_set["gsm8k"].append(question)

    train_dataset = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge")["train"]
    for question, choice_block in zip(
        train_dataset["question"][gpu_id::num_gpus],
        train_dataset["choices"][gpu_id::num_gpus],
    ):
        full_question = question + " Choices:\n"
        for idx, option in enumerate(choice_block["text"]):
            full_question += f"[{pick[idx]}]: {option}\n"
        question_set["arc"].append(full_question)

    train_dataset = datasets.load_dataset("DigitalLearningGmbH/MATH-lighteval", "default")["train"]
    for question in train_dataset["problem"][gpu_id::num_gpus]:
        question_set["math"].append(question)

    train_dataset = datasets.load_dataset("Idavidrein/gpqa", "gpqa_main")["train"]
    for question, true_answer, incorrect1, incorrect2, incorrect3 in zip(
        train_dataset["Question"][gpu_id::num_gpus],
        train_dataset["Correct Answer"][gpu_id::num_gpus],
        train_dataset["Incorrect Answer 1"][gpu_id::num_gpus],
        train_dataset["Incorrect Answer 2"][gpu_id::num_gpus],
        train_dataset["Incorrect Answer 3"][gpu_id::num_gpus],
    ):
        true_answer_set = [true_answer, incorrect1, incorrect2, incorrect3]
        random.shuffle(true_answer_set)
        prompt = question + " Choices:\n"
        for idx, option in enumerate(true_answer_set):
            prompt += f"[{pick[idx]}]: {option}\n"
        if len(prompt) > 1280:
            continue
        question_set["gpqa"].append(prompt)

    for key in question_set:
        random.shuffle(question_set[key])

    return question_set
