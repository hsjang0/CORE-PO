import datasets
import random


def get_training_data(gpu_id, num_gpus):
    question_set = []
    pick = ['I','II','III','IV', 'V']
    question_set = {
        'gsm8k': [],
        'arc': [],
        'gpqa': [],
        'math': []
    }
    train_dataset = datasets.load_dataset("openai/gsm8k", "main")['train']
    for question in train_dataset['question'][gpu_id::num_gpus]:
        question_set['gsm8k'].append(question)


    train_dataset = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge")['train']
    for question, true_answer_set in zip(train_dataset['question'][gpu_id::num_gpus], 
                                            train_dataset["choices"][gpu_id::num_gpus]):
        question = question + " Choices:\n"
        for e, true_answer_set_i in enumerate(true_answer_set['text']):
            question += f'[{pick[e]}]: '+true_answer_set_i+ '\n'
        question_set['arc'].append(question)


    train_dataset = datasets.load_dataset("DigitalLearningGmbH/MATH-lighteval", "default")['train']
    for question in train_dataset["problem"][gpu_id::num_gpus]:  
        question_set['math'].append(question)


    train_dataset = datasets.load_dataset("Idavidrein/gpqa", "gpqa_main")['train']
    for (question, true_answer, incorrect1, incorrect2, incorrect3) in zip(train_dataset["Question"][gpu_id::num_gpus], 
                                                                            train_dataset["Correct Answer"][gpu_id::num_gpus], 
                                                                            train_dataset["Incorrect Answer 1"][gpu_id::num_gpus], 
                                                                            train_dataset["Incorrect Answer 2"][gpu_id::num_gpus], 
                                                                            train_dataset["Incorrect Answer 3"][gpu_id::num_gpus]):
        true_answer_set = [true_answer, incorrect1, incorrect2, incorrect3]
        question = question + " Choices:\n"
        random.shuffle(true_answer_set) 
        for e, true_answer_set_i in enumerate(true_answer_set):
            question += f'[{pick[e]}]: '+true_answer_set_i+ '\n'
        if len(question) > 1280:
            continue
        question_set['gpqa'].append(question)
        

    for key in question_set.keys():
        random.shuffle(question_set[key])
        
        
    return question_set







