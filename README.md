## Self-Training Large Language Models with Confident Reasoning

The official code base for reproducing self-training in **[Self-Training Large Language Models with Confident Reasoning](https://aclanthology.org/2025.findings-emnlp.806/)**. The training code involves (1) sampling math/science questions, (2) generating multiple answers with a LoRA-tuned Llama 3.1, (3) scoring them with an internal judge, and (4) running Direct Preference Optimization (DPO).

---

### Repository Layout

- `main.py` – CLI entry point that parses arguments and launches training.
- `core_po/`
  - `arguments.py` – dataclass for CLI/configuration options.
  - `data.py` – dataset loaders for GSM8K, ARC, MATH, GPQA with rank-aware slicing.
  - `generation.py` – prompt templates, response sampling, preference construction.
  - `judge.py` – confidence estimator for reasoning and answers.
  - `models.py` – helpers to load LoRA-wrapped Llama 3.1 checkpoints.
  - `trainer.py` – LoRA/DPO training loop, checkpointing, and optimizer wiring.
  - `__init__.py` – exposes `ScriptArguments` and `run_training`.

---

### Quick Start for Training

The training requires four NVIDIA A100 GPUs.

```bash
accelerate launch --num_processes 4 main.py \
  --save_directory ./dpo_saved \
  --save_name ours_run \
  --learning_rate 5e-6 \
  --batch_size 4
```

---

### BibTeX

```bibtex
@inproceedings{jang-etal-2025-self,
  title     = {Self-Training Large Language Models with Confident Reasoning},
  author    = {Jang, Hyosoon and Jang, Yunhui and Lee, Sungjae and Ok, Jungseul and Ahn, Sungsoo},
  editor    = {Christodoulopoulos, Christos and Chakraborty, Tanmoy and Rose, Carolyn and Peng, Violet},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  month     = {nov},
  year      = {2025},
  address   = {Suzhou, China},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2025.findings-emnlp.806/},
  doi       = {10.18653/v1/2025.findings-emnlp.806},
  pages     = {14925--14939},
  isbn      = {979-8-89176-335-7}
}
```
