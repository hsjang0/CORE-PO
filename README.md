## Self-Training Large Language Models with Confident Reasoning

The official code base for reproducing self-training in **[Self-Training Large Language Models with Confident Reasoning](https://aclanthology.org/2025.findings-emnlp.806/)**. The training code involves (1) sampling math/science questions, (2) generating multiple answers with a LoRA-tuned Llama 3.1, (3) scoring them with an internal judge, and (4) running Direct Preference Optimization (DPO).

- `CORE_PO.py` – main training loop, LoRA setup, DPO updates, checkpoint saving.
- `training_data.py` – pulls GSM8K, ARC-Challenge, MATH, GPQA with rank-based sharding.
- `utils.py` – prompt templates, judge class, generation helpers, Llama loader.

---

### Quick Start for Training

The training requires four NVIDIA A100 GPUs.

```bash
accelerate launch --num_processes 4 CORE_PO.py \
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
