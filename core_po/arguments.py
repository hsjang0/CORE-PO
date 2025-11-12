from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScriptArguments:
    save_directory: Optional[str] = field(default="./dpo_saved/")
    learning_rate: Optional[float] = field(default=5e-6, metadata={"help": "Optimizer learning rate"})
    batch_size: Optional[int] = field(default=4, metadata={"help": "Per-rank batch size"})
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "Load base model in 8-bit"})
    early_stopping: Optional[bool] = field(default=True, metadata={"help": "Enable early stopping logic"})
    max_grad_norm: Optional[float] = field(default=1.0, metadata={"help": "Gradient clipping norm"})
    save_name: Optional[str] = field(default="OURS", metadata={"help": "Subdirectory for checkpoints"})
    top_p: Optional[float] = field(default=0.9, metadata={"help": "Nucleus sampling probability"})
    top_k: Optional[int] = field(default=40, metadata={"help": "Top-k sampling cutoff"})
