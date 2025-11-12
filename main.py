import os
from transformers import HfArgumentParser
from core_po import ScriptArguments, run_training

def main() -> None:
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    os.makedirs(
        os.path.join(script_args.save_directory, script_args.save_name),
        exist_ok=True,
    )
    run_training(script_args)

if __name__ == "__main__":
    main()
