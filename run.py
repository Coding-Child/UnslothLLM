import warnings
import argparse
from scripts.main import main
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, default='unsloth/llama-2-7b-chat-bnb-4bit')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-a', '--accumulation_step', type=int, default=1)
    parser.add_argument('-w', '--warmup_step', type=int, default=2000)
    parser.add_argument('-s', '--using_scheduler', type=bool, default=False)
    parser.add_argument('-st', '--scheduler_type', type=str, default='cosine')
    parser.add_argument('-e', '--num_epochs', type=int, default=500)
    parser.add_argument('-ml', '--max_len', type=int, default=1024)
    parser.add_argument('-trn', '--train_path', type=str, default='data/english-train_addon.json')
    parser.add_argument('-val', '--val_path', type=str, default='data/english-dev_addon.json')
    parser.add_argument('-tst', '--test_path', type=str, default='data/english-test_addon.json')
    parser.add_argument('-sp', '--save_path', type=str, default='checkpoints')
    parser.add_argument('-r', '--r', type=int, default=16)
    parser.add_argument('-ld', '--lora_dropout', type=float, default=0)
    parser.add_argument('-la', '--lora_alpha', type=float, default=16)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)
