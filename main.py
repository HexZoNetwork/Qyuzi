import argparse
import sys
import os

# Ensure the current directory is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train
from generate import generate

def main():
    parser = argparse.ArgumentParser(description="Qyuzi AGI Entry Point")
    parser.add_argument('mode', nargs='?', choices=['train', 'generate', 'chat'], default='train', help='Mode to run: train, generate, or chat')
    parser.add_argument('--prompt', type=str, default="The future of AGI is", help='Prompt for generation')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("🚀 Launching QYUZI Training Protocol...")
        train()
    elif args.mode in ['generate', 'chat']:
        print(f"🧠 QYUZI Inference (Prompt: '{args.prompt}')")
        generate(args.prompt)

if __name__ == "__main__":
    main()
