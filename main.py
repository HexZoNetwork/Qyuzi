import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Qyuzi System')
    
    parser.add_argument('--stage', type=str, default='fh',
                       choices=['f', 'fh', 'sec', 'fih'],
                       help='Select model stage')
    
    parser.add_argument('--snn', action='store_true',
                       help='Enable Spiking Neural Network module')
    parser.add_argument('--vsa', action='store_true',
                       help='Enable Vector-Symbolic Architecture')
    parser.add_argument('--dream', action='store_true',
                       help='Enable Dream Engine')
    parser.add_argument('--selfmodel', action='store_true',
                       help='Enable Self-Modeling module')
    parser.add_argument('--multimodal', action='store_true',
                       help='Enable multi-modal')
    parser.add_argument('--swarm', action='store_true',
                       help='Enable swarm deployment mode')
    
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'generate', 'eval', 'swarm'],
                       help='Mode')
    parser.add_argument('--prompt', type=str, default='',
                       help='Your Question')
    parser.add_argument('--tokens', type=int, default=200,
                       help='Max Token')
    parser.add_argument('--temp', type=float, default=0.8, #Not temporary dude ;-;
                       help='Sampling temperature')
    parser.add_argument('--steps', type=int, default=8,
                       help='Number of thinking iterations')
    
    parser.add_argument('--nodes', type=int, default=1000,
                       help='Swarm nodes')
    parser.add_argument('--topology', type=str, default='hierarchical',
                       choices=['hierarchical', 'mesh', 'ring', 'star'],
                       help='idk')
    
    args = parser.parse_args()
    
    os.environ['QYUZI_STAGE'] = args.stage
    if args.snn:
        os.environ['QYUZI_SNN'] = '1'
    if args.vsa:
        os.environ['QYUZI_VSA'] = '1'
    if args.dream:
        os.environ['QYUZI_DREAM'] = '1'
    if args.selfmodel:
        os.environ['QYUZI_SELFMODEL'] = '1'
    
    from qyuzi import config, endless_think_training, generate
    
    print("="*70)
    print("  ██████╗ ██╗   ██╗██╗   ██╗███████╗██╗")
    print(" ██╔═══██╗╚██╗ ██╔╝██║   ██║╚══███╔╝██║")
    print(" ██║   ██║ ╚████╔╝ ██║   ██║  ███╔╝ ██║")
    print(" ██║▄▄ ██║  ╚██╔╝  ██║   ██║ ███╔╝  ██║")
    print(" ╚██████╔╝   ██║   ╚██████╔╝███████╗██║")
    print("  ╚══▀▀═╝    ╚═╝    ╚═════╝ ╚══════╝╚═╝")
    print("="*70)
    print(f"    🚀 QYUZI - {args.stage}")
    print("="*70)
    print(f"Model: {config.VERSION}")
    print(f"Architecture: {config.NUM_LAYERS}L x {config.HIDDEN}H x {config.FFN_DIM}FFN")
    print(f"Mode: {'MoE ' + str(config.NUM_EXPERTS) + 'x' + str(config.EXPERTS_ACTIVE) if config.USE_MOE else 'Dense'}")
    print(f"Context: {config.MAX_SEQ} tokens")
    
    features = []
    if args.snn:
        features.append("SNN")
    if args.vsa:
        features.append("VSA")
    if args.dream:
        features.append("Dream")
    if args.selfmodel:
        features.append("SelfModel")
    if args.multimodal:
        features.append("MultiModal")
    if features:
        print(f"Advanced: {', '.join(features)}")
    print("="*70)
    print()
    
    if args.mode == 'train':
        print("Ah Ma Ga Train")
        endless_think_training()
    
    elif args.mode == 'generate':
        if not args.prompt:
            print("use --prompt")
            sys.exit(1)
        
        print(f"Using Brain...\n")
        result = generate(
            args.prompt,
            max_new=args.max_new,
            think_steps=args.think_steps,
            temperature=args.temperature
        )
        print("\n" + "="*70)
        print("RESULT:")
        print("="*70)
        print(result)
        print("="*70)
    
    elif args.mode == 'swarm':
        import asyncio
        from qyuzi_swarm import QuantumSwarm, SwarmTopology
        
        print(f"Swarm node f{args.num_nodes}\n")
        
        topology_map = {
            'hierarchical': SwarmTopology.HIERARCHICAL,
            'mesh': SwarmTopology.MESH,
            'ring': SwarmTopology.RING,
            'star': SwarmTopology.STAR
        }
        
        swarm = QuantumSwarm(num_nodes=args.num_nodes)
        swarm.coordinator.topology = topology_map[args.topology]
        
        async def run_swarm():
            await swarm.deploy()
            print("\nAhhh Gyattt")
        
        asyncio.run(run_swarm())
    
    elif args.mode == 'eval':
        print("📊 Running evaluation suite...\n")
        from qyuzi import QyuziUltimate
        import torch
        
        model = QyuziUltimate().to(config.DEVICE)
        model.load_checkpoint()
        model.eval()
        
        tasks = [
            ("What is consciousness?", "Philosophy"),
            ("Explain quantum entanglement.", "Physics"),
            ("def fibonacci(n):", "Coding"),
            ("The meaning of life is", "Reasoning")
        ]
        
        print("Running evaluation tasks...\n")
        for prompt, category in tasks:
            result = generate(prompt, max_new=100, think_steps=8)
            print(f"[{category}] {prompt[:30]}...")
            print(f"→ {result[:100]}...\n")

if __name__ == "__main__":
    main()
