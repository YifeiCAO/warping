import argparse
import torch
import numpy as np
import random
import pickle

from models import get_model
from train_meta_rl import train_meta_rl

parser = argparse.ArgumentParser()
# Basic setup
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_every', type=int, default=10)
parser.add_argument('--test_every', type=int, default=50)
parser.add_argument('--analyze_every', type=int, default=100)
parser.add_argument('--out_file', default='meta_results.P')

# Dataset
parser.add_argument('--use_images', action='store_true')
parser.add_argument('--image_dir', default='images/faces16')
parser.add_argument('--grid_size', type=int, default=4)
parser.add_argument('--inner_4x4', action='store_true')

# Training
parser.add_argument('--n_runs', type=int, default=5)
parser.add_argument('--n_steps', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)

# Meta-learning settings
parser.add_argument('--meta_batch_size', type=int, default=4)
parser.add_argument('--k_support', type=int, default=8)
parser.add_argument('--k_query', type=int, default=16)

# Model
parser.add_argument('--model_name', default='meta_rnn', choices=['meta_rnn', 'rnn', 'mlp', 'step_mlp', 'trunc_rnn', 'mlp_cc'])


def main(args):
    # Set device
    device = torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() else "cpu")
    args.device = device
    print("Using device:", device)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    all_results, all_analyses = [], []

    for run_i in range(args.n_runs):
        print(f"Starting run {run_i}...")

        # Initialize model
        model = get_model(args)
        model.to(args.device)

        # Meta-training
        results_i, analysis_i = train_meta_rl(run_i, model, args)
        all_results.append(results_i)
        all_analyses.append(analysis_i)

    # Save to file
    output = {'results': all_results, 'analysis': all_analyses}
    with open(f'results/{args.out_file}', 'wb') as f:
        pickle.dump(output, f)
    print(f"[Saved] Meta-RL results saved to results/{args.out_file}")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
