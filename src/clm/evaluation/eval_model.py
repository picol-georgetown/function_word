import os
import sys
import argparse
from glob import glob
import torch  

# Dynamically set project root and source paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
scripts_path = os.path.join(project_root, "multiblimp", "scripts", "lm_eval")
src_path = os.path.join(project_root, "multiblimp", "src", "lm_eval")

sys.path.append(scripts_path)
sys.path.append(src_path)

from load_model import load_hf_model
from score import score_tse

# Check if CUDA is available and print status
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Now that src_path is defined, it's safe to use as a default
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', help='LM to evaluate', required=True)
parser.add_argument('--revision', help='BGP-T training step', default="main")
parser.add_argument('--data_dir', help='Minimal pair directory', default="final_pairs")
parser.add_argument('--src_dir', help='Source directory', default=src_path)
parser.add_argument('--results_dir', help='Dir to write results to', default=None)
parser.add_argument('--cache_dir', help='(optional) HF cache dir', default="/scratch-shared/jumelet")
parser.add_argument('--hf_token', help='Huggingface token (file or token itself)', default=None)
args = parser.parse_args()

# Handle hf_token input
hf_token = None  # Default to None
if args.hf_token:
    if os.path.exists(args.hf_token):  # If a file path is provided
        with open(args.hf_token) as f:
            hf_token = f.read().strip()  # Read the token from the file
    else:
        hf_token = args.hf_token  # Use the provided token directly

print(f"Loading model: {args.model_name} @ step {args.revision}")
lm = load_hf_model(
    args.model_name,
    no_cache=False,
    token=hf_token,  # Will pass None if not provided
    cache_dir=args.cache_dir,
    revision_step=str(args.revision)
)

pair_files = glob(os.path.join(args.data_dir, "**/*.tsv"), recursive=True)
if not pair_files:
    print(f"No .tsv files found in: {args.data_dir}")
    sys.exit(1)

for fn in sorted(pair_files):
    path_parts = fn.split(os.sep)
    if len(path_parts) < 3:
        print(f"Skipping malformed path: {fn}")
        continue

    phenomenon, lang, condition = path_parts[-3:]
    
    # Extract filename without extension
    condition_name = os.path.splitext(condition)[0]

    df = score_tse(lm, fn=fn)
    if df is None or df.empty:
        print(f"No data returned for: {fn}")
        continue

    print(f"{phenomenon} | {lang} | {condition_name} → Score: {(df.sen_nll < df.wrong_nll).mean():.3f}")

    results_dir = args.results_dir or os.path.join("model_results", args.model_name)
    os.makedirs(results_dir, exist_ok=True)

    # Generate output filename with only one .tsv extension
    score_fn = os.path.join(results_dir, f"{phenomenon}_{lang}_{condition_name}.tsv")
    df.to_csv(score_fn, sep='\t')
    print(f"Saved: {score_fn}")
