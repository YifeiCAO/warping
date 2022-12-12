#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --output=rnn.%j.out

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate warping

# Check if directory already exists,
# if it doesnt, create one.
if [ ! -d "~/results" ]; then
        mkdir -p "results"
        mkdir -p "results/tsv"
fi

if [ ! -d "~/figures" ]; then
        mkdir -p "figures"
fi

# Defaults
python main.py \
--use_cuda \
--out_file rnn.P \
--use_images \
--model_name rnn


# Ablation
for SCALE in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python main.py \
    --use_cuda \
    --out_file rnn_ablation_ctx_scale$SCALE.P \
    --use_images \
    --model_name rnn \
    --ctx_scale $SCALE
done