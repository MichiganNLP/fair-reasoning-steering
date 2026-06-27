#!/bin/bash
#SBATCH --job-name=gpt-oss-inference
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --partition=<partition-redacted>
#SBATCH --account=<account-redacted>
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=<email-redacted>
#SBATCH --output=<REPO_ROOT>/slurm_logs/%x-%A_%a.log

# set -euo pipefail
# module load singularity

# SIF=<HOME>/scratch/singularity/trace-verl.sif
# BIND_OPTS="-B <REPO_ROOT>:<REPO_ROOT> \
#            -B <HF_HOME>:<HF_HOME>"

conda activate factory
USER_SCRIPT=$1

# benchmark-results.fair-gcg-seed-phrase-inj-len-ablation
# # Map array index -> inj_len
# lens=(4 8 10 16 32)
# INJ_LEN=${lens[$SLURM_ARRAY_TASK_ID]}
# # DATA_IDX=${SLURM_ARRAY_TASK_ID}
# echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}, using INJ_LEN=${INJ_LEN}"
# --env DATA_IDX=${DATA_IDX} \
# --env INJ_LEN=${INJ_LEN} \

# singularity exec --nv --cleanenv $BIND_OPTS \
#   --env PYTHONUNBUFFERED=1 \
#   --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   $SIF bash -lc "
#   set -euo pipefail
#   export PATH=/usr/local/bin:/usr/bin:/bin
#   nvidia-cuda-mps-control -d
#   which python3; python3 --version
#   nvidia-smi || true
#   bash '$USER_SCRIPT'
# "
bash '$USER_SCRIPT'