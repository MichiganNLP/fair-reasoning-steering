#!/bin/bash
#SBATCH --job-name=factory.llama3.sft
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=144G
#SBATCH --time=2-00:00:00
#SBATCH --partition=<partition-redacted>
#SBATCH --account=<account-redacted>
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=<email-redacted>
#SBATCH --output=<REPO_ROOT>/slurm_logs/%x-%j.log

set -euo pipefail

USER_SCRIPT=$1

conda activate factory

module load cuda
