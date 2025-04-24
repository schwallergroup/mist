#!/bin/bash
#SBATCH --job-name=grpo-chem
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=./reward_logs/%x-%j.out
#SBATCH --err=./reward_logs/%x-%j.err
#SBATCH --environment=vllm071
#SBATCH -A a-a05

set -xe
source ~/.bashrc
cd /Documents/sink/src/open_r1/tasks/crystal_structure/reward_server

FLASK_IP=$(srun --ntasks=1 hostname -I | awk '{print \$1}')
echo "Flask is up at http://$FLASK_IP:9001"

pip install -e . --no-deps;
python app.py
