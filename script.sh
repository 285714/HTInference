#!/bin/bash -l

#$ -j y
#$ -m ea
#$ -pe omp 16

# module load python3/3.8.10
# module load cvxpy
# pip install sklearn # "sklearn>=1.2.1"

module load miniconda
module load julia
conda init zsh
conda activate ht_env
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LD_LIBRARY_PATH=/projectnb/cet-lab/fspaeh/.conda/envs/std_env/lib:$LD_LIBRARY_PATH

pip install julia
pip install pyjulia

echo "STARTING"
python scc_run.py
echo "DONE"

