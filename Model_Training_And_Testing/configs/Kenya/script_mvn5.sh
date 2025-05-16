#!/bin/bash
#SBATCH --job-name=MVN-TEST5
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=24G
#SBATCH --output=computecantest5.out
#SBATCH --account=rrg-bengioy-ad

module load python/3.10

virtualenv --no-download $SLURM_TMPDIR/myenv_test
source $SLURM_TMPDIR/myenv_test/bin/activate

pip install --no-index -r requirements.txt

module load libspatialindex/1.9.3

python train_ccan_3_constr.py --config configs/Kenya/resnet_test_ccan5.yaml

deactivate