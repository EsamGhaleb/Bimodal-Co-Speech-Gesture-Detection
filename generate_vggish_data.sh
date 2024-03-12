#!/bin/bash -l

#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=72
#SBATCH --partition=singlenode
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --job-name=data_buffer_data

unset SLURM_EXPORT_ENV

module load python
python -m venv /tmp/$SLURM_JOB_ID.fritz/foo

source /tmp/$SLURM_JOB_ID.fritz/foo/bin/activate

python -m pip install -q -r requirements.txt

python data/CABB_gen_audio_video_data_vggish.py 