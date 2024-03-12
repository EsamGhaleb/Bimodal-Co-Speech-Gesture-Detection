#!/bin/bash -l

#SBATCH --job-name=logs/co-speech-gesture-detection
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --time=12:00:00
#SBATCH --export=NONE    

unset SLURM_EXPORT_ENV

module load python
python -m venv /tmp/$SLURM_JOB_ID.alex/foo
source /tmp/$SLURM_JOB_ID.alex/foo/bin/activate
python -m pip install -q -r requirements.txt

echo $1

python main_skeletal_speech_framework.py \
        --work-dir /home/atuin/b105dc/data/work/iburenko/co-speech-v2/co-speech-gesture-detection/results \
        --config $1
