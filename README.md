# Overview

This repository contains code for training multimodal gesture detection model.

# Generate data

* Provide a path to gesture keypoints in the form
`$PATH_TO_KEYPOINTS/{pair}_synced_pp{speaker}.npy`

* Run data generation using
`python data/CABB_gen_audio_video_data_vggish.py`

---

# Run the pipeline

python run_skeletal_speech_framework.py -- config $PATH_TO_CONFIG

In the dafault configuration the script will use skeleton data only. The following parameters can be changed to reproduce experiments from the paper:
1. `fusion: model.audio_video_labelers.{Skeleton,Speech,LateFusion,EarlyFusion,CrossAttn}`
2. `fusion_args.buffer: {0,0.25,0.5}`

# Slurm

It is possible to generate data and submit training scrits using Slu rm.
For data generation use
`sbatch generate_vggish_data.sh`

For submitting a training job use
`sbatch run_experiment.sh`

