from __future__ import print_function
import sys
sys.path.extend(['.'])
project_directory = ''

import argparse
import pickle
import os
from itertools import product
from collections import Counter

import numpy as np
import librosa
from tqdm import tqdm
import pandas as pd
from einops import rearrange

from model.vggish_input import waveform_to_examples

# -----------------------------
# Paths
# -----------------------------
# audio_path template: two placeholders -> pair, speaker (e.g., "001_A.wav")
audio_path = "data/sho/{}_{}.wav"
keypoints_path_default = "data/sho/{}_{}.npy"

# -----------------------------
# Marker selection
# -----------------------------
markersbody = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_OUTER',
    'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT',
    'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX',
    'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL',
    'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]

markershands = [
    'LEFT_WRIST', 'LEFT_THUMB_CMC', 'LEFT_THUMB_MCP', 'LEFT_THUMB_IP', 'LEFT_THUMB_TIP',
    'LEFT_INDEX_FINGER_MCP', 'LEFT_INDEX_FINGER_PIP', 'LEFT_INDEX_FINGER_DIP', 'LEFT_INDEX_FINGER_TIP',
    'LEFT_MIDDLE_FINGER_MCP', 'LEFT_MIDDLE_FINGER_PIP', 'LEFT_MIDDLE_FINGER_DIP', 'LEFT_MIDDLE_FINGER_TIP',
    'LEFT_RING_FINGER_MCP', 'LEFT_RING_FINGER_PIP', 'LEFT_RING_FINGER_DIP', 'LEFT_RING_FINGER_TIP',
    'LEFT_PINKY_FINGER_MCP', 'LEFT_PINKY_FINGER_PIP', 'LEFT_PINKY_FINGER_DIP', 'LEFT_PINKY_FINGER_TIP',
    'RIGHT_WRIST', 'RIGHT_THUMB_CMC', 'RIGHT_THUMB_MCP', 'RIGHT_THUMB_IP', 'RIGHT_THUMB_TIP',
    'RIGHT_INDEX_FINGER_MCP', 'RIGHT_INDEX_FINGER_PIP', 'RIGHT_INDEX_FINGER_DIP', 'RIGHT_INDEX_FINGER_TIP',
    'RIGHT_MIDDLE_FINGER_MCP', 'RIGHT_MIDDLE_FINGER_PIP', 'RIGHT_MIDDLE_FINGER_DIP', 'RIGHT_MIDDLE_FINGER_TIP',
    'RIGHT_RING_FINGER_MCP', 'RIGHT_RING_FINGER_PIP', 'RIGHT_RING_FINGER_DIP', 'RIGHT_RING_FINGER_TIP',
    'RIGHT_PINKY_FINGER_MCP', 'RIGHT_PINKY_FINGER_PIP', 'RIGHT_PINKY_FINGER_DIP', 'RIGHT_PINKY_FINGER_TIP'
]

# body indices of: nose, left eye, right eye, left shoulder, right shoulder, left elbow, right elbow
selected_markers = [0, 2, 5, 11, 12, 13, 14]
# append all hands (offset by 33 body markers)
selected_markers.extend([33 + i for i in range(len(markershands))])
print(selected_markers)

# -----------------------------
# Canonical (30 fps) timings and FPS-agnostic conversion
# -----------------------------
# Original 30 fps configuration:
BASE_FPS = 30
BASE_TIME_OFFSET = 2   # frames @ 30 fps (≈ 66.7 ms hop)
BASE_NUM_FRAMES = 15   # frames @ 30 fps (0.5 s window along 'f')
BASE_HISTORY = 40      # steps (≈ 2.667 s of history)

HOP_SEC = BASE_TIME_OFFSET / BASE_FPS
WINDOW_F_SEC = BASE_NUM_FRAMES / BASE_FPS
HISTORY_SEC = BASE_HISTORY * HOP_SEC

def discretize_for_fps(fps: int):
    """
    Convert canonical seconds into integer frame counts for an arbitrary fps.
    Returns (time_offset, num_frames, history).
    """
    time_offset = max(1, round(HOP_SEC * fps))
    num_frames = max(1, round(WINDOW_F_SEC * fps))
    step_sec = time_offset / float(fps)
    history = max(1, round(HISTORY_SEC / step_sec))
    return int(time_offset), int(num_frames), int(history)

def seconds_to_frames(seconds: float, fps: int) -> int:
    return int(round(seconds * fps))

def frame_to_audio_sample(frame_idx: int, fps: int, sr: int = 16000) -> int:
    return int(round((frame_idx / float(fps)) * sr))

# -----------------------------
# Static parameters
# -----------------------------
sample_rate_audio = 16000
buffer_and_window_in_seconds = {0.0: 0.48, 0.25: 0.72, 0.5: 0.96}
num_channels = 3  # x,y,confidence (or similar)

selected_joints = {
    '59': np.concatenate((np.arange(0, 17), np.arange(91, 133)), axis=0),  # 59
    '31': np.concatenate((np.arange(0, 11),
                          [91, 95, 96, 99, 100, 103, 104, 107, 108, 111],
                          [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]), axis=0),  # 31
    '27': np.concatenate(([0, 5, 6, 7, 8, 9, 10],
                          [91, 95, 96, 99, 100, 103, 104, 107, 108, 111],
                          [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]), axis=0),  # 27
    'CABB': np.array(selected_markers)
}

# -----------------------------
# Overlap helpers
# -----------------------------
def calculate_overlap(start1, end1, start2, end2):
    earliest_start = min(start1, start2)
    latest_end = max(end1, end2)
    latest_start = max(start1, start2)
    earliest_end = min(end1, end2)
    overlap = 0
    if latest_start < earliest_end:
        overlap = (earliest_end - latest_start)
    total_duration = (end1 - start1)
    if total_duration > 0:
        percentage_overlap = (overlap / total_duration)
    else:
        percentage_overlap = 0
    return percentage_overlap

def overlap_percentage(window, annotation):
    overlap = max(0, min(window[1], annotation[1]) - max(window[0], annotation[0]) + 1)
    window_length = window[1] - window[0] + 1
    annotation_length = annotation[1] - annotation[0] + 1

    if window[0] >= annotation[0] and window[1] <= annotation[1]:
        percentage = overlap / window_length
        status = 'full'
        started_ended = 'inside'
    elif annotation[0] >= window[0] and annotation[1] <= window[1]:
        percentage = overlap / annotation_length
        status = 'full'
        started_ended = 'inside'
    elif window[1] < annotation[1]:
        percentage = overlap / window_length
        if percentage < 0.05:
            status = 'outside'
        elif percentage < 0.25:
            status = 'starting'
        elif percentage < 0.5:
            status = 'early'
        elif percentage < 0.75:
            status = 'middle'
        else:
            status = 'full'
        started_ended = 'started'
    else:
        percentage = overlap / window_length
        if percentage < 0.05:
            status = 'outside'
        elif percentage < 0.25:
            status = 'ending'
        elif percentage < 0.5:
            status = 'late'
        elif percentage < 0.75:
            status = 'middle'
        else:
            status = 'full'
        started_ended = 'ended'
    percentage = int(percentage * 10000) / 10000
    return percentage, status, started_ended

# -----------------------------
# Data I/O helpers
# -----------------------------
def _load_audio_for_pair_speaker(audio_tmpl, pair, speaker, sr=16000):
    """
    Try a couple of reasonable filename patterns to avoid brittle formatting errors.
    """
    candidates = [
        audio_tmpl.format(pair, speaker),
        f"data/sho/{pair}_{speaker}.wav",
        f"data/sho/{pair}_{pair}_{speaker}.wav",
    ]
    last_err = None
    for p in candidates:
        try:
            if not isinstance(p, str):
                continue
            if not os.path.exists(p):
                continue
            return librosa.load(p, sr=sr)
        except Exception as e:
            last_err = e
            continue
    # Fall back to formatting even if the file may not exist; let librosa raise.
    try:
        return librosa.load(audio_tmpl.format(pair, speaker), sr=sr)
    except Exception:
        if last_err is not None:
            raise last_err
        raise

def load_data(keypoints_path, pairs, speakers, config):
    data_dict = dict.fromkeys(product(pairs, speakers))
    for pair, speaker in product(pairs, speakers):
        pair_str = str(pair).zfill(3)
        kp_path = keypoints_path.format(pair_str, speaker.lower())
        keypoints = np.load(kp_path)
        # Optionally select joints (uncomment if you want to prune here)
        # selected = selected_joints[config]
        # keypoints = keypoints[:, selected, :]
        data_dict[(pair_str, speaker)] = keypoints
    return data_dict

def get_data_size(data_dict, pair_speaker, history, time_offset, num_frames):
    total_num_samples = 0
    for pair in pair_speaker:
        x = data_dict[pair].shape[0]
        case_num_samples = (x - (history + num_frames) * time_offset) // (time_offset * history) + 1
        if case_num_samples > 0:
            total_num_samples += case_num_samples
    return total_num_samples

def get_part_pair_speaker(data_df):
    unique_pair_speaker = {tuple(elem) for elem in data_df[["ID", "speaker"]].values}
    unique_pair_speaker = sorted(unique_pair_speaker)
    # Make sure ID are zero-padded like the files
    unique_pair_speaker = [(str(pid).zfill(3), spk) for (pid, spk) in unique_pair_speaker]
    return unique_pair_speaker

# -----------------------------
# Core generator
# -----------------------------
def generate_data_sw(
        data_dict,
        gestures_info,
        pair,
        speaker,
        all_speakers_fps,
        all_speakers_audio,
        sequences_lengths,
        all_sample_names,
        sequences_labels,
        current_ind,
        history,
        time_offset,
        num_frames,
        audio_path,
        buffer,
        fps,
        sr=16000
        ):
    """
    Sliding window generator using FPS-agnostic (seconds-based) parameters.
    """
    label_format = "overlap_percentage_{:7.5f}_status_{:7}"
    sample_format = "{}_{}_{:06d}_{:06d}"

    data = data_dict[(pair, speaker)]
    start_ind = 0
    end_ind = data.shape[0] - (history + num_frames) * time_offset

    # Load audio (16 kHz)
    input_audio, sample_rate = _load_audio_for_pair_speaker(audio_path, pair, speaker, sr=sr)
    assert sample_rate == sr, f"Expected sr={sr}, got {sample_rate}"

    # Frame-to-audio mapping at the given fps
    samples_per_frame = int(round(sr / float(fps)))  # 320 at 50 fps

    # Buffer (seconds) –> frames (may be fractional; we keep seconds exact below)
    # We want audio window to cover (num_frames / fps) + buffer seconds:
    num_frames_in_seconds = (num_frames / float(fps)) + float(buffer)
    corresponding_audio_window_size = int(round(num_frames_in_seconds * sr))

    # Iterate with step = time_offset * history (same logic as your original code)
    for t in tqdm(range(start_ind, max(start_ind, end_ind), time_offset * history), leave=False):
        # Visual stack: shape [num_frames, history, K, D] -> rearrange to [history, D, num_frames, K]
        arr = np.stack([
            data[t + i*time_offset: t + (history+i)*time_offset: time_offset, ...]
            for i in range(num_frames)
        ])
        arr = rearrange(arr, 'f t k d -> t d f k')
        all_speakers_fps[current_ind] = arr

        percentages = []
        statuses = []
        audio_features = []

        for i in range(t, t + history*time_offset, time_offset):
            start_frame = i
            end_frame = i + num_frames
            window = (start_frame, end_frame)

            # Audio slice aligned to the start frame; cover num_frames_in_seconds
            audio_start_sample = start_frame * samples_per_frame
            audio_end_sample = audio_start_sample + corresponding_audio_window_size

            if audio_start_sample >= len(input_audio):
                audio_window = np.zeros((corresponding_audio_window_size,), dtype=np.float32)
            else:
                # Pad right if needed
                right = max(0, audio_end_sample - len(input_audio))
                audio_window = input_audio[audio_start_sample: min(audio_end_sample, len(input_audio))]
                if right > 0:
                    audio_window = np.concatenate([audio_window, np.zeros((right,), dtype=audio_window.dtype)], axis=0)

            # VGGish features (window=hop=buffer_and_window_in_seconds[buffer])
            vggish_len_sec = buffer_and_window_in_seconds[buffer]
            vggish_input = waveform_to_examples(
                audio_window, sr,
                return_tensor=False,
                EXAMPLE_WINDOW_SECONDS=vggish_len_sec,
                EXAMPLE_HOP_SECONDS=vggish_len_sec
            )
            # Expect shape ~ (N=1, 96, 64) when vggish_len_sec ∈ {0.48,0.72,0.96} and window==hop
            audio_features.append(vggish_input.squeeze())

            # Gesture overlap/status
            all_intersections = gestures_info.apply(
                lambda x: overlap_percentage(window, (x["start_frame"], x["end_frame"])), axis=1
            )
            all_intersections = list(all_intersections)
            if len(all_intersections) == 0:
                status = 'outside'
                percentage = 0.0
            else:
                percentages_i = [elem[0] for elem in all_intersections]
                argmax_i = int(np.argmax(percentages_i))
                percentage = percentages_i[argmax_i]
                status = all_intersections[argmax_i][1]

            percentages.append(percentage)
            statuses.append(status)

        label = [label_format.format(p, s) for p, s in zip(percentages, statuses)]
        sample_names = [
            sample_format.format(pair, speaker, i, i + num_frames)
            for i in range(t, t + history * time_offset, time_offset)
        ]

        all_sample_names[current_ind] = sample_names
        sequences_labels[current_ind] = label
        all_speakers_audio[current_ind] = np.array(audio_features)
        current_ind += 1

    return (all_speakers_fps, all_speakers_audio, sequences_lengths,
            all_sample_names, sequences_labels, [None]*len(sequences_lengths), current_ind)

# -----------------------------
# Top-level driver
# -----------------------------
def gendata(
        all_data,
        label_path,
        out_path,
        audio_out_path,
        keypoints_path,
        part='train',
        config='27',
        save_video=False,
        audio_path=None,
        buffer=0.0,
        fps=50  # <-- default: handle 50 fps
        ):
    # Discretize canonical seconds for the target fps
    time_offset, num_frames, history = discretize_for_fps(fps)

    # Load keypoints
    pairs = np.unique(all_data['ID'].to_numpy())
    speakers = np.unique(all_data['speaker'].to_numpy())
    # Ensure zero-padded IDs early
    pairs = np.array([str(p).zfill(3) for p in pairs])
    data_dict = load_data(keypoints_path, pairs, speakers, config)

    pair_speaker = get_part_pair_speaker(all_data)
    total_num_samples = get_data_size(data_dict, pair_speaker, history, time_offset, num_frames)

    # Allocate arrays
    all_speakers_fps = np.zeros(
        (total_num_samples, history, num_channels, num_frames, int(config)),
        dtype=np.float32
    )
    # VGGish time dim: 48/72/96 frames for 0.48/0.72/0.96 s
    vggish_T = int(buffer_and_window_in_seconds[buffer] * 100)
    all_speakers_audio = np.zeros(
        (total_num_samples, history, vggish_T, 64),
        dtype=np.float32
    )

    sequences_lengths = history * np.ones(total_num_samples, dtype=np.int32)
    all_sample_names = np.empty((total_num_samples, history), dtype='U22')
    sequences_labels = np.empty((total_num_samples, history), dtype='U47')

    all_pair_speaker_referent = []
    current_ind = 0

    # Drive per (pair, speaker)
    for pair, speaker in tqdm(pair_speaker, leave=True):
        data = all_data[(all_data['ID'] == pair) & (all_data['speaker'] == speaker)]
        data = data.reset_index(drop=True)
        data = data[['start_frame', 'end_frame', 'speaker', 'ID', 'value']].drop_duplicates()
        iconic_gestures = data[data['value'].str.lower().str.strip() == 'gesture']

        ret_val = generate_data_sw(
            data_dict=data_dict,
            gestures_info=iconic_gestures,
            pair=pair,
            speaker=speaker,
            all_speakers_fps=all_speakers_fps,
            all_speakers_audio=all_speakers_audio,
            sequences_lengths=sequences_lengths,
            all_sample_names=all_sample_names,
            sequences_labels=sequences_labels,
            current_ind=current_ind,
            history=history,
            time_offset=time_offset,
            num_frames=num_frames,
            audio_path=audio_path,
            buffer=buffer,
            fps=fps,
            sr=sample_rate_audio
        )
        (all_speakers_fps, all_speakers_audio, sequences_lengths,
         all_sample_names, sequences_labels, _, current_ind) = ret_val

    # Trim any overhang due to empty labels at the end
    c = Counter()
    for label in sequences_labels:
        c.update([elem.split('_')[-1] for elem in label])

    overhead = c[''] // history if '' in c else 0
    print('overhead:', overhead)
    if overhead > 0:
        all_speakers_fps = all_speakers_fps[:-overhead]
        sequences_lengths = sequences_lengths[:-overhead]
        all_sample_names = all_sample_names[:-overhead]
        sequences_labels = sequences_labels[:-overhead]
        all_speakers_audio = all_speakers_audio[:-overhead]

    with open(label_path, 'wb') as f:
        pickle.dump((all_sample_names, all_pair_speaker_referent, sequences_labels, sequences_lengths), f)

    print(all_speakers_fps.shape)
    np.save(out_path, all_speakers_fps)
    np.save(audio_out_path, all_speakers_audio)
    print('saved to', out_path)

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process audio and video data (FPS-agnostic).')
    parser.add_argument('--keypoints-path', type=str, default=keypoints_path_default,
                        help='Path template to keypoints numpy files, e.g. "data/sho/{}_{}.npy"')
    parser.add_argument('--fps', type=int, default=50, help='Video FPS (default: 50)')
    args = parser.parse_args()

    keypoints_path = args.keypoints_path
    fps = int(args.fps)

    out_folder = project_directory + 'data/mm_data_vggish_sho/'
    history_canonical = BASE_HISTORY  # not used directly (derived per fps), kept for reference

    new_gestures_info_path = "data/sho/gestures_data_frames.csv"
    if os.path.exists(new_gestures_info_path):
        gestures_info = pd.read_csv(new_gestures_info_path)
    else:
        raise ValueError("Provide correct path to the gestures CSV file!")

    # Standardize columns / values
    gestures_info = gestures_info.rename(columns={'from_frame': 'start_frame', 'to_frame': 'end_frame'})
    gestures_info['ID'] = gestures_info['ID'].apply(lambda x: str(x).zfill(3))
    gestures_info['value'] = gestures_info['value'].str.lower().str.strip()
    gestures_info['pair_speaker'] = gestures_info.apply(lambda x: f"{x['ID']}_{x['speaker']}", axis=1)

    # Example subset (keep if you need the same filtering as before)
    gestures_info = gestures_info[gestures_info['ID'] == '001']
    gestures_info = gestures_info[gestures_info['speaker'] == 'A']
    gestures_info = gestures_info.reset_index(drop=True)

    # simple stats
    print('Number of unique labels:', gestures_info['value'].nunique())
    print('Number of items per label:')
    print(gestures_info['value'].value_counts())

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for buffer in [0.0, 0.25, 0.5]:
        data_out_path = f'{out_folder}data_27_joint_buffer_{buffer}.npy'
        audio_out_path_np = f'{out_folder}audio_buffer_{buffer}.npy'
        label_out_path = f'{out_folder}27_labels_buffer_{buffer}.pkl'
        referents_speakers_path = f'{out_folder}27_referent_speakers_buffer_{buffer}.pkl'

        gendata(
            gestures_info,
            label_out_path,
            data_out_path,
            audio_out_path_np,
            keypoints_path,
            part='train',
            config='27',
            save_video=False,
            audio_path=audio_path,
            buffer=buffer,
            fps=fps
        )
