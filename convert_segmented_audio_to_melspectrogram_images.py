import os
import argparse
import math

from tqdm import tqdm

import numpy as np
import librosa

import cv2


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio_dir", type=str, default=None)
    parser.add_argument("--audio_format", type=str, default="mp3")
    parser.add_argument("--hop_length", type=int, default=2048)
    parser.add_argument("--sample_frames", type=int, default=8192)
    parser.add_argument("--sr_desired", type=int, default=44100)
    parser.add_argument("--n_mels", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seconds_per_segments", type=int, default=10)

    args = parser.parse_args()

    return args


def enumerate_audio_files(audio_dir: str, audio_format: str):
    audio_files = []

    for dirs, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(audio_format) and not file.startswith("."):
                audio_files.append(os.path.join(dirs, file))

    return audio_files


def read_audio_file(audio_path: str, sr_desired=44100):
    y, sr = librosa.load(audio_path, sr=None)

    if sr != sr_desired:
        y = librosa.core.resample(y, sr, sr_desired)
        sr = sr_desired

    return y, sr


def scale_min_max(y: np.ndarray, min=0., max=1.):
    y_std = (y - y.min()) / (y.max() - y.min())
    y_scaled = y_std * (max - min) + min
    return y_scaled


if __name__ == '__main__':
    args = get_args()

    num_ts_per_segments = args.seconds_per_segments * args.sr_desired

    if args.save_dir is not None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    audio_files = enumerate_audio_files(args.audio_dir, args.audio_format)

    print("Number of audio files: {}".format(len(audio_files)))

    audio_files.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])

    if os.path.exists("index.txt"):
        with open("index.txt", "r") as f:
            index = int(f.read())
    else:
        index = 0
    audio_files = audio_files[index:]

    for file in tqdm(audio_files):
        #print("Audio file: {}".format(file))
        y, sr = read_audio_file(file, args.sr_desired)
        #print("y shape: {}".format(y.shape))
        num_segments = math.ceil(y.shape[0] / num_ts_per_segments)
        print(num_segments)
        #segments = np.split(y, num_segments)
        for i in range(num_segments):
            segment = y[i * num_ts_per_segments: (i + 1) * num_ts_per_segments]
            #print("segment shape: {}".format(segment.shape))

            if segment.shape[0] < 250000:
                continue
            if segment.shape[0] >= num_ts_per_segments:
                segment = segment[:num_ts_per_segments]
            else:
                segment = np.pad(segment, (0, num_ts_per_segments - segment.shape[0]), mode="constant")

            mel_spectrogram = librosa.feature.melspectrogram(segment, sr=sr,
                                                             n_fft=args.sample_frames,
                                                             hop_length=args.hop_length,
                                                             n_mels=args.n_mels)

            mel_spectrogram = np.log(mel_spectrogram + 1e-9)
            img = scale_min_max(mel_spectrogram, 0, 255).astype(np.uint8)
            img = np.flip(img, axis=0)
            img = 255 - img

            #print(img.shape)

            relpath = os.path.relpath(file, args.audio_dir)
            save_dir = os.path.join(args.save_dir, os.path.dirname(relpath))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            cv2.imwrite(os.path.join(save_dir, os.path.splitext(os.path.basename(file))[0] + "_" + str(i) + ".jpg"), img)

        index += 1
        with open("index.txt", "w+") as f:
            f.write(str(index))
