import os
import numpy as np
import yaml
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count
import argparse
from pathlib import Path

sys.path.insert(0, '.')

from utilities import AudioProcessor

def preprocess(data_dir, output_dir, metadata_file, num_jobs, config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs(output_dir, exist_ok=True)
    print('[INFO] Root directory:', data_dir)

    audio_processor = AudioProcessor(**config['audio'])
    executor = ProcessPoolExecutor(max_workers=num_jobs)
    futures = []

    with open(metadata_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(data_dir, f'{parts[0]}.wav')
            text = parts[2]
            if not os.path.exists(wav_path):
                print(f"[WARNING] File does not exist: {wav_path}")
                continue
            job = executor.submit(partial(process_utterance, wav_path, text, output_dir, audio_processor))
            futures.append(job)

    print('[INFO] Preprocessing =>', len(futures), 'audio files found')
    results = [future.result() for future in tqdm(futures)]
    
    results = [result for result in results if result is not None]

    output_metadata_path = os.path.join(output_dir, 'ljspeech_meta.txt')
    with open(output_metadata_path, 'w') as f:
        for result in results:
            f.write('|'.join(map(str, result)) + '\n')

def process_utterance(wav_path, text, output_dir, audio_processor, store_mel=True, store_linear=True):
    try:
        wav = audio_processor.load_wav(wav_path)
        mel = audio_processor.melspectrogram(wav).astype(np.float32).T
        linear = audio_processor.spectrogram(wav).astype(np.float32).T
        n_frames = linear.shape[0]
        file_id = Path(wav_path).stem
        mel_path = f'{file_id}-mel.npy'
        linear_path = f'{file_id}-linear.npy'
        if store_mel:
            np.save(os.path.join(output_dir, mel_path), mel, allow_pickle=False)
        if store_linear:
            np.save(os.path.join(output_dir, linear_path), linear, allow_pickle=False)
        return mel_path, linear_path, n_frames, text
    except Exception as e:
        print(f"[ERROR] Failed to process {wav_path}: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess audio data for training.')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing the wav files.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the processed data.')
    parser.add_argument('--metadata-file', type=str, required=True, help='Path to the metadata file.')
    parser.add_argument('--config-path', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--num-jobs', type=int, default=cpu_count(), help='Number of parallel jobs.')

    args = parser.parse_args()

    preprocess(args.data_dir, args.output_dir, args.metadata_file, args.num_jobs, args.config_path)
