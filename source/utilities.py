import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from symbols import txt2seq
from functools import partial

import librosa
import librosa.filters
import numpy as np
from scipy import signal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join
import soundfile as sf

def createDataLoader(mode, meta_path, data_dir, batch_size, r, n_jobs, use_gpu, **kwargs):
    shuffle = True if mode == 'train' else False if mode == 'test' else NotImplementedError
    dataset = CustomDataset(meta_path, data_dir)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=n_jobs, collate_fn=partial(collate_batch, r=r), pin_memory=use_gpu
    )
    return data_loader

def pad_sequence(seq, max_length):
    return np.pad(seq, (0, max_length - len(seq)), mode='constant')

def pad_2d_sequence(seq, max_length):
    return np.pad(seq, ((0, max_length - len(seq)), (0, 0)), mode='constant')

class CustomDataset(Dataset):
    
    def __init__(self, meta_path, data_dir):
        metadata = {'text': [], 'mel': [], 'spec': []}
        with open(meta_path) as file:
            for line in file.readlines():
                mel_file, spec_file, n_frames, text = line.split('|')
                metadata['text'].append(text)
                metadata['mel'].append(mel_file)
                metadata['spec'].append(spec_file)

        self.texts = metadata['text']
        self.mel_files = [os.path.join(data_dir, f) for f in metadata['mel']]
        self.spec_files = [os.path.join(data_dir, f) for f in metadata['spec']]
        assert len(self.texts) == len(self.mel_files) == len(self.spec_files)
        self.texts = [txt2seq(text) for text in self.texts]

    def __getitem__(self, idx):
        return (
            self.texts[idx],
            np.load(self.mel_files[idx]),
            np.load(self.spec_files[idx])
        )

    def __len__(self):
        return len(self.texts)

def collate_batch(batch, r):

    input_lengths = [len(item[0]) for item in batch]
    max_input_length = np.max(input_lengths)
    max_target_length = np.max([len(item[1]) for item in batch]) + 1
    
    if max_target_length % r != 0:
        max_target_length += r - max_target_length % r
    assert max_target_length % r == 0

    input_batch_np = np.array([pad_sequence(item[0], max_input_length) for item in batch])
    input_batch = torch.LongTensor(input_batch_np)
    input_lengths = torch.LongTensor(input_lengths)

    mel_batch_np = np.array([pad_2d_sequence(item[1], max_target_length) for item in batch])
    mel_batch = torch.FloatTensor(mel_batch_np)

    spec_batch_np = np.array([pad_2d_sequence(item[2], max_target_length) for item in batch])
    spec_batch = torch.FloatTensor(spec_batch_np)
    
    return input_batch, input_lengths, mel_batch, spec_batch
class AudioProcessor(object):
    
    def __init__(self, sample_rate, num_mels, num_freq, frame_length_ms, frame_shift_ms, preemphasis,
            min_level_db, ref_level_db, griffin_lim_iters, power):
        self.sr = sample_rate
        self.n_mels = num_mels
        self.n_fft = (num_freq - 1) * 2
        self.hop_length = int(frame_shift_ms / 1000 * sample_rate)
        self.win_length = int(frame_length_ms / 1000 * sample_rate)
        self.preemph = preemphasis
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db
        self.GL_iter = griffin_lim_iters
        self.mel_basis = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels)
        self.power = power

    def load_wav(self, path):
        return librosa.core.load(path, sr=self.sr)[0]

    def save_wav(self, wav, path):
        
        sf.write(path, wav, self.sr, subtype='PCM_16')

    def preemphasis(self, wav):
        return signal.lfilter([1, -self.preemph], [1], wav)

    def inv_preemphasis(self, wav_preemph):
        return signal.lfilter([1], [1, -self.preemph], wav_preemph)

    def spectrogram(self, wav):
        D = self._stft(self.preemphasis(wav))
        S = self._amp_to_db(np.abs(D)) - self.ref_level_db
        return self._normalize(S)

    def inv_spectrogram(self, linear_spect):
        
        S = self._db_to_amp(self._denormalize(linear_spect) + self.ref_level_db)  
        return self.inv_preemphasis(self._griffin_lim(S ** self.power))  
        
    def melspectrogram(self, wav):
        D = self._stft(self.preemphasis(wav))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D)))
        return self._normalize(S)

    def _griffin_lim(self, S):
        
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex128)
        y = self._istft(S_complex * angles)
        for i in range(self.GL_iter):
          angles = np.exp(1j * np.angle(self._stft(y)))
          y = self._istft(S_complex * angles)
        return y

    def _stft(self, x):
        return librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)

    def _istft(self, x):
        return librosa.istft(x, hop_length=self.hop_length, win_length=self.win_length)

    def _linear_to_mel(self, linear_spect):
        return np.dot(self.mel_basis, linear_spect)

    def _amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def _normalize(self, x):
        return np.clip((x - self.min_level_db) / -self.min_level_db, 0, 1)

    def _denormalize(self, x):
        return (np.clip(x, 0, 1) * -self.min_level_db) + self.min_level_db

def make_spec_figure(spec, audio_processor):
    spec = audio_processor._denormalize(spec)
    fig = plt.figure(figsize=(16, 10))
    plt.imshow(spec.T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    return fig

def make_attn_figure(attn):
    fig, ax = plt.subplots()
    im = ax.imshow(
        attn.T,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    return fig

