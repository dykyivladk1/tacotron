import argparse
import yaml
import os
from symbols import txt2seq
from model import Tacotron
from utilities import AudioProcessor
import torch
import numpy as np

n_vocab = 250
embedding_size = 256
mel_size = 80
linear_size = 1025
r = 5

def generate_speech(args):
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    model, step = load_checkpoint(config, args.checkpoint_path)
    if model is None:
        return
    seq = np.asarray(txt2seq(args.text))
    seq = torch.from_numpy(seq).unsqueeze(0)
    
    with torch.no_grad():
        mel, spec, attn = model(seq)
    
    ap = AudioProcessor(**config['audio'])
    wav = ap.inv_spectrogram(spec[0].numpy().T)
    ap.save_wav(wav, args.output)

def load_checkpoint(config, checkpoint_path):
    model = Tacotron(**config['model']['tacotron'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["global_step"]
        print("@ step {} => loaded checkpoint: {}".format(step, checkpoint_path))
        model.encoder.eval()
        model.postnet.eval()
        return model, step
    else:
        print("Checkpoint file not found at: {}".format(checkpoint_path))
        return None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthesize speech')
    parser.add_argument('--text', type=str, default='Welcome to national taiwan university speech lab.', help='Text to synthesize')
    parser.add_argument('--output', type=str, default='samples/output.wav', help='Output path')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()
    generate_speech(args)
