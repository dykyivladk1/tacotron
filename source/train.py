import torch
import torch.nn as nn
import yaml
from model import Tacotron
from utilities import createDataLoader
import math


# my personal library for AI developers
# download it by: 'pip install polip'
from polip import decider


with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

audio_config = config["audio"]
model_config = config["model"]
solver_config = config["solver"]

sample_rate = audio_config["sample_rate"]

n_vocab = model_config["tacotron"]["n_vocab"]
embedding_size = model_config["tacotron"]["embedding_size"]
mel_size = model_config["tacotron"]["mel_size"]
linear_size = model_config["tacotron"]["linear_size"]
r = model_config["tacotron"]["r"]
lr = model_config["optimizer"]["lr"]

max_step = solver_config["total_steps"]
batch_size = solver_config["batch_size"]
data_dir = solver_config["data_dir"]
n_jobs = solver_config["n_jobs"]
EPOCHS = solver_config["epochs"]

from polip import printer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

data_loader = createDataLoader(
    mode="train",
    meta_path="training_data/meta_train.txt",
    data_dir="training_data",
    batch_size=batch_size,
    r=r,
    n_jobs=n_jobs,
    use_gpu=torch.cuda.is_available()
)

test_loader = createDataLoader(
    mode='test',
    meta_path='training_data/meta_test.txt',
    data_dir='training_data',
    batch_size=batch_size,
    r=r,
    n_jobs=n_jobs,
    use_gpu=torch.cuda.is_available()
)


model = Tacotron(
    n_vocab=n_vocab,
    embedding_size=embedding_size,
    mel_size=mel_size,
    linear_size=linear_size,
    r=r
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
criterion = nn.L1Loss()

def verbose(msg):
    print(' ' * 100, end='\r')
    print("[INFO]", msg)

def train(model, data_tr, criterion, optim, scheduler, device, max_epoch, verbose_fn):
    step = 0
    fs = sample_rate
    linear_dim = model.linear_size
    n_priority_freq = int(3000 / (fs * 0.5) * linear_dim)
    model.train()
    verbose_fn('Start training: {} batches per epoch'.format(len(data_tr)))

    for epoch in range(max_epoch):
        verbose_fn(f'Epoch: {epoch + 1}/{max_epoch}')
        for curr_b, (txt, text_lengths, mel, spec) in enumerate(data_tr):
            sorted_lengths, indices = torch.sort(text_lengths.view(-1), dim=0, descending=True)
            sorted_lengths = sorted_lengths.long().cpu().numpy()
            txt, mel, spec = txt[indices], mel[indices], spec[indices]

            txt = txt.to(device)
            mel = mel.to(device)
            spec = spec.to(device)

            optim.zero_grad()
            mel_outputs, linear_outputs, attn = model(txt, mel, text_lengths=sorted_lengths)

            mel_loss = criterion(mel_outputs, mel)
            linear_loss = 0.5 * criterion(linear_outputs, spec) + \
                          0.5 * criterion(linear_outputs[:, :, :n_priority_freq], spec[:, :, :n_priority_freq])

            loss = mel_loss + linear_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if math.isnan(grad_norm):
                verbose_fn('Error: grad norm is NaN @ step ' + str(step))
            else:
                optim.step()
                scheduler.step()

            if step % 5 == 0:
                verbose_fn(f'Step: {step}, Loss: {loss.item()}, Mel Loss: {mel_loss.item()}, Linear Loss: {linear_loss.item()}, Grad Norm: {grad_norm}')
            step += 1

    verbose_fn(f'Training completed for {max_epoch} epochs.')


def evaluate(model, data_loader, criterion, device, verbose_fn):
    
    model.eval()
    total_loss = 0
    total_mel_loss = 0
    total_linear_loss = 0
    total_samples = 0
    fs = sample_rate
    linear_dim = model.linear_size
    n_priority_freq = int(3000 / (fs * 0.5) * linear_dim)

    with torch.no_grad():
        for curr_b, (txt, text_lengths, mel, spec) in enumerate(data_loader):
            txt = txt.to(device)
            mel = mel.to(device)
            spec = spec.to(device)
            text_lengths = text_lengths.to("cpu")  

            mel_outputs, linear_outputs, attn = model(txt, mel, text_lengths=text_lengths)

            mel_loss = criterion(mel_outputs, mel)
            linear_loss = 0.5 * criterion(linear_outputs, spec) + \
                          0.5 * criterion(linear_outputs[:, :, :n_priority_freq], spec[:, :, :n_priority_freq])

            loss = mel_loss + linear_loss

            total_loss += loss.item() * txt.size(0)
            total_mel_loss += mel_loss.item() * txt.size(0)
            total_linear_loss += linear_loss.item() * txt.size(0)
            total_samples += txt.size(0)

    avg_loss = total_loss / total_samples
    avg_mel_loss = total_mel_loss / total_samples
    avg_linear_loss = total_linear_loss / total_samples

    verbose_fn(f'Evaluation - Average Loss: {avg_loss:.4f}, Average Mel Loss: {avg_mel_loss:.4f}, Average Linear Loss: {avg_linear_loss:.4f}')

    model.train()  
    return avg_loss, avg_mel_loss, avg_linear_loss

if __name__ == "__main__":
    epochs = EPOCHS
    for epoch in range(epochs):
        verbose(f'Starting epoch {epoch + 1}/{epochs}')
        
        train(model = model,
              data_tr = data_loader,
              criterion = criterion,
              optim = optimizer,
              scheduler = scheduler,
              device = decider(),
              max_epoch = epochs,
              verbose_fn = verbose
              )
        evaluate(
            model=model,
            data_loader=data_loader,
            criterion=criterion,
            device=device,
            verbose_fn=verbose
        )
