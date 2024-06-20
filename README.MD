<img src="assets/Tacotron.png" alt="Tacotron" style="width:100%;">


# Tacotron Implementation

## Table of contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Structure](#model-structure)
- [Training](#training)
- [References](#references)
- [Licence](#license)


## Introduction
This repository contains an implementation of the Tacotron text-to-speech model from scratch. This implementation follows the structure and principles described in the original paper by Wang et al., with some modifications and improvements.


## Features
- End-to-end text-to-speech synthesis
- Mel-spectrogram generation
- Griffin-Lim vocoder for waveform synthesis
- Support for custom datasets


## Installation
Clone the repository, and install the required dependencies:

```bash
git clone
cd tacotron
pip install -r requirements.txt
```

**Ensure you have the necessary libraries installed. The main dependencies are**:
- PyTorch
- Numpy
- Librosa
- Matplotplib

## Usage

**In order to train the model for custom dataset, prepare the data in the following format**

- `data/`: Contains all files
    - `.wav`: all wav files
    - `metadata.csv`: contains all the text transcripts and audio file names

The `metadata.csv` should have the following format:

```bash
filename|transcription
```

#### Preprocessing

Run the preprocessing script to convert audio files into mel-spectrograms:

```bash
python source/prepare_data.py --data-dir data --output_dir processed_data --metadata_file metadata.csv --config-path config.json --num-jobs 4
```

After run the `split` file for splitting the dataset into training and testing datasets:

```bash
python source/split.py --metadata metadata.txt
```

The `metadata.txt` is the file, containing the indexes of all files.

#### Training the model

```bash
python source/train.py 
```

#### Inference the model 

```bash
python source/inference.py
```


**Thanks for attention!**
