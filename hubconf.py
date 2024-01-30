dependencies = ['torch', 'torchaudio', 'numpy', 'einops', 'scipy']

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor

import hierspeech_utils
from speechsr48k.speechsr import SynthesizerTrn
from speechsr24k.speechsr import SynthesizerTrn as SpeechSR24
from speechsr48k.speechsr import SynthesizerTrn as SpeechSR48


def speechsr(pretrained=True, progress=True, output_sr=48000, device=None) -> SynthesizerTrn:
    """ Load 16->48kHz (default) or 16->24kHz (if `output_sr=24000` arg given) SpeechSR model. 

    Example usage:
    ```python
    device = torch.device('cuda')
    speechsr = torch.hub.load('RF5/HierSpeechpp', 'speechsr', output_sr=48000, device=device)
    
    # Prompt load
    audio, sr = torchaudio.load(path)

    # support only single channel
    audio = audio.mean(dim=0)[None]
    # Resampling
    if sr != 16000: audio = torchaudio.functional.resample(audio, sr, 16000, resampling_method="kaiser_window") 
    with torch.inference_mode():
        converted_audio = speechsr(audio.unsqueeze(1).to(device))
        converted_audio = converted_audio.squeeze()
        converted_audio = converted_audio / (torch.abs(converted_audio).max()) * 0.999

    torchaudio.save('out.wav', converted_audio.cpu()[None], sample_rate=speechsr.output_sr)
    ```
    """
    assert output_sr in [48000, 24000], f"output_sr argument must be either 48000 or 24000."

    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: device = torch.device(device)

    # h_sr = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt_sr)[0], 'config.json') )
    sr48_path = Path(__file__).parent/'speechsr48k'
    h_sr48 = hierspeech_utils.get_hparams_from_file(str(sr48_path/'config.json'))
    sr24_path = Path(__file__).parent/'speechsr24k'
    h_sr = hierspeech_utils.get_hparams_from_file(str(sr24_path/'config.json'))


    if output_sr == 48000:
        speechsr = SpeechSR48(h_sr48.data.n_mel_channels,
            h_sr48.train.segment_size // h_sr48.data.hop_length,
            **h_sr48.model)
        speechsr.h = h_sr48
        speechsr.output_sr = 48000
        if pretrained: hierspeech_utils.load_checkpoint(str(sr48_path/'G_100000.pth'), speechsr, None)
    else:
        # 24000 Hz
        speechsr = SpeechSR24(h_sr.data.n_mel_channels,
        h_sr.train.segment_size // h_sr.data.hop_length,
        **h_sr.model)
        speechsr.h = h_sr
        speechsr.output_sr = 24000
        if pretrained: hierspeech_utils.load_checkpoint(str(sr24_path/'G_340000.pth'), speechsr, None)
    speechsr.dec.remove_weight_norm()
    speechsr = speechsr.to(device).eval()
    
    print(f"Loaded SpeechSR with {sum(param.numel() for param in speechsr.parameters()):,d} parameters.")
    return speechsr
