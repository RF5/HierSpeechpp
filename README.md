# HierSpeech2: Hierarhical Variational Autoencoder is a Strong Zero-shot Speech Synthesizer 
The official implementation of HierSpeech2 | [Paper]() | [Demo page]()

**Sang-Hoon Lee, Ha-Yeong Choi, Seong-Whan Lee**

 Department of Artificial Intelligence, Korea University, Seoul, Korea  

## Abstract


## Previous Works
- [1] HierSpeech: Bridging the Gap between Text and Speech by Hierarchical Variational Inference using Self-supervised Representations for Speech Synthesis
- [2] HierVST: Hierarchical Adaptive Zero-shot Voice Style Transfer

This paper is an extenstion version of above papers.

## Todo
### Speech Backbone
- [ ] HierSpeech2 (Efficient but Strong Zero-shot Speech Synthesizier)
- [ ] HierSpeech2-Large (For Much More Strong Zero-shot Speech Synthesizer)
- [ ] HierSpeech2-Lite (Fast and Efficient Zero-shot Speech Synthesizer)
<!--
- [ ] HierSpeech2-24k-Large-Full (For High-resolutional and High-quality Speech Synthesizer)
- [ ] HierSpeech2-48k-Large-Full (For Industrial-level High-resolution and High-quality Speech Synthesizer)
-->
### Text-to-Linguistic Model
- [ ] Text-to-Vec (For Text-to-Speech)
- [ ] Hierarchical Text-to-Vec (For Much More Expressive Text-to-Speech)

### Audio Super-resolution Model (16k --> 24k or 48k) 
- [ ] SpeechSR-24k
- [ ] SpeechSR-48k

## Getting Started

### Pre-requisites

## Checkpoint
### Speech Backbone
| Model |Sampling Rate|Params|Dataset |Hour|Speaker|Checkpoint|
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| HierSpeech2 |16 kHz|90M| LibriTTS (train-460) |245|1,151|-|
| HierSpeech2 |16 kHz|90M| LibriTTS (train-960) |555.08|2,311|-|
| HierSpeech2-Full|16 kHz|90M| Libri-light, AiHub-Korean) |20k| - |-|

<!--
| HierSpeech2-Lite|16 kHz|-| LibriTTS (train-960))  |-|
| HierSpeech2-Lite|16 kHz|-| LibriTTS (train-960) NIKL, AudioBook-Korean)  |-|
| HierSpeech2-Large-CL|16 kHz|200M| LibriTTS (train-960), Libri-Light, NIKL, AudioBook-Korean, Japanese, Chinese, CSS, MLS)  |-|
| HierSpeech2-Large-Full|24 kHz|200M| Not Available |Not Available|
| HierSpeech2-Large-Full|48 kHz|200M| Not Available |Not Available|
-->
### TTV
| Model |Language|Params|Dataset|Hour|Speaker|Checkpoint|
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| TTV |English|50M| LibriTTS (train-960) |555.08|2,311|-|
| TTV-Lite |English|10M| LibriTTS (train-960) |555.08|2,311|-|

### SpeechSR
| Model |Sampling Rate|Params|Dataset |Checkpoint|
|------|:---:|:---:|:---:|:---:|
| SpeechSR-24k |16kHz --> 24 kHz|0.03M| LibriTTS (train-960), AudioBook-Korean |-|
| SpeechSR-48k |16kHz --> 48 kHz|0.05M| AudioBook-Korean, AudioSet |-|

## Voice Conversion
- Todo
- 
## Text-to-Speech
- Todo
- 
## F0 Extraction
- Todo
- 
## Speech Super-resolution
- Todo
- 

## GAN VS Diffusion

We think that we could not confirm which is better yet. There are many advatanges for each model so you can utilize each model for your own purposes and each study must be actively conducted simultaneously.  

### GAN (Specifically, GAN-based End-to-End Speech Synthesis Models)
- (pros) Fast Inference Speed 
- (pros) High-quality Audio
- (cons) Slow Training Speed (Over 7~20 Days)
- (cons) Lower Voice Style Transfer Performance than Diffusion Models
- (cons) Perceptually High-quality but Over-smoothed Audio because of Information Bottleneck by the sampling from the low-dimensional Latent Variable
   
### Diffusion (Diffusion-based Mel-spectrogram Generation Models)
- (pros) Fast Training Speed (within 3 Days)
- (pros) High-quality Voice Style Transfer
- (cons) Slow Inference Speed
- (cons) Lower Audio quality than End-to-End Speech Synthesis Models 

### (In this wors) Our Approaches for GAN-based End-to-End Speech Synthesis Models 
- Improving Voice Style Transfer Performance in End-to-End Speech Synthesis Models for OOD (Zero-shot Voice Style Transfer for Novel Speaker)
- Improving the Audio Quality beyond Perceptal Quality for Much more High-fidelity Audio Generation

### (Our other works) Diffusion-based Mel-spectrogram Generation Models
- DDDM-VC: Disentangled Denoising Diffusion Models for High-quality and High-diversity Speech Synthesis Models
- Diff-hierVC: Hierarhical Diffusion-based Speech Synthesis Model with Diffusion-based Pitch Modeling
- SDT: Efficient Speech Diffusion Transformer Models

### Our Goals
- Integrating each model for High-quality, High-diversity and High-fidelity Speech Synthesis Models 

## Reference
### Our Previous Works
- HierSpeech/HierSpeech-U for Hierarchical Speech Synthesis Framework: https://openreview.net/forum?id=awdyRVnfQKX
- HierVST for Baseline Speech Backbone: https://www.isca-speech.org/archive/interspeech_2023/lee23i_interspeech.html
- DDDM-VC: https://dddm-vc.github.io/
- Diff-HierVC: https://diff-hiervc.github.io/
- SDT: https://sdt.github.io/

  
### Baseline Model
- VITS: https://github.com/jaywalnut310/vits
- Period VITS: https://ieeexplore.ieee.org/abstract/document/10096480
- NaturalSpeech: 
- NANSY for Audio Perturbation: https://github.com/revsic/torch-nansy
- Speech Resynthesis: https://github.com/facebookresearch/speech-resynthesis
  
### Waveform Generator for High-quality Audio Generation
- HiFi-GAN: https://github.com/jik876/hifi-gan 
- BigVGAN for High-quality Generator: https://arxiv.org/abs/2206.04658
- UnivNET: https://github.com/mindslab-ai/univnet
- EnCodec: https://github.com/facebookresearch/encodec

### Self-supervised Speech Model 
- Wav2Vec 2.0: https://arxiv.org/abs/2006.11477
- XLS-R: https://huggingface.co/facebook/wav2vec2-xls-r-300m
- MMS: https://huggingface.co/facebook/facebook/mms-300m

### Other Large Language Model based Speech Synthesis Model
- VALL-E & VALL-E-X:
- SPEAR-TTS:
- NaturalSpeech 2: 
- Make-a-Voice:
- MEGA-TTS & MEGA-TTS 2:

Although there are some recent works utilizing hierarchical speech synthesis frameworks using self-supervised speech representation, it is worth noting that HierSpeech [NeurIPS2022] is the first hierarchical speech synthesis models using self-supervised speech representation.
