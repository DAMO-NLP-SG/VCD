<p align="center" width="100%">
<a target="_blank"><img src="figs/VCD_logo.png" alt="Visual Contrastive Decoding" style="width: 50%; min-width: 200px; display: block; margin: auto;"></a>
</p>

# VCD: Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding
<!-- **VCD: Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding** -->
This is the official repo for Visual Contrastive Decoding, a simple, training-free method for mitigating hallucinations in LVLMs during decoding.

<div style='display:flex; gap: 0.25rem; '>
<a href='LICENCE'><img src='https://img.shields.io/badge/License-MIT-g.svg'></a>
<a href='https://arxiv.org/abs/2306.02858'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
</div>

## Overview
![VCD](figs/figure1.png)

We introduce Visual Contrastive Decoding (VCD) to mitigate hallucinations in Large Vision-Language Models. This simple, training-free method exploits the difference between output distributions derived from original and distorted visual inputs for more contextually aligned generation.


## How to use VCD


## Experiment

## Case Study
![Case](figs/case.jpg)
A case study showing how VCD can mitigate object hallucinations during LVLMs' decoding process.

## Citation
If you find our project useful, we hope you can star our repo and cite our paper as follows:
```
@article{damonlpsgvcd,
  author = {Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, Lidong Bing},
  title = {Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding},
  year = 2023,
}
```
