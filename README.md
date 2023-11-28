<p align="center" width="100%">
<a target="_blank"><img src="figs/VCD_logo_title.png" alt="Visual Contrastive Decoding" style="width: 75%; min-width: 200px; display: block; margin: auto;"></a>
</p>

# VCD: Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding
<!-- **VCD: Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding** -->
This is the official repo for Visual Contrastive Decoding, a simple, training-free method for mitigating hallucinations in LVLMs during decoding without utilizing external tools.

<div style='display:flex; gap: 0.25rem; '>
<a href='LICENCE'><img src='https://img.shields.io/badge/License-MIT-g.svg'></a>
<a href='https://arxiv.org/abs/2306.02858'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
</div>

## 🔥 Update
**[2023-11-29]: Paper submitted to Arxiv. Check out [VCD]() for details.**

**[2023-11-28]: Codes released.**

## 🎯 Overview
We introduce Visual Contrastive Decoding (VCD), **a simple and training-free** method that contrasts output distributions derived from original and distorted visual inputs.
The proposed VCD effectively reduces the over-reliance on **statistical bias** and **unimodal priors**, two essential causes of object hallucinations.
This adjustment ensures the generated content is closely grounded to visual inputs, resulting in **contextually accurate outputs**.

![VCD](figs/figure1.png)

Specifically, given a textual query ${x}$ and a visual input ${v}$, the model generates two distinct output distributions: one conditioned on the original ${v}$ and the other on the distorted visual input ${v'}$, which is derived by applying pre-defined distortions (i.e., Gaussian noise mask) to ${v}$. 
Then, a new contrastive probability distribution is computed by exploiting the differences between the two initially obtained distributions. 
The new contrastive distribution $p_{vcd}$ is formulated as:
```math
p_{vcd}(y \mid v, v', x) = softmax[ (1+\alpha)\times logit_\theta (y \mid v, x) - \alpha \times logit_\theta(y \mid v', x)],
```
where larger $\alpha$ values indicate a stronger amplification of differences between the two distributions ($\alpha=0$ reduces to regular decoding). 
From the adjusted output distribution $p_{vcd}$, we can apply various sampling strategies, such as nucleus sampling and beam search.


## 🕹️ How to use VCD


## 🏅 Experiments
Our experiments show that VCD, without either additional training or the usage of external tools, significantly mitigates the object hallucination issue across different LVLM families. 
Beyond mitigating object hallucinations, VCD also excels in general LVLM benchmarks, highlighting its wide-ranging applicability. Please refer to our paper for detailed experimental results.

![exp1](figs/exp1.png)
Results on POPE. Regular decoding denotes direct sampling, whereas VCD refers to sampling from our proposed contrastive distribution pvcd. The best performances within each setting are bolded.

![exp2](figs/exp2.png)
MME full set results on LLaVA-1.5. VCD consistently enhances LVLMs’ perception capacities while preserving their recognition competencies.

<img src="figs/exp3.png" width="500" height="250">

Results of GPT-4V-aided evaluation on open-ended generation. Accuracy measures the response’s alignment with the image content, and Detailedness gauges the richness of details in the response. Both metrics are on a scale of 10.



## 📌 Case Study
![Case1](figs/case.jpg)
Illustration of hallucination correction by our proposed VCD with two samples from LLaVA-Bench. Hallucinated objects from LVLM's regular decoding are highlighted in red.

![Case2](figs/case_general.jpg)
More examples from LLaVA-Bench of our proposed VCD for enhanced general perception and recognition capacities.

![Case3](figs/case_hallu.jpg)
More examples from LLaVA-Bench of our proposed VCD for hallucination corrections. Hallucinated objects from LVLM's regular decoding are highlighted in red.


## 📑 Citation
If you find our project useful, we hope you can star our repo and cite our paper as follows:
```
@article{damonlpsgvcd,
  author = {Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, Lidong Bing},
  title = {Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding},
  year = 2023,
}
```

## 📝 Related Projects
- [Contrastive Decoding: Open-ended Text Generation as Optimization](https://github.com/XiangLi1999/ContrastiveDecoding)
