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
* [2023-11-29]: ⭐️ Paper of VCD online. Check out [this link]() for details.
* [2023-11-28]: 🚀🚀 Codes released.

## 🎯 Overview
![VCD](figs/figure1.png)
- We introduce Visual Contrastive Decoding (VCD), **a simple and training-free** method that contrasts output distributions derived from original and distorted visual inputs.
- The new **contrastive probability distribution** for decoding is formulated as follows:
```math
p_{vcd}(y \mid v, v', x) = softmax[ (1+\alpha)\times logit_\theta (y \mid v, x) - \alpha \times logit_\theta(y \mid v', x)],
```
- The proposed VCD effectively reduces the over-reliance on **statistical bias** and **unimodal priors**, two essential causes of object hallucinations.


## 🕹️ Usage
### Environment Setup
```bash
conda create -yn vcd python=3.9
conda activate vcd

https://github.com/DAMO-NLP-SG/VCD.git
cd VCD
pip install -r requirements.txt
```

### How to Use VCD in LVLMs


## 🏅 Experiments
- **VCD significantly mitigates the object hallucination issue across different LVLM families.**
![exp1](figs/exp1.png)
*table 1(Part of). Results on POPE. Regular decoding denotes direct sampling, whereas VCD refers to sampling from our proposed contrastive distribution pvcd. The best performances within each setting are bolded.*

- **Beyond mitigating object hallucinations, VCD also excels in general LVLM benchmarks, highlighting its wide-ranging applicability.**
![exp2](figs/exp2.png)
*figure 4. MME full set results on LLaVA-1.5. VCD consistently enhances LVLMs’ perception capacities while preserving their recognition competencies.*
<p align="center" width="80%">
<a target="_blank"><img src="figs/exp3.png" alt="GPT4V aided evaluation" style="width: 50%; min-width: 200px; display: block; margin: auto;"></a>
</p>

*table 3. Results of GPT-4V-aided evaluation on open-ended generation. Accuracy measures the response’s alignment with the image content, and Detailedness gauges the richness of details in the response. Both metrics are on a scale of 10.*

- **Please refer to [our paper]() for detailed experimental results.**



## 📌 Examples
![Case1](figs/case.jpg)
*figure 5. Illustration of hallucination correction by our proposed VCD with two samples from LLaVA-Bench. Hallucinated objects from LVLM's regular decoding are highlighted in red.*

![Case2](figs/case_general.jpg)
*figure 8. More examples from LLaVA-Bench of our proposed VCD for enhanced general perception and recognition capacities.*

![Case3](figs/case_hallu.jpg)
*figure 7. More examples from LLaVA-Bench of our proposed VCD for hallucination corrections. Hallucinated objects from LVLM's regular decoding are highlighted in red.*


## 📑 Citation
If you find our project useful, we hope you can star our repo and cite our paper as follows:
```
@article{damonlpsg2023vcd,
  author = {Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, Lidong Bing},
  title = {Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding},
  year = 2023,
  journal = {arXiv preprint arXiv:xxx},
  url = {https://arxiv.org/abs/xxxx}
}
```

## 📝 Related Projects
- [Contrastive Decoding: Open-ended Text Generation as Optimization](https://github.com/XiangLi1999/ContrastiveDecoding)
