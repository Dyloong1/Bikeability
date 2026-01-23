<p align="center">
<h1 align="center"><strong>Persona-Aware Bikeability Assessment & StreetDesignAI</strong></h1>
  <p align="center">
              <a>Yilong Dai<sup>3,*</sup>,</a>
              <a>Ziyi Wang<sup>1,*</sup>,</a>
              <a>Chenguang Wang<sup>2</sup>,</a>
              <a>Duanya Lyu<sup>3</sup>,</a>
              <a>Mateo Nader<sup>3</sup>,</a>
              <a>Sihan Chen<sup>4</sup>,</a>
              <a>Kexin Zhou<sup>3</sup>,</a>
              <a>Yiheng Qian<sup>3</sup>,</a>
              <a>Wanghao Ye<sup>1</sup>,</a>
              <a>Zijian Ding<sup>1</sup>,</a>
              <a>Susu Xu<sup>5</sup>,</a>
              <a>Xiang Yan<sup>3,†</sup></a>
    <br>
    <sup>1</sup>University of Maryland &nbsp;
    <sup>2</sup>Stony Brook University &nbsp;
    <sup>3</sup>University of Florida &nbsp;
    <sup>4</sup>Carnegie Mellon University &nbsp;
    <sup>5</sup>Johns Hopkins University
  </p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.15671" target="_blank">
    <img src="https://img.shields.io/badge/ArXiv-2601.15671-red">
  </a>
  <a href="https://github.com/Dyloong1/Bikeability" target="_blank">
    <img src="https://img.shields.io/badge/Project-Bikeability-blue">
  </a>
  <a href="https://github.com/Dyloong1/Bikeability" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-green">
  </a>
</p>
</p>

This repository hosts two interconnected research projects on **human-centered cycling infrastructure assessment**. The first, **BikeabilityAssessment**, develops a persona-aware Vision-Language Model (VLM) backbone that produces cyclist-type-specific bikeability ratings with interpretable explanations, grounded in the established "Four Types of Cyclists" typology and trained on 12,400 persona-conditioned assessments from 427 real cyclists in Washington DC. The second, **StreetDesignAI**, builds on this VLM backbone to create an interactive multi-persona evaluation system that enables infrastructure designers to receive parallel feedback from simulated cyclist personas, iteratively modify street designs with AI-rendered visualizations, and navigate trade-offs across diverse user needs.

---

## Table of Contents

- [BikeabilityAssessment: VLM Backbone](#-bikeabilityassessment-vlm-backbone)
- [StreetDesignAI: Interactive Design Evaluation](#-streetdesignai-interactive-design-evaluation)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

---

## 🧠 BikeabilityAssessment: VLM Backbone

<p align="center">
  <img src="assets/bikeability_framework.png" alt="BikeabilityAssessment Framework" width="700"/>
</p>

**BikeabilityAssessment** proposes a persona-aware framework that integrates Vision-Language Models with the "Four Types of Cyclists" typology (Strong & Fearless, Enthused & Confident, Interested but Concerned, No Way No How) to produce cyclist-type-specific bikeability assessments. Key contributions:

- **Theory-grounded persona conditioning** based on established cyclist typology, generating persona-specific explanations via chain-of-thought reasoning
- **Multi-granularity supervised fine-tuning** with dynamic task weighting that bridges the gap between abundant structured ratings and coherent reasoning narratives
- **AI-enabled counterfactual image augmentation** via generative image editing to isolate individual infrastructure variable impacts
- **Publicly released dataset** of 12,400 persona-conditioned assessments from 427 cyclists collected through an immersive 360° street-view survey in Washington DC

### Project Structure

```
BikeabilityAssessment/
├── code/
│   ├── train_qwen_vl.py              # Qwen VL model training
│   ├── qwen_vl_predict.py            # Qwen VL prediction
│   ├── qwen_feature_extraction.py    # Feature extraction using Qwen
│   ├── sft_data_preparation.py       # SFT data preparation
│   ├── ablation_training_script.py   # Ablation study training
│   ├── gpt4o_baseline.py             # GPT-4o baseline
│   ├── gpt4o_eval.py                 # GPT-4o evaluation
│   ├── gpt4o_eval_language.py        # GPT-4o language evaluation
│   ├── extract_gpt4o_factors.py      # Factor extraction from GPT-4o
│   └── rf_baseline_training.py       # Random Forest baseline
├── data/
│   ├── bike_lane_info.csv            # Bike lane infrastructure attributes
│   ├── indicator_pool.json           # Evaluation indicator pool
│   ├── locations_info.json           # Survey location metadata
│   ├── ratings_anonymous.json        # Anonymized cyclist ratings
│   └── surveys_anonymous.json        # Anonymized survey responses
```

---

## 🛠️ StreetDesignAI: Interactive Design Evaluation

<p align="center">
  <img src="assets/streetdesignai_overview.png" alt="StreetDesignAI System Overview" width="700"/>
</p>

**StreetDesignAI** is an interactive evaluation system for inclusive cycling infrastructure design. It operationalizes persona-based multi-agent evaluation to make experiential conflicts explicit during the design process. The system enables designers to:

1. **Ground evaluation in real street context** through Street View imagery and OpenStreetMap data
2. **Receive parallel feedback** from simulated cyclist personas spanning confident to cautious users
3. **Iteratively modify designs** with AI-rendered street-level visualizations while the system surfaces conflicts across perspectives

A within-subjects study with 26 transportation professionals demonstrated that structured multi-perspective feedback significantly improves designers' understanding of diverse persona needs, confidence in translating those needs into inclusive design decisions, and overall satisfaction compared to general-purpose AI chatbots.

### Data Structure

```
StreetDesignAI/
├── Interaction Data/          # Per-participant interaction logs (26 participants)
│   ├── 01/ - 26/             # Each folder contains:
│   │   ├── existing-condition.*       # Original street view images
│   │   ├── design-scenario-*.png      # AI-rendered design modifications
│   │   ├── bike-lane-design-*.json    # Design parameter specifications
│   │   └── Screenshot*.png            # Session screenshots
├── Transcript/                # Interview transcripts (26 sessions)
│   ├── 01.txt - 26.txt
├── log data/                  # System interaction logs (JSON)
├── survey data1/              # Survey responses
│   ├── ChatGPT Survey1.numbers
│   ├── Pre-study Survey.numbers
│   └── StreetDesignAI Survey1.numbers
├── Study hot take-pilots.docx
└── Study hot takes-professionals.docx
```

---

## 📖 Citation

If you find this work useful, please cite our papers:

```bibtex
@article{dai2026persona,
  title={Persona-aware and Explainable Bikeability Assessment: A Vision-Language Model Approach},
  author={Dai, Yilong and Wang, Ziyi and Wang, Chenguang and Zhou, Kexin and Qian, Yiheng and Xu, Susu and Yan, Xiang},
  journal={Landscape and Urban Planning},
  year={2026}
}

@article{wang2026streetdesignai,
  title={StreetDesignAI: A Multi-Persona Evaluation System for Inclusive Infrastructure Design},
  author={Wang, Ziyi and Dai, Yilong and Lyu, Duanya and Nader, Mateo and Chen, Sihan and Ye, Wanghao and Ding, Zijian and Yan, Xiang},
  journal={arXiv preprint arXiv:2601.15671},
  year={2026}
}
```

---

## 🙏 Acknowledgments

This work is supported by the **University of Florida** and the **National Science Foundation (NSF)**.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/en/thumb/1/14/University_of_Florida_seal.svg/200px-University_of_Florida_seal.svg.png" alt="University of Florida" height="80"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/National_Science_Foundation_logo.svg/200px-National_Science_Foundation_logo.svg.png" alt="NSF" height="80"/>
</p>
