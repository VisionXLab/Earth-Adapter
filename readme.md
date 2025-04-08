# Earth Adapter: Bridge the Geospatial Domain Gaps with Mixture of Frequency Adaptation

[![Paper](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official PyTorch implementation of [Earth Adapter: Bridge the Geospatial Domain Gaps with Mixture of Frequency Adaptation]
## TODO
- [ ] complete training and evaluation instruction
- [ ] paper link
- [ ] demo.ipynb
- [ ] ...

## 📖 Introduction

This repository contains the official implementation of [Earth Adapter: Bridge the Geospatial Domain Gaps with Mixture of Frequency Adaptation]. Our method achieves state-of-the-art performance on 8 widely-used cross-domain geospatial datasets. The code is still under development, and we are currently providing the model, weights, and dataset. The complete training and testing pipelines will be fully released within a week.

Paper: [Paper Link](https://arxiv.org/abs/XXXX.XXXXX)

## 🛠️ Requirements

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0 (if using GPU)
- Other dependencies in `requirements.txt`

## 🚀 Installation

- Clone this repository and install dependencies:

```bash
# Clone the repo
git clone https://github.com/xiaoxing2001/Earth-Adapter.git
cd Earth-Adapter

# Create virtual environment
conda create -n earth-adapter python=3.9 -y

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
pip install "mmdet>=3.0.0"
pip install xformers=='0.0.20'
pip install -r requirements.txt
pip install future tensorboard
```

## 📂 Dataset Preparation


- Download the LoveDA, ISPRS Potsdam, ISPRS Vaihingen at the [Baidu Cloud](https://pan.baidu.com/s/1WGoVqLuJTJXc2AVDyBxXYQ?pwd=s6rk)
- Construct the data as follows:

```bash
Earth-Adapter/
|-- data/
|---|--- loveda_uda
|---|--- potsdamRGB
|---|--- vaihingen
```

## 🔥 Usage

### Training

```bash
./tools/train.sh
```

### Evaluation
The Checkpoint can be downloaded from [Baidu Cloud](https://pan.baidu.com/s/1vZm9VvSgRmPeXfu-21nudA?pwd=ys74),put the checkpoint in the `checkpoints` folder. Then run:
```bash
./tools/test.sh
```
<!-- 
To evaluate the trained model, run:

```bash
python eval.py --checkpoint path/to/checkpoint.pth
```

### Demo

To run inference on a single image:

```bash
python demo.py --input path/to/image.jpg --output path/to/output.jpg
```

## 📊 Results

### Quantitative Results

| Method        | Dataset | Accuracy | mIoU |
|--------------|--------|----------|------|
| Our Method   | XYZ    | XX.X%    | XX.X% |
| Baseline     | XYZ    | XX.X%    | XX.X% |

### Qualitative Results

Example predictions:

![Sample Result](assets/sample_result.png)

## 📜 Citation

If you find our work helpful, please cite our paper:

```bibtex
@article{yourpaper2025,
  title={Your Paper Title},
  author={Author1 and Author2 and Others},
  journal={Conference/Journal},
  year={2025},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgments

We thank [Project/Library Name] for their contributions to our work. This work was supported by [Funding Source]. -->
