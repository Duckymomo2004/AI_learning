# PyTorch Vision Notebook Refactor

This repository reorganizes the original Jupyter notebooks into reusable PyTorch modules. Model code lives in `model/`, training loops live in `trainer/`, dataset code lives in `dataloader/`, and each model family has its own runnable script in `experiments/`.

## Requirements

- Python
- `torch`
- `torchvision`
- `numpy`
- `Pillow`

Install the libraries with:

```bash
pip install -r requirements.txt
```

If the `pytorch-ml` conda environment already exists, use it:

```bash
conda activate pytorch-ml
```

## Folder Structure

```text
.
├── model/                         # Model definitions extracted from the notebooks
│   ├── cbam_resnet.py
│   ├── cosine_transformer.py
│   ├── fewshot_backbones.py
│   ├── mlp_mixer.py
│   ├── mobilenet.py
│   └── relation_net.py
├── trainer/                       # Shared training and evaluation loops
├── dataloader/                    # CIFAR-10, ImageFolder, and few-shot data loading
├── experiments/                   # One training entrypoint per model family
│   ├── cbam_resnet/
│   ├── ctx/
│   ├── few_shot_transformer/
│   ├── mlp_mixer/
│   ├── mobilenet_v1/
│   ├── mobilenet_v2/
│   ├── mobilenet_v3/
│   └── relation_net/
├── data/                          # Downloaded datasets and local metadata
├── cbam-resnet.ipynb              # Original CBAM-ResNet notebook
├── cosine-transformer.ipynb       # Original few-shot transformer notebook
├── mlpmixer.ipynb                 # Original MLP-Mixer notebook
├── mobile-nets-v1-to-v3.ipynb     # Original MobileNet notebook
├── relation-net.ipynb             # Original RelationNet notebook
├── requirements.txt
├── README.md
└── .gitignore
```

## How To Run

1. Activate the environment.

```bash
conda activate pytorch-ml
```

2. Install the libraries if the environment does not already include them.

```bash
pip install -r requirements.txt
```

3. Put the data in `data/`.

- `data/images/train` and `data/images/validation` are used by the MobileNet scripts.
- `data/miniimagenet/base.json` and `data/miniimagenet/val.json` are used by the few-shot scripts.
- CIFAR-10 is downloaded under `data/` by the MLP-Mixer and CBAM-ResNet scripts.

4. Run the training script you want.

```bash
python experiments/mlp_mixer/train.py
python experiments/cbam_resnet/train.py
python experiments/mobilenet_v1/train.py
python experiments/mobilenet_v2/train.py
python experiments/mobilenet_v3/train.py
python experiments/relation_net/train.py
python experiments/few_shot_transformer/train.py
python experiments/ctx/train.py
```

Each script saves its checkpoint as `best.pt` inside its own experiment folder.

## Notebook Training And Testing Summary

The table below summarizes the saved outputs found in the original notebooks. These numbers come from the notebook files themselves.

| Notebook | Model | Data | Saved result |
| --- | --- | --- | --- |
| `cbam-resnet.ipynb` | `ResNet50 + CBAM` | `CIFAR-10` | Best test accuracy `0.8492`, best test loss `0.4921`, trained for `20` epochs |
| `mlpmixer.ipynb` | `MLPMixer` | `CIFAR-10` | Best test accuracy `0.8237`, best test loss `0.9572`, trained for `100` epochs |
| `mobile-nets-v1-to-v3.ipynb` | `MobileNetV1` | `ImageFolder` face-expression data | Final epoch: train acc `0.9515`, test acc `0.5745`, train loss `0.1465`, test loss `2.8234` |
| `mobile-nets-v1-to-v3.ipynb` | `MobileNetV2` | `ImageFolder` face-expression data | Final epoch: train acc `0.8483`, test acc `0.5730`, train loss `0.4270`, test loss `1.5417` |
| `mobile-nets-v1-to-v3.ipynb` | `MobileNetV3` | `ImageFolder` face-expression data | Final epoch: train acc `0.8354`, test acc `0.5553`, train loss `0.4689`, test loss `1.5849` |
| `relation-net.ipynb` | `RelationNet` | `MiniImageNet` episodic data (`5-way`, `1-shot`, `15` query, `84x84`) | Best test accuracy `0.5104`, best test loss `52.5273`, trained for `50` epochs |
| `cosine-transformer.ipynb` | `FewShotTransformer` | `MiniImageNet` episodic JSON splits | Model and data-preparation code are present, but the saved notebook output does not include final training or testing logs |
| `cosine-transformer.ipynb` | `CTX` | `MiniImageNet` episodic JSON splits | Model and data-preparation code are present, but the saved notebook output does not include final training or testing logs |
