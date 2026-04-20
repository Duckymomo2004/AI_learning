# Image Processing model

## Project Layout

```text
model/         
trainer/       
dataloader/    
experiments/  
data/          
```

## Environment

Use the `pytorch-ml` conda environment.

```bash
conda activate pytorch-ml
```

## Library Requirements

Main libraries used by the project:

- `torch`
- `torchvision`
- `numpy`
- `Pillow`

Install from the requirements file if needed:

```bash
pip install -r requirements.txt
```

## Data Paths

Classification scripts use:

```text
data/
  images/
    train/
      class_a/
      class_b/
    validation/
      class_a/
      class_b/
```

Few-shot scripts use:

```text
data/
  miniimagenet/
    base.json
    val.json
    train.json
    novel.json
```

## Training

Run each model with:

```bash
python experiments/mlp_mixer/train.py
python experiments/mobilenet_v1/train.py
python experiments/mobilenet_v2/train.py
python experiments/mobilenet_v3/train.py
python experiments/relation_net/train.py
python experiments/few_shot_transformer/train.py
python experiments/ctx/train.py
```

Each script saves its checkpoint as `best.pt` inside its own experiment folder.
