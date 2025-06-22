# diffusion-mario

## Steup 

```bash
conda create -n mario_diffusion python=3.9 -y
conda activate mario_diffusion
pip install torch torchvision
pip install gym-super-mario-bros nes-py opencv-python tqdm numpy
```

## Run

```bash
python dataset_gen.py
python train_diffusion.py
```
