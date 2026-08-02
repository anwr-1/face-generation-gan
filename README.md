# 🧑‍🎨 Realistic Human Face Generation using GAN

A DCGAN (with SAGAN-style self-attention, minibatch-stddev, R1 gradient
penalty, and an EMA generator) trained from scratch to generate realistic
synthetic human face images from random noise.

## 🔗 Links

- **Training notebook (Kaggle):** [realistic-human-face-generation-using-gan](https://www.kaggle.com/code/anwernasr/realistic-human-face-generation-using-gan?scriptVersionId=339631921)
- **Trained model checkpoints:** [gan_models_backup.zip](https://www.kaggle.com/code/anwernasr/realistic-human-face-generation-using-gan/output?scriptVersionId=339631921&select=gan_models_backup.zip)

## How it works

- **Generator** — takes a 128-dim random noise vector and upsamples it
  through transposed convolutions (with a self-attention block at 32x32)
  into a 64x64 RGB face.
- **Discriminator** — mirrors the Generator downward, with self-attention,
  dropout, an R1 gradient penalty, and minibatch-stddev to distinguish real
  faces from generated ones.
- Trained in two stages: a Discriminator warm-up on real faces, then full
  adversarial training with instance noise and an EMA copy of the Generator
  for cleaner final samples.
- Quality tracked over time with FID (Fréchet Inception Distance) against a
  held-out set of real faces.

## Generating faces

```bash
pip install -r requirements.txt
```

Download `gan_models_backup.zip` from the link above, unzip it, and pick a
`generator_ema_*.pth` checkpoint (the EMA generator gives the cleanest
samples):

```bash
python generate.py --checkpoint generator_ema_gan_epoch_045.pth --num-images 64 --output faces.png
```

## Project structure

```
face-generation-gan/
├── generate.py       # loads a trained checkpoint and generates a grid of faces
├── requirements.txt
└── README.md
```

## License

MIT
