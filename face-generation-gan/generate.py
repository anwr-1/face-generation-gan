"""
Generate synthetic human faces from a trained GAN generator checkpoint.

Usage:
    python generate.py --checkpoint generator_ema_gan_epoch_045.pth --num-images 64
"""
import argparse
import torch
import torch.nn as nn
import torchvision.utils as vutils

# Must exactly match the architecture used during training (see the training
# notebook's Section 3 / Model Architecture).
LATENT_DIM = 128
GEN_FEAT = 96
CHANNELS = 3
IMG_SIZE = 64          # set to 128 if you trained with img_size=128 instead
USE_ATTENTION = True


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, max(channels // 8, 1), 1)
        self.key = nn.Conv2d(channels, max(channels // 8, 1), 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, feat=GEN_FEAT, channels=CHANNELS,
                 img_size=IMG_SIZE, use_attention=USE_ATTENTION):
        super().__init__()
        assert img_size in (64, 128), "This architecture supports img_size 64 or 128."

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feat * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feat * 8), nn.ReLU(True),
            nn.ConvTranspose2d(feat * 8, feat * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat * 4), nn.ReLU(True),
            nn.ConvTranspose2d(feat * 4, feat * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat * 2), nn.ReLU(True),
            nn.ConvTranspose2d(feat * 2, feat, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat), nn.ReLU(True),
        )
        self.attention = SelfAttention(feat) if use_attention else nn.Identity()

        if img_size == 64:
            self.block2 = nn.Sequential(
                nn.ConvTranspose2d(feat, channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        else:  # 128
            self.block2 = nn.Sequential(
                nn.ConvTranspose2d(feat, feat // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feat // 2), nn.ReLU(True),
                nn.ConvTranspose2d(feat // 2, channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)
        x = self.block1(z)
        x = self.attention(x)
        return self.block2(x)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic faces from a trained GAN generator.")
    parser.add_argument("--checkpoint", default="generator_ema.pth",
                         help="Path to a generator_ema_*.pth checkpoint from training.")
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--output", default="generated_faces.png")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator().to(device)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=device))
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(args.num_images, LATENT_DIM, device=device)
        faces = generator(noise).detach().cpu()

    grid = vutils.make_grid(faces, nrow=8, normalize=True, padding=2)
    vutils.save_image(grid, args.output)
    print(f"Saved a grid of {args.num_images} generated faces to {args.output}")


if __name__ == "__main__":
    main()
