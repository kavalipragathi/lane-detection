# Lane Detection with a Simple CNN (PyTorch)

A minimal, self-contained example of lane detection framed as **binary semantic segmentation**. The notebook generates a synthetic lane dataset, trains a small encoder–decoder CNN in PyTorch, and visualizes the predicted lane mask. No external dataset download is required.

## Overview

| Item | Detail |
|---|---|
| Task | Binary segmentation (lane pixel vs. background) |
| Framework | PyTorch |
| Data | 500 synthetic 128×128 images with matching masks |
| Model | `LaneCNN` – 3-layer conv encoder + 3-layer transposed-conv decoder |
| Loss / Optimizer | `BCELoss` / Adam (lr = 0.001) |
| Training | 15 epochs, batch size 8 |
| Output | `lane_detection_cnn.pt` (saved weights) |

## How it works

1. **Synthetic dataset generation** – Each sample is a black 128×128 image with two white lane lines drawn using OpenCV. The bottom x-coordinates of the lines are randomized (`x1` in 30–50, `x2` in 80–100), while the top points are fixed at (60, 0) and (90, 0). A matching mask is saved alongside each image.
2. **Dataset & DataLoader** – `LaneDataset` loads image/mask pairs, converts BGR → RGB, resizes to 128×128, normalizes masks to `[0, 1]`, and returns tensors of shape `(3, 128, 128)` and `(1, 128, 128)`.
3. **Model** – `LaneCNN` is a small autoencoder-style network:
   - **Encoder:** Conv(3→16) → Conv(16→32) → Conv(32→64), each followed by ReLU and 2×2 max-pooling (128 → 16 spatial size).
   - **Decoder:** Three `ConvTranspose2d` layers upsample back to 128×128 with a final Sigmoid producing per-pixel lane probability.
4. **Training** – Standard loop with BCE loss and Adam. In the recorded run, total epoch loss drops from ~28.3 (epoch 1) to ~0.06 (epoch 15).
5. **Inference** – The model is switched to eval mode, a sample image is passed through, and the output is thresholded at 0.5 to produce a binary lane mask, displayed next to the input with Matplotlib.

## Project structure

```
.
├── lane_detection.ipynb      # Main notebook
├── dataset/                  # Generated when the notebook runs
│   ├── images/train/         # *.jpg synthetic images
│   └── masks/train/          # *.png ground-truth masks
└── lane_detection_cnn.pt     # Saved model weights (after training)
```

## Requirements

- Python 3.8+
- `torch`, `torchvision`
- `opencv-python`
- `numpy`
- `matplotlib`
- `jupyter` (or any notebook environment)

```bash
pip install torch torchvision opencv-python numpy matplotlib jupyter
```

A GPU is optional. The notebook auto-selects CUDA if available and falls back to CPU (the recorded run used CPU).

## Usage

```bash
jupyter notebook lane_detection.ipynb
```

Then run all cells in order. The notebook will generate the dataset, train the model, save the weights, and display a prediction.

### Loading the saved model

```python
import torch
model = LaneCNN()  # class defined in the notebook
model.load_state_dict(torch.load("lane_detection_cnn.pt", map_location="cpu"))
model.eval()
```

## Limitations

- **Synthetic, toy data:** The images are clean white lines on a black background, so results will not transfer to real road footage.
- **No validation/test split:** Training and the demo inference both use the training set, so the results show the pipeline works, not that it generalizes.
- **No quantitative metrics:** Only training loss is reported; there is no IoU or Dice score.
- **Small model:** No skip connections (as in U-Net), so fine detail is limited.

## Ideas for improvement

- Train on a real dataset such as [TuSimple](https://github.com/TuSimple/tusimple-benchmark) or [CULane](https://xingangpan.github.io/projects/CULane.html).
- Add a validation split and report IoU / Dice.
- Add data augmentation (noise, brightness, blur, curved lanes, varying backgrounds).
- Upgrade to a U-Net or another architecture with skip connections.
- Overlay the predicted mask on the input image and try video inference.

## License

Add your preferred license here (e.g., MIT).
