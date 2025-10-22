import os
import torch
import numpy as np
from urllib.request import urlretrieve

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def download_rife_model():
    """Download RIFE model if not present"""
    model_path = "rife47.pth"
    if not os.path.exists(model_path):
        url = "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-v4.7.pkl"
        urlretrieve(url, model_path)
    return model_path


class RIFEModel:
    def __init__(self):
        self.device = DEVICE
        self.model = None

    def load_model(self):
        if self.model is None:
            model_path = download_rife_model()
            self.model = torch.jit.load(model_path).eval().to(self.device)

    def interpolate(self, img1, img2):
        """Interpolate between two frames"""
        self.load_model()

        # Convert numpy arrays to tensors
        img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Normalize to -1 to 1
        img1 = (img1 * 2 - 1).float()
        img2 = (img2 * 2 - 1).float()

        # Run inference
        with torch.no_grad():
            middle = self.model(img1, img2)[0]

        # Convert back to numpy
        middle = ((middle + 1) / 2).clamp(0, 1)
        middle = middle.squeeze(0).permute(1, 2, 0).cpu().numpy()

        return middle


def interpolate_sequence(frames, target_frames, loop_seamless=False):
    """Interpolate a sequence of frames to target length using RIFE"""
    rife = RIFEModel()
    output = []

    # Calculate positions for target frames
    if loop_seamless:
        # For looping, use modulo to wrap around
        target_positions = np.linspace(0, len(frames), target_frames + 1)[:-1]
    else:
        target_positions = np.linspace(0, len(frames) - 1, target_frames)

    for pos in target_positions:
        idx_low = int(np.floor(pos)) % len(frames)  # Use modulo for looping
        idx_high = (idx_low + 1) % len(frames)  # Wrap around to first frame
        fraction = pos - int(pos)

        if idx_low == idx_high:
            output.append(frames[idx_low])
        else:
            interpolated = rife.interpolate(frames[idx_low], frames[idx_high])
            if fraction <= 0.5:
                output.append(frames[idx_low])
            else:
                output.append(interpolated)

    return np.stack(output)
