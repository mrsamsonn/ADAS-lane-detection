
### utils/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import numpy as np
import torch

class LaneDataset(Dataset):
    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx].transpose(2, 0, 1)  # channel-first
        y = self.labels[idx].transpose(2, 0, 1)  # single channel
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        return x, y


class Lanes:
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

    def update(self, lane_mask):
        self.recent_fit.append(lane_mask)
        if len(self.recent_fit) > 5:
            self.recent_fit = self.recent_fit[1:]
        self.avg_fit = np.mean(self.recent_fit, axis=0)


def road_lines(image, model, lanes):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    small_img = cv2.resize(img_rgb, (160, 80))
    small_img_t = small_img.transpose((2, 0, 1))
    small_img_t = np.expand_dims(small_img_t, 0)
    small_img_t = torch.FloatTensor(small_img_t)

    with torch.no_grad():
        pred_t = model(small_img_t)
    pred_np = pred_t.numpy()[0, 0, :, :]
    lane_mask = (pred_np > 0.80).astype(np.uint8) * 255
    lanes.update(lane_mask)
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit.astype(np.uint8), blanks))
    lane_image = cv2.resize(lane_drawn, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(image, 1, lane_image, 1, 0)