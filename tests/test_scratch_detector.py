import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch_detector.data import draw_gaussian, gaussian_radius
from scratch_detector.eval import average_precision, iou_matrix
from scratch_detector.loss import detection_loss
from scratch_detector.model import OUTPUT_STRIDE, CenterNetLite, decode


def test_model_forward_and_decode_shapes():
    model = CenterNetLite(pretrained_backbone=False).eval()
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    grid = 128 // OUTPUT_STRIDE
    assert out["heatmap"].shape == (1, 1, grid, grid)
    assert out["wh"].shape == (1, 2, grid, grid)
    assert out["offset"].shape == (1, 2, grid, grid)
    boxes = decode(out, conf_threshold=0.0)
    assert len(boxes) == 1 and boxes[0].shape[1] == 5


def test_decode_recovers_planted_peak():
    grid = 32
    out = {
        "heatmap": torch.full((1, 1, grid, grid), -10.0),
        "wh": torch.zeros(1, 2, grid, grid),
        "offset": torch.zeros(1, 2, grid, grid),
    }
    out["heatmap"][0, 0, 10, 20] = 10.0  # strong peak at (x=20, y=10)
    out["wh"][:, 0, 10, 20] = 4.0  # width  = 4 cells
    out["wh"][:, 1, 10, 20] = 6.0  # height = 6 cells
    boxes = decode(out, conf_threshold=0.5)[0]
    assert len(boxes) == 1
    x1, y1, x2, y2, score = boxes[0].tolist()
    assert score > 0.99
    assert abs((x1 + x2) / 2 - 20 * OUTPUT_STRIDE) < 1e-4
    assert abs((x2 - x1) - 4 * OUTPUT_STRIDE) < 1e-4
    assert abs((y2 - y1) - 6 * OUTPUT_STRIDE) < 1e-4


def test_loss_decreases_toward_perfect_prediction():
    grid = 16
    target_heat = torch.zeros(1, 1, grid, grid)
    target_heat[0, 0, 8, 8] = 1.0
    targets = {
        "heatmap": target_heat,
        "wh": torch.zeros(1, 2, grid, grid),
        "offset": torch.zeros(1, 2, grid, grid),
        "mask": (target_heat > 0.99).float(),
    }
    perfect = {
        "heatmap": (target_heat * 2 - 1) * 12.0,  # logits: +12 at GT, -12 elsewhere
        "wh": torch.zeros(1, 2, grid, grid),
        "offset": torch.zeros(1, 2, grid, grid),
    }
    random = {
        "heatmap": torch.randn(1, 1, grid, grid),
        "wh": torch.randn(1, 2, grid, grid),
        "offset": torch.randn(1, 2, grid, grid),
    }
    assert detection_loss(perfect, targets)["total"] < detection_loss(random, targets)["total"]


def test_gaussian_radius_and_splat():
    assert gaussian_radius(10, 10) >= 1.0  # small boxes clamp to the 1-cell minimum
    radius = gaussian_radius(30, 30)
    assert radius > 1.0
    heat = np.zeros((32, 32), dtype=np.float32)
    draw_gaussian(heat, 16, 16, int(radius))
    assert heat[16, 16] == 1.0
    assert 0 < heat[16, 16 + int(radius)] < 1.0  # neighbours get soft positives


def test_iou_matrix_and_average_precision():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    perfect = np.hstack([gt, [[0.9], [0.8]]])
    np.testing.assert_allclose(iou_matrix(gt, gt).diagonal(), 1.0, rtol=1e-6)
    assert average_precision([perfect], [gt]) > 0.99
    # All-wrong detections → AP 0
    wrong = np.array([[50, 50, 60, 60, 0.9]])
    assert average_precision([wrong], [gt]) == 0.0
