"""From-scratch chapter: a minimal anchor-free (CenterNet-style) detector.

Continues the owner's "CNN from scratch" series: instead of calling
`ultralytics`, this package implements detection primitives by hand — gaussian
heatmap targets, penalty-reduced focal loss, and max-pool NMS decoding — and
benchmarks the result against the fine-tuned YOLO baseline.
"""
