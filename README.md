# Real-Time Monocular 2D Ball Detection System (YOLO11n + ONNX FP16)

A complete, end-to-end, high-performance ball detection system optimized for maximum F1 score and real-time FPS on a monocular 2D webcam or video stream.

---

## 1. Technical Stack & Architectural Rationale

- **Model Backbone**: **Ultralytics YOLO11n** (Nano variant). Chosen for its optimal speed/accuracy trade-off on edge/desktop GPUs & CPUs. Pretrained on COCO (class 32: `sports ball`), enabling fast transfer learning and high F1 score convergence in under 35 epochs.
- **Data Augmentations**: Albumentations & Ultralytics pipeline simulating motion blur, cutout (occlusion), HSV color shifts, perspective transforms, and scale variation across diverse lighting conditions.
- **Threshold Optimization**: Sweeps confidence thresholds (0.10 to 0.90) to find the exact threshold that maximizes the validation **F1 score** ($F1 = 2 \cdot \frac{P \cdot R}{P + R}$ @ IoU $\ge 0.5$).
- **Inference Acceleration**: **ONNX Runtime (FP16 precision + CUDA/CPU EP)**, reducing latency by up to 3x compared to raw PyTorch `.pt`.
- **Real-Time Threading**: Multi-threaded producer-consumer frame pipeline decoupling OpenCV video decoding from model inference to prevent frame buffer queuing and eliminate I/O bottlenecks.

---

## 2. Directory Structure

```
ball-detection/
├── data/                    # Dataset images, labels, and data.yaml (80/10/10 split)
├── models/                  # Trained PyTorch (best.pt) and optimized ONNX (ball_detect_opt.onnx)
├── results/                 # Evaluation plots (precision_recall_f1_curve.png), metrics, and screenshots
├── scripts/
│   ├── download_data.py     # Automated dataset downloading & YOLO conversion
│   ├── train.py             # Fine-tunes YOLO11n with heavy augmentations
│   ├── metrics.py           # Sweeps confidence thresholds & generates F1 PR curves
│   ├── export_optimize.py   # Exports to ONNX FP16 and benchmarks FPS
│   └── run_pipeline.py      # End-to-end automated execution manager
├── realtime_detect.py       # Multi-threaded real-time detection app
├── requirements.txt         # Dependency declarations
├── results.md               # Detailed evaluation & benchmark table
└── README.md                # System documentation
```

---

## 3. Quick Start & Execution Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Option A: Run Full Automated Pipeline (End-to-End)
```bash
python scripts/run_pipeline.py
```

### Option B: Run Modular Steps Manually

1. **Download & Prepare Dataset**:
   ```bash
   python scripts/download_data.py
   ```
2. **Fine-Tune Model**:
   ```bash
   python scripts/train.py
   ```
3. **Evaluate & Find Optimal F1 Confidence Threshold**:
   ```bash
   python scripts/metrics.py
   ```
4. **Export & Speed Optimization**:
   ```bash
   python scripts/export_optimize.py
   ```
5. **Launch Real-Time Detection Application**:
   - **Webcam**:
     ```bash
     python realtime_detect.py --source 0
     ```
   - **Video File**:
     ```bash
     python realtime_detect.py --source path/to/video.mp4
     ```
   - **Interactive Hotkeys in Real-Time App**:
     - `q`: Quit application
     - `s`: Save detection screenshot to `results/screenshots/`

---

## 4. Combined System Score Formula

$$\text{Combined Score} = (F1 \times 0.7) + (\min(\frac{\text{FPS}}{30}, 1.0) \times 0.3)$$

- Rewards high detection quality ($F1$ score weighted at 70%).
- Rewards real-time responsiveness (FPS normalized to target 30 FPS weighted at 30%).
