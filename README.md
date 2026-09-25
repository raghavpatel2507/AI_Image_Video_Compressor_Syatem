# ⚡ AI-Based Image & Video Compression System

> **Content-adaptive AI compression for images and videos using feature extraction, Machine Learning, automated quality evaluation, and iterative parameter adjustment.**

An intelligent media compression system that analyzes the visual characteristics of each image or video and predicts an appropriate compression parameter using Computer Vision and Machine Learning.

The system aims to **reduce file size while maintaining acceptable visual quality**. After compression, the output is automatically evaluated using quality metrics. If the quality does not meet the required threshold, the system adjusts the compression parameter and retries.

---

# 1. 🎯 Problem Statement

The objective of this task is to build an **AI-assisted image and video compression system** that significantly reduces media file sizes while maintaining acceptable visual quality.

The system should:

* Compress both **images and videos**.
* Intelligently select or optimize compression parameters based on the characteristics of the input media.
* Automatically evaluate the compressed output and adjust compression parameters when required.
* Compare the **AI-assisted approach** with a traditional fixed-parameter compression approach.
* Report:

  * File-size reduction
  * Quality metrics
  * Processing time
  * AI/ML results

### Example Target

```text
Video: 50 MB → ~15 MB
Image: 10 MB → ~2 MB
```

The main objective is to achieve **significant file-size reduction while preserving acceptable visual quality through content-aware compression and automatic quality optimization**.

---

# 2. 💡 Proposed Approach

Instead of applying one fixed compression setting to every file, the system uses a **content-adaptive AI approach**.

The system first analyzes the input media, extracts relevant visual characteristics, and uses a Machine Learning model to predict an appropriate compression parameter.

The compressed output is then evaluated automatically. If the quality is below the required threshold, the system adjusts the parameter and performs compression again.

```text
Input Image / Video
        ↓
Feature Extraction
        ↓
ML Parameter Prediction
        ↓
Compression
        ↓
Quality Evaluation
        ↓
Quality Target Met?
     ↙       ↘
   YES        NO
    ↓          ↓
Final Output  Adjust Parameter
                 ↓
             Re-compress
                 ↓
            Maximum 3 Retries
```

---

# 3. 🧠 AI/ML Approach

The system uses `RandomForestRegressor` to predict compression parameters based on the extracted characteristics of the media.

|                      | Images                     | Videos                     |
| -------------------- | -------------------------- | -------------------------- |
| **Algorithm**        | `RandomForestRegressor`    | `RandomForestRegressor`    |
| **Prediction**       | WebP Quality               | H.264 CRF                  |
| **Prediction Range** | 45–95                      | 18–34                      |
| **Quality Target**   | SSIM ≥ 0.95                | SSIM ≥ 0.95                |
| **Fallback**         | Content-adaptive heuristic | Content-adaptive heuristic |

### Image Features

* Resolution
* Entropy
* Edge density
* Color variance
* Laplacian variance / sharpness

### Video Features

* Resolution
* FPS
* Duration
* Motion magnitude
* Edge density
* Entropy

---

# 4. 📊 Training Data & Label Generation

Training labels are generated automatically using **compression parameter sweeping** rather than manual labeling.

## 📷 Image Model

### Dataset

* **Dataset:** DIV2K
* **Subset:** `DIV2K_train_HR`
* **Images:** 800 high-resolution images

### Label Generation

For each image:

1. Extract visual features using OpenCV and NumPy.
2. Test multiple WebP quality values:

```text
30, 40, 50, 60, 70, 80, 90
```

3. Calculate SSIM against the original image.
4. Select the **minimum quality value achieving SSIM ≥ 0.95**.
5. Store the generated training data in:

```text
data/image_training_data.csv
```

### Model

```text
models/image_quality_model.joblib
```

---

## 🎬 Video Model

### Dataset

* **Source:** Pexels
* **Videos:** 18 HD/1080p videos
* **Content:** Low-, medium-, and high-motion videos

### Label Generation

For each video:

1. Extract video features.
2. Test different H.264 CRF values:

```text
18, 22, 26, 30, 34, 38, 42
```

3. Calculate frame-averaged SSIM.
4. Select the **maximum CRF maintaining SSIM ≥ 0.95**.
5. Store the generated training data in:

```text
data/video_training_data.csv
```

### Model

```text
models/video_quality_model.joblib
```

---

# 5. 🔄 Compression & Quality Optimization Pipeline

The complete pipeline is:

```text
┌─────────────────────────────┐
│     Input Image / Video     │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│      Feature Extraction     │
│   OpenCV / NumPy / PIL       │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│    ML Parameter Prediction  │
│    RandomForestRegressor     │
│                             │
│ Image → WebP Quality        │
│ Video → H.264 CRF           │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│      Compression Engine     │
│      Pillow / FFmpeg        │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│      Quality Evaluation     │
│      SSIM / PSNR / VMAF     │
└──────────────┬──────────────┘
               ↓
        ┌───────────────┐
        │ SSIM ≥ 0.95 ? │
        └───────┬───────┘
             YES│       │NO
                ↓       ↓
          Final Output  Adjust Parameter
                         ↓
                     Re-compress
                         ↓
                   Maximum 3 Retries
```

---

# 6. 📐 Automatic Quality Evaluation

The system evaluates the compressed output using objective quality metrics.

### SSIM

**Structural Similarity Index**

Used as the primary quality threshold:

```text
Target: SSIM ≥ 0.95
```

### PSNR

**Peak Signal-to-Noise Ratio**

Used as an additional objective quality metric.

### VMAF

**Video Multimethod Assessment Fusion**

Used for video perceptual-quality evaluation when FFmpeg supports `libvmaf`.

```text
VMAF: 0–100
```

---

# 7. 🔁 Automatic Parameter Adjustment

The ML prediction is followed by an automatic quality check.

If the output satisfies the target:

```text
SSIM ≥ 0.95
       ↓
Accept Output
```

If the output does not satisfy the target:

```text
SSIM < 0.95
       ↓
Adjust Compression Parameter
       ↓
Re-compress
       ↓
Evaluate Again
```

The system performs up to **3 adjustment iterations**.

This allows the system to correct cases where the initial ML prediction does not produce the required quality.

---

# 8. ⚖️ AI-Assisted vs Traditional Compression

The system compares its content-adaptive approach against a traditional fixed-parameter baseline.

| Metric                       | Traditional Fixed Compression | AI-Assisted Compression |
| ---------------------------- | ----------------------------- | ----------------------- |
| Compression parameter        | Fixed                         | Predicted per file      |
| Content awareness            | No                            | Yes                     |
| Automatic quality evaluation | No                            | Yes                     |
| Parameter adjustment         | No                            | Yes                     |
| Quality metrics              | Not enforced                  | SSIM / PSNR / VMAF      |
| Parameter selection          | Manual / fixed                | ML + fallback heuristic |

The comparison focuses on:

* File-size reduction
* Visual quality
* Processing time
* Compression parameters
* ML prediction results

---

# 9. 📊 Evaluation & Reporting

For every compression run, the system records relevant results.

### Image / Video Metrics

```text
Original File Size
Compressed File Size
Size Reduction %
Compression Parameter
SSIM
PSNR
VMAF (Video)
Processing Time
Number of Adjustment Iterations
```

### Example

```text
Original Size      : 50 MB
Compressed Size    : 15 MB
Size Reduction     : 70%
SSIM               : 0.96
Processing Time    : X seconds
Iterations         : 1
```

The same metrics are used to compare the AI-assisted approach with the traditional fixed-parameter baseline.

---

# 10. 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │    Image / Video     │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │   Feature Extractor  │
                    │ OpenCV / NumPy / PIL │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │  ML Parameter Model  │
                    │ RandomForestRegressor│
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Compression Engine   │
                    │ Pillow / FFmpeg      │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │  Quality Evaluator   │
                    │ SSIM / PSNR / VMAF   │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Auto-Adjustment Loop │
                    │ Max 3 iterations     │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Final Output +       │
                    │ Evaluation Metrics   │
                    └──────────────────────┘
```

---

# 11. 📁 Project Structure

```text
ai_compression_system/
│
├── README.md
├── SETUP_PLAN.md
├── requirements.txt
├── pyproject.toml
├── app.py
│
├── src/
│   ├── __init__.py
│   ├── image_features.py
│   ├── video_features.py
│   ├── image_compressor.py
│   ├── video_compressor.py
│   ├── quality_metrics.py
│   ├── ml_predictor.py
│   ├── generate_training_data.py
│   ├── pipeline.py
│   ├── baseline.py
│   └── report.py
│
├── data/
│   ├── image_training_data.csv
│   └── video_training_data.csv
│
├── models/
│   ├── image_quality_model.joblib
│   └── video_quality_model.joblib
│
└── Test_outputs/
```

---

# 12. 🧰 Technology Stack

| Technology       | Purpose                                |
| ---------------- | -------------------------------------- |
| **Python 3.10+** | Core application                       |
| **OpenCV**       | Computer vision and feature extraction |
| **NumPy**        | Numerical processing                   |
| **Pillow**       | Image compression                      |
| **FFmpeg**       | Video encoding                         |
| **scikit-learn** | Machine Learning                       |
| **scikit-image** | SSIM and PSNR                          |
| **Pandas**       | Training data and reports              |
| **Joblib**       | ML model persistence                   |
| **Streamlit**    | Web application                        |
| **Pytest**       | Automated testing                      |

---

# 13. 🔍 Feature Extraction

### Image

`extract_image_features(image_path)` extracts:

* Shannon entropy
* Canny edge density
* Color-channel variance
* Laplacian variance / sharpness
* Resolution in megapixels

### Video

`extract_video_features(video_path)` samples frames and calculates:

* Average frame entropy
* Average edge density
* Optical-flow motion magnitude
* Resolution
* FPS
* Duration

Video motion is calculated using the **Farneback optical-flow algorithm**.

---

# 14. 🤖 Machine Learning Engine

The ML engine contains separate models for image and video compression.

### Image Model

```text
RandomForestRegressor
        ↓
WebP Quality
        ↓
45–95
```

### Video Model

```text
RandomForestRegressor
        ↓
H.264 CRF
        ↓
18–34
```

### Fallback

If a trained model is unavailable, the system uses a **content-adaptive heuristic** to determine the compression parameter.

---

# 15. 🗜️ Compression Engines

### Images

Pillow is used for:

* WebP
* JPEG

with configurable quality parameters.

### Videos

FFmpeg is used for:

* H.264 / `libx264`
* AAC audio
* `yuv420p`
* `faststart` optimization

---

# 16. ⚙️ Setup

## Linux / Ubuntu

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv ffmpeg
```

## macOS

```bash
brew install python ffmpeg
```

## Windows

1. Install Python 3.10+.
2. Enable **Add Python to PATH**.
3. Install FFmpeg.
4. Add the FFmpeg `bin` directory to the system PATH.

---

# 17. 📦 Install Dependencies

```bash
cd ai_compression_system

python3 -m venv .venv
```

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# 18. 🚀 Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# 19. 🔁 Model Retraining

## Image Model

Generate training data:

```bash
python -m src.generate_training_data \
    --images /path/to/your/image_folder \
    --out-dir data
```

Train the model:

```bash
python -c "import pandas as pd; from src.ml_predictor import train_image_model; df=pd.read_csv('data/image_training_data.csv'); train_image_model(df); print('Image ML Model Trained Successfully!')"
```

## Video Model

Generate training data:

```bash
python -m src.generate_training_data \
    --videos /path/to/your/video_folder \
    --out-dir data
```

Train the model:

```bash
python -c "import pandas as pd; from src.ml_predictor import train_video_model; df=pd.read_csv('data/video_training_data.csv'); train_video_model(df); print('Video ML Model Trained Successfully!')"
```

---

## 🖼️ Testing Samples — Side-by-Side Comparison

Real-world compression results from the AI pipeline. Each pair shows the **Original** vs the **AI-Compressed** output, with file sizes displayed below.

```
### 📷 Image Comparison

| 🔵 Original Image | 🟢 AI-Compressed Image |
|:---:|:---:|
| ![Original Image](Test_outputs/example.jpg) | ![AI Compressed Image](Test_outputs/ai_compressed_example.webp) |
| 📁 `example.jpg` | 📁 `ai_compressed_example.webp` |
| 📦 **Size: 1.21 MB** | 📦 **Size: 0.56 MB** ✅ *~53.6% smaller* |

---

### 🎬 Video Comparison
| 🔵 Original Video | 🟢 AI-Compressed Video |
|:---:|:---:|
| [▶️ Watch Original Video](https://drive.google.com/file/d/1phyabcHTg4MG_lMOs-qcjteiHEkth8i1/view?usp=drive_link) | [▶️ Watch AI-Compressed Video](https://drive.google.com/file/d/1rirNjvNv9zFsW4EnATLjeC1lcIPB0E09/view?usp=drive_link) |
| 📁 `ex.mp4` | 📁 `ai_compressed_ex.mp4` |
| 📦 **53.18 MB** | 📦 **5.46 MB** |
| — | 📉 **89.7% smaller** |

### Demo Video

[▶️ Watch Demo Video](https://drive.google.com/file/d/1TYAPIBi8EHDpfGi38akrcIA2aB3q7mlb/view?usp=sharing)
---

# 21. 🎯 Key Takeaway

The system combines:

```text
Computer Vision
      +
Feature Extraction
      +
Machine Learning
      +
Content-Adaptive Compression
      +
Automatic Quality Evaluation
      +
Parameter Adjustment
```

to build an **AI-assisted image and video compression system**.

The system does not rely on a single fixed compression parameter. Instead, it analyzes each media file, predicts an appropriate parameter, evaluates the compressed output, and adjusts the parameter when necessary.

> **Objective: Significantly reduce file size while maintaining acceptable visual quality and providing measurable AI/ML, quality, and performance results.**
