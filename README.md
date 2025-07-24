# 🧠 Probabilistic Age Estimation

<div align="center">

🌟 A deep learning framework for modeling **age prediction as a probability distribution** rather than a point estimate.  
<br>
Predict not just an age — but how **confident** your model is.

</div>

---

## 📌 Overview

Most age estimation models regress a single number. But human appearance is ambiguous — someone may **look** older or younger than they are. This project introduces a **probabilistic approach** to age estimation that captures this uncertainty.

### 🔍 Why Probabilistic?

- ✅ **Uncertainty Quantification** – Know how confident the model is in its prediction  
- 📉 **Error-Aware Decisions** – Avoid overconfident wrong guesses  
- 📊 **Richer Output** – Distributions over ages are more interpretable than raw numbers  

We modify a standard ResNet architecture to output the parameters of a **Gaussian distribution** (mean + variance) and train it using the **Negative Log-Likelihood (NLL)** loss.

---

## ✨ Key Features

- 🧠 **Gaussian Age Prediction** (mean + variance output)
- 🛠️ **Clean & Modular Codebase**
- 📈 **Distribution Visualizations**
- 🔁 **Data Augmentation with `torchvision.transforms`**
- 🧪 **Easy Evaluation & Inference Pipelines**
- 🧩 **Extensible to New Datasets & Architectures**

---

## 🗂 Project Structure

```
probabilistic-age-estimation/
├── data/                         # Input datasets
├── models/
│   └── resnet_probabilistic.py  # ResNet-based Gaussian regressor
├── saved_models/                # Checkpoints
├── utils/
│   ├── data_loader.py           # Custom Dataset & Loader
│   └── transforms.py            # Augmentation logic
├── train.py                     # Model training script
├── evaluate.py                  # Evaluation script
├── predict.py                   # Inference on new images
├── requirements.txt             # Dependencies
└── README.md
```

---

## ⚙️ Getting Started

### ✅ Prerequisites

Make sure you have Python ≥ 3.8 and the following libraries:

- PyTorch
- NumPy
- OpenCV-Python
- scikit-learn
- matplotlib

Install everything with:

```bash
pip install -r requirements.txt
```

---

## 📦 Installation

```bash
git clone https://github.com/es15326/probabilistic-age-estimation.git
cd probabilistic-age-estimation
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🗃 Dataset Preparation

Expected directory format:

```
data/
└── your_dataset/
    ├── train/
    ├── val/
    └── labels.csv  # Format: filename, age
```

---

## 🏋️‍♀️ Training the Model

```bash
python train.py \
  --dataset_path data/your_dataset \
  --label_file data/your_dataset/labels.csv \
  --model_output_dir saved_models/ \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-4
```

---

## 🔎 Inference

```bash
python predict.py \
  --image_path /path/to/image.jpg \
  --model_path saved_models/best_model.pth
```

Output includes:

- ✅ Mean predicted age
- 🔁 Standard deviation (uncertainty)
- 📉 Probability distribution plot

---

## 📊 Results

| Metric | Value |
|--------|-------|
| MAE    | 3.21  |
| RMSE   | 4.05  |

### 🎯 Example Predictions

**Low Uncertainty Prediction**

<img src="docs/low_uncertainty_example.png" width="400"/>

**High Uncertainty Prediction**

<img src="docs/high_uncertainty_example.png" width="400"/>

---

## 🧬 Model Architecture

Our model is based on **ResNet-34**, pre-trained on ImageNet and adapted for probabilistic regression:

- 🔵 **μ (Mean)**: Predicted center of the age distribution  
- 🔴 **log(σ²) (Log Variance)**: Captures uncertainty while ensuring positivity

Loss function: **Gaussian Negative Log-Likelihood (NLL)**

This allows the model to output **both the age and its confidence**.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a PR.  

```bash
# Example flow:
git checkout -b feature/YourAmazingFeature
git commit -m "Add feature"
git push origin feature/YourAmazingFeature
```

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for full details.

---

## 🙏 Acknowledgments

- [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- [IMDB-WIKI Dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
- ["Deep Expectation of Real and Apparent Age from a Single Image"](https://openaccess.thecvf.com/content_cvpr_2018/html/Shen_Deep_Expectation_of_CVPR_2018_paper.html)

---

## 📬 Contact

**Elham Soltani Kazemi**  
📧 your.email@example.com  
🔗 [Project Repository](https://github.com/es15326/probabilistic-age-estimation)

