# Fake News Detection

This project implements a simple Fake News Detection model using Python and machine learning.

## 📊 Dataset

- The dataset used is `sample_fake_news_dataset.csv` containing news articles labeled as FAKE or REAL.
- Make sure this file is placed in the `dataset/` folder before running the code.

## 🚀 How it works

- The dataset is loaded using pandas.
- The text is converted into numerical features using `TfidfVectorizer`.
- A Passive Aggressive Classifier (PAC) is trained to classify the news articles.
- Model performance is evaluated using accuracy score and confusion matrix.

## 🛠️ Technologies Used

- Python 3.x
- pandas
- scikit-learn

## 🗂 Project Structure
FakeNewsDetection/
│
├── dataset/
│ └── sample_fake_news_dataset.csv
│
├── src/
│ └── fake_news_detection.py
│
├── README.md
└── requirements.txt

## ⚙️ How to run

1. Clone or download this repository.
2. Install required packages:

```bash
pip install -r requirements.txt
Navigate to the src/ directory and run the Python file:
python fake_news_detection.py
License
This project is open-source and free to use

---

## 📄 `requirements.txt`

```txt
pandas
scikit-learn
