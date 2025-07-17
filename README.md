# 🏠 House Price Prediction with Scikit-learn

This project builds a machine learning model using a Random Forest Regressor to predict median house values based on housing data. It includes both training and inference pipelines, with automatic model saving and loading using `joblib`.

---

## 📁 Project Structure

```
├── housing.csv            # Main dataset
├── input.csv              # New data for inference
├── output.csv             # Predictions saved here
├── model.pkl              # Trained model (auto-generated)
├── pipeline.pkl           # Data preprocessing pipeline (auto-generated)
├── main.py                # Main script for training & inference
└── README.md              # Project documentation
```

---

## ⚙️ How It Works

### ✅ Training Phase (runs automatically if `model.pkl` is not found):

- Loads `housing.csv`
- Performs stratified sampling based on income category
- Splits into training and testing
- Separates numerical and categorical columns
- Prepares data using pipelines
- Trains a `RandomForestRegressor`
- Saves trained model and pipeline using `joblib`

### 🔍 Inference Phase (if model already exists):

- Loads `input.csv`
- Transforms data using saved pipeline
- Uses model to predict house prices
- Saves predictions to `output.csv`

---

## 🚀 How to Run

1. Make sure you have the following files:
   - `housing.csv` — for training
   - `input.csv` — new data you want to predict on (optional)

2. Run the script:
```bash
python main.py
```

- If no model is found, it trains one and saves it.
- If a model exists, it uses it for prediction on `input.csv`.

---

## 🧠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## ⚠️ Large File Note

GitHub doesn't allow pushing files over 100MB.

- If your `model.pkl` is large, use [Git LFS](https://git-lfs.github.com/)
- Or add it to `.gitignore` and store it elsewhere (e.g., Google Drive)

---

## 📬 Output Example

Your `output.csv` will look like this:

```csv
longitude,latitude,housing_median_age,...,median_house_value
-122.23,37.88,41.0,...,452600.0
```

---


