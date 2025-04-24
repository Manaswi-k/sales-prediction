# sales-prediction
This project builds a machine learning model to predict product sales based on customer and demographic data, helping businesses make data-driven marketing and budgeting decisions.

-> What It Does

Predicts Car Purchase Amount based on customer profiles

Handles missing data, removes outliers, and scales features

Trains a Random Forest Regressor for accurate predictions

Evaluates model with R² Score and Mean Squared Error

Highlights the most important features affecting sales

-> Dataset

Source: car_purchasing.csv 

link:- https://www.kaggle.com/datasets/yashpaloswal/ann-car-sales-price-prediction/data

Includes customer demographics, personal income, and spending behavior

->Technologies Used

Python

pandas, numpy, matplotlib, seaborn

scikit-learn (for ML modeling & preprocessing)

joblib (for saving the model)

/SalesPrediction/
├── sales_prediction.py      # Full code for data prep, training, and evaluation

├── car_purchasing.csv       # Dataset

├── scaler.pkl               # Saved Scaler

├── sales_prediction_model.pkl # Trained ML model

└── README.md                # Documentation

✅ Results

R² Score: 0.9505

MSE: 3,561,146.23

➡️ The model explains 95% of the variation in purchase behavior!

## 📌 How to Run
1. Clone the repo and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the script:
   ```
   python sales_prediction.py
   ```

3. Check model performance and saved files.
