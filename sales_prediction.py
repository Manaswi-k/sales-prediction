
# Sales Prediction Project
# Forecast product sales using Machine Learning

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import joblib

# 2. Load Data
data = pd.read_csv('car_purchasing.csv')

# 3. Fix column names (remove leading/trailing spaces)
data.columns = data.columns.str.strip()

# 4. Initial Exploration
print(data.head())
print(data.info())
print(data.describe())

# 5. Data Preprocessing
## Handle missing values
data = data.dropna()

## Detect outliers (Z-score method)
z_scores = np.abs(stats.zscore(data.select_dtypes(include=np.number)))
data = data[(z_scores < 3).all(axis=1)]

## Feature and target selection
# 'Car Purchase Amount' is the sales target
target = 'Car Purchase Amount'
features_to_drop = [target, 'Customer Name', 'Customer e-mail', 'Country']
existing_features_to_drop = [col for col in features_to_drop if col in data.columns]
features = data.drop(columns=existing_features_to_drop)
y = data[target]
X = features

# 6. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 8. Model Building
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 9. Evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R2 Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.2f}")

# 10. Feature Importance
feature_importance = pd.DataFrame({'Feature': features.columns, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
