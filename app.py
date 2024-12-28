# app.py (Optimized Version)

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the Dataset
data = pd.read_csv('Dataset/mushrooms.csv')

# Display initial data info
print("Dataset Head:\n", data.head())
print("\nDataset Info:\n", data.info())

# Step 2: Preprocessing and Encoding
# Encode all categorical features using LabelEncoder
label_encoder = LabelEncoder()
for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column])

# Step 3: Exploratory Data Analysis (EDA)
# Plot class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='class', data=pd.read_csv('Dataset/mushrooms.csv'), palette='viridis')
plt.title('Class Distribution (Edible vs Poisonous)')
plt.show()

# Correlation Heatmap (Top 10 Features)
top_corr_features = data.corr()['class'].abs().sort_values(ascending=False).head(10).index
plt.figure(figsize=(10, 8))
sns.heatmap(data[top_corr_features].corr(), cmap='coolwarm', annot=True)
plt.title('Correlation Heatmap (Top 10 Features)')
plt.show()

# Step 4: Model Training and Evaluation
# Splitting dataset
X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training (Fewer Estimators for Faster Training)
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 5: Feature Importance Plot
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot Top 5 Important Features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(5), palette='viridis')
plt.title('Top 5 Important Features')
plt.show()
