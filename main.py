import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score


df = pd.read_excel("Final_Excel_data_NCS_UK.xlsx")
df = df.drop(columns=['S.No.', 'Range', 'FC Classes', 'Distance Classes', 'Unit Conversion'])
df = df.dropna()

# Create classification target column based on fire points
def classify_fire(points):
    if points >= 300:
        return 'High'
    elif points >= 100:
        return 'Medium'
    else:
        return 'Low'

df['Fire_Level'] = df['Fire Point(VIIRS)'].apply(classify_fire)

# Prepare training and test datasets
# Features (common for both)
X = df.drop(columns=['Fire Point(VIIRS)', 'Fire_Level'])

# Regression target
y_reg = df['Fire Point(VIIRS)']

# Classification target
y_clf = df['Fire_Level']

# Splitting data (70% training, 30% testing)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.3, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.3, random_state=42)

# Train Random Forest Regressor
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_reg, y_train_reg)

# Predict fire points on test set
y_pred_reg = reg_model.predict(X_test_reg)

# Regression Evaluation
print("\n Regression Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test_reg, y_pred_reg))
print("R^2 Score:", r2_score(y_test_reg, y_pred_reg))

# Train Random Forest Classifier
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_clf, y_train_clf)

# Predict fire class on test set
y_pred_clf = clf_model.predict(X_test_clf)

# Classification Evaluation
print("\n Classification Report:")
print(classification_report(y_test_clf, y_pred_clf))
print("Accuracy Score:", accuracy_score(y_test_clf, y_pred_clf))

# Get feature importances from the trained classifier model
feature_importances = clf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance - Random Forest Classifier')
plt.tight_layout()
plt.grid(True)
plt.show()
