import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
df = pd.read_csv("/Users/shrinithask/Downloads/dataset.csv")


# Encode target variable
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"NO": 0, "YES": 1})

# Encode categorical column 'GENDER'
le = LabelEncoder()
df["GENDER"] = le.fit_transform(df["GENDER"])

# Features and target
X = df.drop(columns=["LUNG_CANCER"])
y = df["LUNG_CANCER"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
