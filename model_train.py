# ===================== 1. LOAD DATA =====================
import pandas as pd

df = pd.read_csv("model_ready_data.csv")

# ===================== 2. SPLIT FEATURES & TARGET =====================
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== 3. POWER TRANSFORMATION =====================
from sklearn.preprocessing import PowerTransformer
import joblib

num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

pt = PowerTransformer(method="yeo-johnson")

X_train[num_cols] = pt.fit_transform(X_train[num_cols])
X_test[num_cols] = pt.transform(X_test[num_cols])

# Save PowerTransformer
joblib.dump(pt, "power_transformer.pkl")

# ===================== 4. SCALING =====================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Save Scaler
joblib.dump(scaler, "scaler.pkl")

# ===================== 5. MODEL TRAINING =====================
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===================== 6. MODEL EVALUATION =====================
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ===================== 7. SAVE MODEL =====================
joblib.dump(model, "loan_model.pkl")

print("✅ Model, scaler, and transformer saved successfully.")
