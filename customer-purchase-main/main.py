# main.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. تحميل البيانات
df = pd.read_csv("data/customers.csv")

# 2. ترميز البيانات النصية
le_gender = LabelEncoder()
le_profession = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Profession"] = le_profession.fit_transform(df["Profession"])

# 3. فصل البيانات
X = df.drop("Purchased", axis=1)
y = df["Purchased"]

# 4. تقسيم Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. تدريب الموديل
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# 6. التقييم
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc:.2f}")

# 7. حفظ الموديل
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/purchase_model.joblib")
print("💾 Model saved to model/purchase_model.joblib")

# 8. تجربة على عميل جديد
new_customer = pd.DataFrame({
    'Age': [35],
    'Income': [60000],
    'Gender': le_gender.transform(['Male']),
    'Profession': le_profession.transform(['Engineer'])
})

prediction = model.predict(new_customer)
print("🔍 Will the customer buy?", "Yes ✅" if prediction[0] == 1 else "No ❌")