# ================================
# Pizza Price Prediction Project
# All-in-One Python Script (VS Code)
# ================================

# ---------- Imports ----------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from tkinter import *

# ---------- Load Dataset ----------
data = pd.read_csv("pizza_v2.csv")

# ---------- Data Preprocessing ----------
data.rename({'price_rupiah': 'price'}, axis=1, inplace=True)

data['price'] = data['price'].str.replace("Rp", "")
data['price'] = data['price'].str.replace(",", "").astype('int32')

def convert(value):
    return value * 0.0054

data['price'] = data['price'].apply(convert)

data['diameter'] = data['diameter'].str.replace("inch", "")
data['diameter'] = data['diameter'].str.replace(" ", "").astype('float32')

# ---------- Remove Outliers ----------
data = data.drop(data.index[[6, 11, 16, 80]])

# ---------- Label Encoding ----------
cat_cols = data.select_dtypes(include=['object']).columns
encoder = LabelEncoder()

for col in cat_cols:
    data[col] = encoder.fit_transform(data[col])

# ---------- Feature & Target ----------
X = data.drop('price', axis=1)
Y = data['price']

# ---------- Train Test Split ----------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42
)

# ---------- Models ----------
lr = LinearRegression()
svr = SVR()
rf = RandomForestRegressor()
gbr = GradientBoostingRegressor()
xgb = XGBRegressor()

# ---------- Training ----------
lr.fit(X_train, Y_train)
svr.fit(X_train, Y_train)
rf.fit(X_train, Y_train)
gbr.fit(X_train, Y_train)
xgb.fit(X_train, Y_train)

# ---------- Predictions ----------
pred_lr = lr.predict(X_test)
pred_svr = svr.predict(X_test)
pred_rf = rf.predict(X_test)
pred_gbr = gbr.predict(X_test)
pred_xgb = xgb.predict(X_test)

# ---------- Evaluation ----------
print("LR R2 :", r2_score(Y_test, pred_lr))
print("SVR R2:", r2_score(Y_test, pred_svr))
print("RF R2 :", r2_score(Y_test, pred_rf))
print("GB R2 :", r2_score(Y_test, pred_gbr))
print("XGB R2:", r2_score(Y_test, pred_xgb))

# ---------- Save Best Model ----------
joblib.dump(xgb, "Pizza_price_predict")
model = joblib.load("Pizza_price_predict")

# ---------- Test Prediction ----------
test_df = pd.DataFrame({
    'company': [1],
    'diameter': [22.0],
    'topping': [2],
    'variant': [8],
    'size': [1],
    'extra_sauce': [1],
    'extra_cheese': [1],
    'extra_mushrooms': [1]
})

print("Sample Prediction:", model.predict(test_df))

# ================================
# GUI - Tkinter App
# ================================

def predict_price():
    values = [
        float(e1.get()), float(e2.get()), float(e3.get()), float(e4.get()),
        float(e5.get()), float(e6.get()), float(e7.get()), float(e8.get())
    ]

    df = pd.DataFrame([values], columns=[
        'company', 'diameter', 'topping', 'variant',
        'size', 'extra_sauce', 'extra_cheese', 'extra_mushrooms'
    ])

    result = model.predict(df)
    result_label.config(text=f"Predicted Price: ₹{result[0]:.2f}")

# ---------- GUI Window ----------
root = Tk()
root.title("Pizza Price Predictor")
root.geometry("400x400")

labels = [
    "Company", "Diameter", "Topping", "Variant",
    "Size", "Extra Sauce", "Extra Cheese", "Extra Mushrooms"
]

entries = []
for i, text in enumerate(labels):
    Label(root, text=text).grid(row=i+1, column=0, padx=10, pady=5)
    entry = Entry(root)
    entry.grid(row=i+1, column=1)
    entries.append(entry)

e1, e2, e3, e4, e5, e6, e7, e8 = entries

Button(root, text="Predict Price", command=predict_price).grid(row=10, column=1, pady=15)
result_label = Label(root, text="")
result_label.grid(row=11, column=1)

root.mainloop()
