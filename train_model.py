# ============ STEP 0: IMPORTS & SETUP ============
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
# Current directory where the script is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
INPUT_FILE = os.path.join(BASE_DIR, 'patient.csv')
OUTPUT_DIR = BASE_DIR  # Save models in the same folder

print("\n" + "=" * 80)
print("TRAINING RANDOM FOREST MODEL - LOCAL EXECUTION")
print("=" * 80)
print(f"Working Directory: {BASE_DIR}")

# ============ STEP 1: LOAD DATA ============

print("\nSTEP 1: Loading data...")

if not os.path.exists(INPUT_FILE):
    print(f"❌ ERROR: Could not find '{INPUT_FILE}'")
    print("   Please ensure patient.csv is in the same folder as this script.")
    exit()

df = pd.read_csv(INPUT_FILE)
print(f"✓ Loaded data from: {INPUT_FILE}")

print(f"✓ Data shape: {df.shape}")
print(f"✓ Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ============ STEP 2: PREPARE FEATURES ============

print("\nSTEP 2: Preparing features...")

# Add target
# Ensure unitdischargeoffset is numeric before calculation
df["unitdischargeoffset"] = pd.to_numeric(df["unitdischargeoffset"], errors='coerce')
df["icu_los_hours"] = df["unitdischargeoffset"] / 60.0
df = df.dropna(subset=["icu_los_hours"])

print(f"✓ Created target variable: icu_los_hours")
print(f"  Mean LOS: {df['icu_los_hours'].mean():.2f} hours")
print(f"  Median LOS: {df['icu_los_hours'].median():.2f} hours")

# Select features to exclude
exclude_cols = [
    "unitdischargeoffset",      # Used to create target
    "icu_los_minutes",          # Derived from target
    "icu_los_hours",            # TARGET
    "hospitaldischargeoffset",  # Correlated with target
    "patientunitstayid",        # Identifier
    "patienthealthsystemstayid",# Identifier
    "uniquepid",                # Identifier
    "hospitaladmitoffset"       # Time offset
]

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

# Get categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n✓ Feature selection:")
print(f"  Numeric features: {len(numeric_cols)}")
print(f"  Categorical features: {len(categorical_cols)}")

# Create feature matrix
X = df[numeric_cols + categorical_cols].copy()
y = df["icu_los_hours"].copy()

print(f"\n✓ Feature matrix shape: {X.shape}")

# ============ STEP 3: ENCODE CATEGORICAL ============

print("\nSTEP 3: Encoding categorical features...")

le_dict = {}
for col in categorical_cols:
    # Fill missing values
    X[col] = X[col].fillna("unknown")

    # Create encoder
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

    print(f"  ✓ {col:30}: {len(le.classes_)} classes")

print(f"\n✓ Encoded {len(le_dict)} categorical features")

# ============ STEP 4: HANDLE MISSING VALUES ============

print("\nSTEP 4: Handling missing values...")

# Check for missing values before
missing_before = X.isnull().sum().sum()
print(f"  Missing values before: {missing_before}")

# Fill with mean
X = X.fillna(X.mean())

# Check after
missing_after = X.isnull().sum().sum()
print(f"  Missing values after: {missing_after}")
print(f"✓ Missing values handled")

# ============ STEP 5: SCALE FEATURES ============

print("\nSTEP 5: Scaling features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f"✓ Features scaled")

# ============ STEP 6: SPLIT ============

print("\nSTEP 6: Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")

# ============ STEP 7: TRAIN ============

print("\nSTEP 7: Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rf_model.fit(X_train, y_train)
print(f"✓ Model trained")

# ============ STEP 8: EVALUATE ============

print("\nSTEP 8: Evaluating model...")

# 1) Predictions
y_test_pred  = rf_model.predict(X_test)

# 2) Numeric metrics
test_mae  = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2   = r2_score(y_test, y_test_pred)

print("\nResults:")
print(f"  Test  MAE: {test_mae:.4f} hours")
print(f"  Test RMSE: {test_rmse:.4f} hours")
print(f"  Test R²  : {test_r2:.4f}")

# 3) Plots (Will open in a new window)
print("  > Generating plots... (Close the plot window to continue)")

residuals_test = y_test - y_test_pred

plt.figure(figsize=(12,5))

# 3.a Residuals vs predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test_pred, residuals_test, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
plt.xlabel("Predicted values")
plt.ylabel("Residuals (y_true - y_pred)")
plt.title("Residuals vs Predicted (Test)")
plt.grid(True)

# 3.b Residuals distribution
plt.subplot(1, 2, 2)
sns.histplot(residuals_test, bins=30, kde=True, color="blue")
plt.xlabel("Residual")
plt.title("Residuals distribution (Test)")
plt.grid(True)

plt.tight_layout()
# Save plot instead of just showing, so it persists
plt.savefig(os.path.join(OUTPUT_DIR, 'evaluation_residuals.png'))
print(f"✓ Plot saved: evaluation_residuals.png")
# plt.show() # Uncomment if you want to see it pop up

# ============ STEP 9: SAVE ============

print("\nSTEP 9: Saving model components...")

model_path = os.path.join(OUTPUT_DIR, 'best_model.pkl')
joblib.dump(rf_model, model_path)
print(f"✓ Model: {model_path}")

scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler: {scaler_path}")

le_path = os.path.join(OUTPUT_DIR, 'label_encoders.pkl')
joblib.dump(le_dict, le_path)
print(f"✓ Encoders: {le_path}")

cols_path = os.path.join(OUTPUT_DIR, 'feature_columns.pkl')
joblib.dump(X.columns.tolist(), cols_path)
print(f"✓ Columns: {cols_path}")

# ============ STEP 10: VERIFY ============

print("\nSTEP 10: Verifying files...")

for name, path in [('Model', model_path), ('Scaler', scaler_path),
                   ('Encoders', le_path), ('Columns', cols_path)]:
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024
        print(f"✓ {name:15}: {size:>8.2f} KB")

# ============ STEP 11: TEST LOAD ============

print("\nSTEP 11: Testing reload...")
try:
    loaded_model = joblib.load(model_path)
    print(f"✓ Successfully reloaded model: {type(loaded_model).__name__}")
except Exception as e:
    print(f"❌ Error reloading model: {e}")

print("\n" + "="*80)
print("✓ COMPLETE - You can now run the Streamlit app.")
print("="*80)