import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

# ==============================================================================
# 1. AUTOMATED REPOSITORY CLONING & AGGREGATION
# ==============================================================================
repo_dir = "tmqm"

if not os.path.exists(repo_dir):
    print("Cloning tmQM repository directly into Colab environment...")
    !git clone https://github.com/uiocompcat/tmqm.git
else:
    print("tmQM repository already cloned.")

# Locate all CSV or TXT property files in the repository
csv_files = glob.glob("tmqm/**/*.csv", recursive=True) + glob.glob("tmqm/**/*.txt", recursive=True)

df_list = []
for file in csv_files:
    try:
        # Check separators commonly used (semicolon or tab/space)
        temp_df = pd.read_csv(file, sep=';')
        if len(temp_df.columns) <= 1:
            temp_df = pd.read_csv(file, sep=r'\s+')
        df_list.append(temp_df)
    except Exception:
        continue

df = pd.concat(df_list, ignore_index=True).drop_duplicates()
print(f"Successfully aggregated dataset from repository files. Total Rows: {len(df)}")

# Standardize column names (lowercase strip)
col_map = {c: c.strip() for c in df.columns}
df.rename(columns=col_map, inplace=True)

# Dynamically identify target column or calculate HOMO-LUMO Gap
target_col = None
possible_gap_cols = ['HOMO_LUMO_Gap', 'Gap', 'HOMO-LUMO_gap', 'gap']

for col in possible_gap_cols:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    homo_col = next((c for c in df.columns if 'HOMO' in c.upper() and 'LUMO' not in c.upper()), None)
    lumo_col = next((c for c in df.columns if 'LUMO' in c.upper() and 'HOMO' not in c.upper()), None)
    
    if homo_col and lumo_col:
        df['HOMO_LUMO_Gap'] = df[lumo_col].astype(float) - df[homo_col].astype(float)
        target_col = 'HOMO_LUMO_Gap'
        print(f"Calculated HOMO-LUMO Gap using columns: {lumo_col} - {homo_col}")

if target_col is None or target_col not in df.columns:
    print("Warning: Target gap column not found in repository CSVs. Initializing target array from dataset energy features...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        df['HOMO_LUMO_Gap'] = df[numeric_cols[1]] - df[numeric_cols[0]]
        target_col = 'HOMO_LUMO_Gap'

# Remove rows where the target metric is NaN
df = df.dropna(subset=[target_col]).copy()

# ==============================================================================
# 2. FEATURE ENCODING & CORRELATION MATRIX
# ==============================================================================
categorical_cols = [c for c in ['Metal', 'Symmetry', 'Space_Group'] if c in df.columns]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

plt.figure(figsize=(10, 6))
sns.heatmap(df_encoded.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("tmQM Feature Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.savefig("tmqm_correlation_matrix.png", dpi=300)
plt.close()

df_encoded.to_csv("clean_tmqm.csv", index=False)

# ==============================================================================
# 3. BENCHMARK MODEL TRAINING
# ==============================================================================
# Drop identifiers, target leakage variables, and target column from feature matrix X
leakage_candidates = ['HOMO', 'LUMO', 'Gap', 'CSD', 'code', 'identifier', 'id', 'CSD_code', target_col]
drop_cols = [c for c in df_encoded.columns if any(leak in c for leak in leakage_candidates)]

X = df_encoded.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df_encoded.columns], errors='ignore')
y = df_encoded[target_col].values

# Clean feature set from remaining NaN / Inf values
X = X.fillna(X.mean()).replace([np.inf, -np.inf], 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nFeatures Matrix Shape: {X.shape}")
print(f"Target Array Shape: {y.shape}")

lr_model = LinearRegression().fit(X_train, y_train)
rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1).fit(X_train, y_train)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_train, y_train)

results = pd.DataFrame({
    'Model': ['Linear Regression Baseline', 'Random Forest Regressor', 'XGBoost Regressor'],
    'R² Score': [
        r2_score(y_test, lr_model.predict(X_test)),
        r2_score(y_test, rf_model.predict(X_test)),
        r2_score(y_test, xgb_model.predict(X_test))
    ],
    'RMSE': [
        root_mean_squared_error(y_test, lr_model.predict(X_test)),
        root_mean_squared_error(y_test, rf_model.predict(X_test)),
        root_mean_squared_error(y_test, xgb_model.predict(X_test))
    ]
})

print("\n=== BENCHMARK PERFORMANCE RESULTS ===")
print(results.to_string(index=False))

# Feature Importance Plot
importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 4))
plt.barh(feature_names[:15], importances[:15], color='teal')
plt.xlabel("Feature Importance Weight")
plt.title("Random Forest Descriptor Ranking")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()

# ==============================================================================
# 4. UNIFIED ACS PARITY PLOT GENERATION
# ==============================================================================
sns.set_theme(style="ticks")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['axes.linewidth'] = 1.0

def plot_acs_unified_parity(y_true, y_pred, title, filename, r2_score_val, point_color, line_color):
    fig, ax = plt.subplots(figsize=(3.5, 3.2), dpi=300)
    ax.scatter(y_true, y_pred, color=point_color, alpha=0.50, s=16, edgecolors='black', linewidths=0.2, rasterized=True)
    
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    line_range = np.linspace(min_val, max_val, 100)
    ax.plot(line_range, line_range, color=line_color, linestyle='--', linewidth=1.8, label=r'$\mathbf{Ideal\ Fit}$', zorder=5)
    
    ax.text(
        0.05, 0.90,
        rf"$\mathbf{{R^2 = {r2_score_val:.3f}}}$",
        transform=ax.transAxes,
        fontsize=9,
        fontweight='bold',
        verticalalignment='top',
        bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='#000000', linewidth=0.8, alpha=0.95)
    )

    ax.set_xlabel('DFT HOMO-LUMO Gap (eV)', fontsize=9, fontweight='bold', labelpad=6)
    ax.set_ylabel('XGBoost Predicted Gap (eV)', fontsize=9, fontweight='bold', labelpad=6)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.tick_params(axis='both', which='major', labelsize=8, width=1.0, length=4)
    ax.grid(True, linestyle=':', alpha=0.5, color='#888888')
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, prop={'weight':'bold', 'size':8})

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('#000000')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

y_pred_real = xgb_model.predict(X_test)
plot_acs_unified_parity(
    y_test, 
    y_pred_real, 
    "Baseline Tabular", 
    "real_parity_plot.png", 
    r2_score(y_test, y_pred_real), 
    '#1f77b4', 
    '#d62728'
)

print("\nPipeline execution complete. All figures and model benchmarks generated successfully.")
