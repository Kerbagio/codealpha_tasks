import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------------------------------------------
# 1. Load and explore
# ------------------------------------------------------------------
df = pd.read_csv('Advertising.csv')

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())

# Drop the unnamed index column -- it's just a row number, not real data
df = df.drop(columns=['Unnamed: 0'])
print("\nColumns after cleanup:", df.columns.tolist())

print("\nCorrelation with Sales:")
print(df.corr()['Sales'].sort_values(ascending=False))

# ------------------------------------------------------------------
# 2. Visual exploration
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(df['TV'], df['Sales'], alpha=0.6, color='steelblue')
axes[0].set_xlabel('TV Advertising ($ thousands)')
axes[0].set_ylabel('Sales (thousands of units)')
axes[0].set_title('Sales vs TV Spend')

axes[1].scatter(df['Radio'], df['Sales'], alpha=0.6, color='darkorange')
axes[1].set_xlabel('Radio Advertising ($ thousands)')
axes[1].set_title('Sales vs Radio Spend')

axes[2].scatter(df['Newspaper'], df['Sales'], alpha=0.6, color='crimson')
axes[2].set_xlabel('Newspaper Advertising ($ thousands)')
axes[2].set_title('Sales vs Newspaper Spend')

plt.tight_layout()
plt.savefig('sales_vs_channels.png', dpi=150, bbox_inches='tight')
plt.show()

corr_matrix = df.corr()

plt.figure(figsize=(6, 5))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation coefficient')
plt.xticks(ticks=np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns, rotation=45)
plt.yticks(ticks=np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns)

for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black')

plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# 3. Preprocessing (minimal -- data is already clean and fully numeric)
# ------------------------------------------------------------------
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# ------------------------------------------------------------------
# 4. Split into training and testing sets
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# ------------------------------------------------------------------
# 5. Train and compare Linear Regression vs Random Forest
# ------------------------------------------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test).clip(min=0)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test).clip(min=0)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\n--- Model Comparison ---")
print(f"Linear Regression -> MAE: {lr_mae:.2f} | R2: {lr_r2:.3f}")
print(f"Random Forest      -> MAE: {rf_mae:.2f} | R2: {rf_r2:.3f}")

model_comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'MAE': [round(lr_mae, 2), round(rf_mae, 2)],
    'R2 Score': [round(lr_r2, 3), round(rf_r2, 3)]
})

plt.figure(figsize=(7, 5))
plt.bar(model_comparison['Model'], model_comparison['R2 Score'], color=['gray', 'seagreen'])
plt.ylabel('R2 Score')
plt.title('Model Comparison: R2 Score')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Use whichever model performed better
if rf_r2 > lr_r2:
    best_model_name = 'Random Forest'
    y_pred = rf_pred
else:
    best_model_name = 'Linear Regression'
    y_pred = lr_pred

print(f"\nBest performing model: {best_model_name}")

# ------------------------------------------------------------------
# 6. Final comparison table + image
# ------------------------------------------------------------------
comparison = pd.DataFrame({
    'Actual Sales': y_test.values.round(2),
    'Predicted Sales': y_pred.round(2)
})
comparison['Difference'] = (comparison['Actual Sales'] - comparison['Predicted Sales']).round(2)

print(f"\nSample predictions vs actual using {best_model_name} (first 10 test rows):")
print(comparison.head(10).to_string(index=False))

table_preview = comparison.head(10).round(2).astype(str)

fig, ax = plt.subplots(figsize=(7, 4))
ax.axis('off')
tbl = ax.table(cellText=table_preview.values, colLabels=table_preview.columns.tolist(),
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)

for col_index in range(len(table_preview.columns)):
    header_cell = tbl[(0, col_index)]
    header_cell.set_facecolor('#2a578a')
    header_cell.set_text_props(color='white', weight='bold')

for row_index in range(len(table_preview)):
    diff = comparison['Difference'].iloc[row_index]
    if abs(diff) > 1.5:
        for col_index in range(len(table_preview.columns)):
            tbl[(row_index + 1, col_index)].set_facecolor('#fbe1e1')

plt.title(f'Sample Predictions vs Actual -- {best_model_name} (First 10 Test Rows)', pad=15)
plt.tight_layout()
plt.savefig('predictions_table.png', dpi=150, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# 7. Predicted vs Actual chart
# ------------------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue')

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect prediction')

plt.xlabel('Actual Sales (thousands of units)')
plt.ylabel('Predicted Sales (thousands of units)')
plt.title(f'Predicted vs Actual Sales -- {best_model_name}')
plt.legend()
plt.tight_layout()
plt.savefig('predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.show()



importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(importances.to_string(index=False))

plt.figure(figsize=(7, 5))
plt.bar(importances['Feature'], importances['Importance'], color='seagreen')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()


# Add an explicit TV x Radio interaction term
df['TV_Radio_Interaction'] = df['TV'] * df['Radio']

X_interaction = df[['TV', 'Radio', 'Newspaper', 'TV_Radio_Interaction']]

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_interaction, y, test_size=0.2, random_state=42)

lr_interaction_model = LinearRegression()
lr_interaction_model.fit(X_train_i, y_train_i)
lr_interaction_pred = lr_interaction_model.predict(X_test_i).clip(min=0)

lr_interaction_mae = mean_absolute_error(y_test_i, lr_interaction_pred)
lr_interaction_r2 = r2_score(y_test_i, lr_interaction_pred)

print("\n--- Testing the TV x Radio Synergy Effect ---")
print(f"Linear Regression (no interaction) -> MAE: {lr_mae:.2f} | R2: {lr_r2:.3f}")
print(f"Linear Regression (with TV x Radio) -> MAE: {lr_interaction_mae:.2f} | R2: {lr_interaction_r2:.3f}")
print(f"Random Forest                        -> MAE: {rf_mae:.2f} | R2: {rf_r2:.3f}")

synergy_comparison = pd.DataFrame({
    'Model': ['Linear Regression\n(no interaction)', 'Linear Regression\n(with TV x Radio)', 'Random Forest'],
    'R2 Score': [round(lr_r2, 3), round(lr_interaction_r2, 3), round(rf_r2, 3)]
})

plt.figure(figsize=(8, 5))
plt.bar(synergy_comparison['Model'], synergy_comparison['R2 Score'], color=['gray', 'orange', 'seagreen'])
plt.ylabel('R2 Score')
plt.title('Testing the TV x Radio Synergy Effect')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('synergy_test.png', dpi=150, bbox_inches='tight')
plt.show()

X_simplified = df[['TV', 'Radio', 'TV_Radio_Interaction']]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simplified, y, test_size=0.2, random_state=42)

lr_simplified_model = LinearRegression()
lr_simplified_model.fit(X_train_s, y_train_s)
lr_simplified_pred = lr_simplified_model.predict(X_test_s).clip(min=0)

lr_simplified_mae = mean_absolute_error(y_test_s, lr_simplified_pred)
lr_simplified_r2 = r2_score(y_test_s, lr_simplified_pred)

print("\n--- Simplified Model: Dropping Newspaper Entirely ---")
print(f"TV + Radio + interaction only -> MAE: {lr_simplified_mae:.2f} | R2: {lr_simplified_r2:.3f}")
print(f"All features + interaction    -> MAE: {lr_interaction_mae:.2f} | R2: {lr_interaction_r2:.3f}")