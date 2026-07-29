import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------------
df = pd.read_csv('car data.csv')
print("Rows before cleaning:", df.shape[0])

# ------------------------------------------------------------------
# 2. Remove motorcycles. This file mixes motorcycle listings in with
# real cars. Some bikes are recorded as "Brand Model" (e.g. "Honda
# Activa 125"), others as just "Model" (e.g. "Activa 3g") -- so both
# patterns need to be checked.
# ------------------------------------------------------------------
bike_brands = ['Bajaj', 'Hero', 'Honda Activa', 'Honda CB', 'Honda Dream', 'Honda Karizma',
               'Hyosung', 'KTM', 'Mahindra Mojo', 'Royal Enfield', 'Suzuki Access',
               'TVS', 'UM Renegade', 'Yamaha', 'Activa']

is_bike = df['Car_Name'].apply(lambda name: any(name.startswith(b) for b in bike_brands))
df = df[~is_bike].reset_index(drop=True)

print("Rows after removing motorcycles:", df.shape[0])

# ------------------------------------------------------------------
# 3. Extract Brand from Car_Name -- feature engineering matching the
# brief's "brand goodwill" example.
# ------------------------------------------------------------------
brand_map = {
    '800': 'Maruti Suzuki', 'alto 800': 'Maruti Suzuki', 'alto k10': 'Maruti Suzuki',
    'amaze': 'Honda', 'baleno': 'Maruti Suzuki', 'brio': 'Honda', 'camry': 'Toyota',
    'ciaz': 'Maruti Suzuki', 'city': 'Honda', 'corolla': 'Toyota', 'corolla altis': 'Toyota',
    'creta': 'Hyundai', 'dzire': 'Maruti Suzuki', 'elantra': 'Hyundai', 'eon': 'Hyundai',
    'ertiga': 'Maruti Suzuki', 'etios cross': 'Toyota', 'etios g': 'Toyota', 'etios gd': 'Toyota',
    'etios liva': 'Toyota', 'fortuner': 'Toyota', 'grand i10': 'Hyundai', 'i10': 'Hyundai',
    'i20': 'Hyundai', 'ignis': 'Maruti Suzuki', 'innova': 'Toyota', 'jazz': 'Honda',
    'land cruiser': 'Toyota', 'omni': 'Maruti Suzuki', 'ritz': 'Maruti Suzuki',
    's cross': 'Maruti Suzuki', 'swift': 'Maruti Suzuki', 'sx4': 'Maruti Suzuki',
    'verna': 'Hyundai', 'vitara brezza': 'Maruti Suzuki', 'wagon r': 'Maruti Suzuki',
    'xcent': 'Hyundai'
}

df['Brand'] = df['Car_Name'].map(brand_map)

unmapped = df[df['Brand'].isnull()]
if len(unmapped) > 0:
    print("\nWarning -- unmapped car names found:")
    print(unmapped['Car_Name'].unique())
else:
    print("All car names mapped to a brand successfully.")

print("\nBrand counts:")
print(df['Brand'].value_counts())

# ------------------------------------------------------------------
# 4. Explore the cleaned, cars-only dataset
# ------------------------------------------------------------------
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe())

print("\nFuel Type options:")
print(df['Fuel_Type'].value_counts())
print("\nSelling type options:")
print(df['Selling_type'].value_counts())
print("\nTransmission options:")
print(df['Transmission'].value_counts())
print("\nOwner options:")
print(df['Owner'].value_counts())

# ------------------------------------------------------------------
# 5. Visual exploration
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.hist(df['Selling_Price'], bins=30, color='steelblue', edgecolor='black')
plt.xlabel('Selling Price (in lakhs)')
plt.ylabel('Number of cars')
plt.title('Distribution of Selling Prices (Cars Only)')
plt.tight_layout()
plt.savefig('price_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(7, 6))
plt.scatter(df['Present_Price'], df['Selling_Price'], alpha=0.6, color='darkorange')
plt.xlabel('Present Price (current showroom price, lakhs)')
plt.ylabel('Selling Price (lakhs)')
plt.title('Selling Price vs Present Price')
plt.tight_layout()
plt.savefig('price_vs_present_price.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(7, 6))
plt.scatter(df['Year'], df['Selling_Price'], alpha=0.6, color='seagreen')
plt.xlabel('Year')
plt.ylabel('Selling Price (lakhs)')
plt.title('Selling Price vs Manufacturing Year')
plt.tight_layout()
plt.savefig('price_vs_year.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(7, 6))
plt.scatter(df['Driven_kms'], df['Selling_Price'], alpha=0.6, color='crimson')
plt.xlabel('Kilometers Driven')
plt.ylabel('Selling Price (lakhs)')
plt.title('Selling Price vs Kilometers Driven')
plt.tight_layout()
plt.savefig('price_vs_kms.png', dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fuel_avg = df.groupby('Fuel_Type')['Selling_Price'].mean().sort_values(ascending=False)
axes[0].bar(fuel_avg.index, fuel_avg.values, color='steelblue')
axes[0].set_title('Average Selling Price by Fuel Type')
axes[0].set_ylabel('Selling Price (lakhs)')
trans_avg = df.groupby('Transmission')['Selling_Price'].mean().sort_values(ascending=False)
axes[1].bar(trans_avg.index, trans_avg.values, color='darkorange')
axes[1].set_title('Average Selling Price by Transmission')
plt.tight_layout()
plt.savefig('price_by_category.png', dpi=150, bbox_inches='tight')
plt.show()

brand_avg = df.groupby('Brand')['Selling_Price'].mean().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
plt.bar(brand_avg.index, brand_avg.values, color='purple')
plt.ylabel('Average Selling Price (lakhs)')
plt.title('Average Selling Price by Brand')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('price_by_brand.png', dpi=150, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# 6. Preprocessing
# ------------------------------------------------------------------
df['Car_Age'] = 2020 - df['Year']
df = pd.get_dummies(df, columns=['Fuel_Type', 'Selling_type', 'Transmission', 'Brand'], drop_first=True)
df = df.drop(['Car_Name', 'Year'], axis=1)
df['Selling_Price_USD'] = (df['Selling_Price'] * 100000 / 96.21).round(0)

print("\nCleaned dataset preview:")
print(df.head())
print(df.dtypes)

# ------------------------------------------------------------------
# 7. Split into training and testing sets
# ------------------------------------------------------------------
X = df.drop(['Selling_Price', 'Selling_Price_USD'], axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# ------------------------------------------------------------------
# 8. Train and evaluate Linear Regression (baseline model)
# ------------------------------------------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test).clip(min=0)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

# ------------------------------------------------------------------
# 9. Train and evaluate Random Forest Regressor (better suited to
# capture nonlinear interactions between features, e.g. mileage
# mattering more for older cars than newer ones)
# ------------------------------------------------------------------
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test).clip(min=0)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# ------------------------------------------------------------------
# 10. Compare both models honestly
# ------------------------------------------------------------------
print("\n--- Model Comparison ---")
print(f"Linear Regression   -> MAE: {lr_mae:.2f} lakhs | R2: {lr_r2:.3f}")
print(f"Random Forest        -> MAE: {rf_mae:.2f} lakhs | R2: {rf_r2:.3f}")

model_comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'MAE (lakhs)': [round(lr_mae, 2), round(rf_mae, 2)],
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

# Use whichever model performed better for the final results
if rf_r2 > lr_r2:
    best_model_name = 'Random Forest'
    y_pred = rf_pred
else:
    best_model_name = 'Linear Regression'
    y_pred = lr_pred

print(f"\nBest performing model: {best_model_name}")

# ------------------------------------------------------------------
# 11. Final comparison table + image, using the best model
# ------------------------------------------------------------------
comparison = pd.DataFrame({
    'Actual (lakhs)': y_test.values.round(2),
    'Predicted (lakhs)': y_pred.round(2)
})
comparison['Difference (lakhs)'] = (comparison['Actual (lakhs)'] - comparison['Predicted (lakhs)']).round(2)

print(f"\nSample predictions vs actual using {best_model_name} (first 10 test cars):")
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
    diff = comparison['Difference (lakhs)'].iloc[row_index]
    if abs(diff) > 1.5:
        for col_index in range(len(table_preview.columns)):
            tbl[(row_index + 1, col_index)].set_facecolor('#fbe1e1')

plt.title(f'Sample Predictions vs Actual -- {best_model_name} (First 10 Test Cars)', pad=15)
plt.tight_layout()
plt.savefig('predictions_table.png', dpi=150, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# 12. Predicted vs Actual chart, using the best model
# ------------------------------------------------------------------
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue')

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect prediction')

plt.xlabel('Actual Selling Price (lakhs)')
plt.ylabel('Predicted Selling Price (lakhs)')
plt.title(f'Predicted vs Actual Selling Price -- {best_model_name}')
plt.legend()
plt.tight_layout()
plt.savefig('predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.show()


# --- Which feature does the winning model actually rely on? ---
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(importances.to_string(index=False))

plt.figure(figsize=(8, 6))
plt.barh(importances['Feature'], importances['Importance'], color='seagreen')
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Does Brand actually help, once Present_Price/Age are already known? ---
brand_cols = [col for col in X.columns if col.startswith('Brand_')]
X_no_brand = X.drop(columns=brand_cols)

X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_no_brand, y, test_size=0.2, random_state=42)

rf_no_brand = RandomForestRegressor(n_estimators=200, random_state=42)
rf_no_brand.fit(X_train_nb, y_train_nb)
rf_no_brand_pred = rf_no_brand.predict(X_test_nb).clip(min=0)

rf_no_brand_mae = mean_absolute_error(y_test_nb, rf_no_brand_pred)
rf_no_brand_r2 = r2_score(y_test_nb, rf_no_brand_pred)

print("\n--- Brand Ablation Test ---")
print(f"With Brand    -> MAE: {rf_mae:.2f} | R2: {rf_r2:.3f}")
print(f"Without Brand -> MAE: {rf_no_brand_mae:.2f} | R2: {rf_no_brand_r2:.3f}")