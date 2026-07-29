import pandas as pd

df1 = pd.read_csv('Unemployment in India.csv')
df2 = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')

print("=== Dataset 1: Unemployment in India ===")
print(df1.head())
print(df1.columns.tolist())

print("\n=== Dataset 2: Unemployment Rate up to 11/2020 ===")
print(df2.head())
print(df2.columns.tolist())

print("\n=== Dataset 1 info ===")
print(df1.info())
print(df1.isnull().sum())

print("\n=== Dataset 2 info ===")
print(df2.info())
print(df2.isnull().sum())

# --- Clean Dataset 1 ---

# Strip leading/trailing whitespace from column names
df1.columns = df1.columns.str.strip()

# Drop rows that are entirely empty
df1 = df1.dropna()

# Convert Date column from text to an actual date type
df1['Date'] = pd.to_datetime(df1['Date'].str.strip(), format='%d-%m-%Y')

print("Dataset 1 shape after cleaning:", df1.shape)
print(df1.columns.tolist())
print(df1.dtypes)


# --- Clean Dataset 2 ---

df2.columns = df2.columns.str.strip()

# Rename the confusing duplicate 'Region.1' to something clearer
df2 = df2.rename(columns={'Region.1': 'Zone'})

df2['Date'] = pd.to_datetime(df2['Date'].str.strip(), format='%d-%m-%Y')

print("\nDataset 2 shape after cleaning:", df2.shape)
print(df2.columns.tolist())
print(df2.dtypes)

import matplotlib.pyplot as plt

# Group by date, averaging the unemployment rate across all regions for that date
trend = df1.groupby('Date')['Estimated Unemployment Rate (%)'].mean()

plt.figure(figsize=(10, 5))
plt.plot(trend.index, trend.values, color='darkred')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.title('India Unemployment Rate Over Time (2019-2020)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('trend_over_time.png', dpi=150, bbox_inches='tight')
plt.show()

rural_urban = df1.groupby(['Date', 'Area'])['Estimated Unemployment Rate (%)'].mean().unstack()

plt.figure(figsize=(10, 5))
plt.plot(rural_urban.index, rural_urban['Rural'], label='Rural', color='green')
plt.plot(rural_urban.index, rural_urban['Urban'], label='Urban', color='blue')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.title('Rural vs Urban Unemployment Rate Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('rural_vs_urban.png', dpi=150, bbox_inches='tight')
plt.show()


# Find the exact peak date
peak_date = trend.idxmax()
print("Peak unemployment date:", peak_date)
print("Peak national average rate:", round(trend.max(), 2), "%")

# Show the 10 worst-hit states/regions on that specific date
peak_data = df1[df1['Date'] == peak_date].sort_values(
    'Estimated Unemployment Rate (%)', ascending=False
)
print("\nTop 10 regions by unemployment rate on peak date:")
print(peak_data[['Region', 'Area', 'Estimated Unemployment Rate (%)']].head(10))



# Define "before" as end of 2019, "after" as the peak Covid months (Apr-Jun 2020)
before_covid = df1[df1['Date'] < '2020-01-01']['Estimated Unemployment Rate (%)'].mean()
during_covid = df1[(df1['Date'] >= '2020-04-01') & (df1['Date'] <= '2020-06-30')]['Estimated Unemployment Rate (%)'].mean()

print(f"\nAverage unemployment rate BEFORE Covid (2019): {before_covid:.2f}%")
print(f"Average unemployment rate DURING Covid peak (Apr-Jun 2020): {during_covid:.2f}%")
print(f"Increase: {during_covid - before_covid:.2f} percentage points")

# Combine Region + Area into one label for clarity (e.g., "Jharkhand (Urban)")
peak_data_top10 = peak_data.head(10).copy()
peak_data_top10['Label'] = peak_data_top10['Region'] + ' (' + peak_data_top10['Area'] + ')'

plt.figure(figsize=(10, 6))
plt.barh(peak_data_top10['Label'], peak_data_top10['Estimated Unemployment Rate (%)'], color='crimson')
plt.xlabel('Estimated Unemployment Rate (%)')
plt.title(f'Top 10 Regions by Unemployment Rate on Peak Date ({peak_date.strftime("%B %Y")})')
plt.gca().invert_yaxis()  # highest value at the top
plt.tight_layout()
plt.savefig('top10_regions.png', dpi=150, bbox_inches='tight')
plt.show()

# Average unemployment rate by broader geographic zone
zone_avg = df2.groupby('Zone')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)

print("\nAverage unemployment rate by Zone:")
print(zone_avg)

plt.figure(figsize=(8, 5))
plt.bar(zone_avg.index, zone_avg.values, color='steelblue')
plt.xlabel('Zone')
plt.ylabel('Average Estimated Unemployment Rate (%)')
plt.title('Average Unemployment Rate by Region (2020)')
plt.tight_layout()
plt.savefig('unemployment_by_zone.png', dpi=150, bbox_inches='tight')
plt.show()


from scipy import stats

# --- Is the before/after Covid gap statistically real, or could it be noise? ---
before_data = df1[df1['Date'] < '2020-01-01']['Estimated Unemployment Rate (%)']
during_data = df1[(df1['Date'] >= '2020-04-01') & (df1['Date'] <= '2020-06-30')]['Estimated Unemployment Rate (%)']

t_stat, p_value = stats.ttest_ind(before_data, during_data, equal_var=False)

print(f"\nT-test: Before vs During Covid Peak")
print(f"T-statistic: {t_stat:.2f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    print("This difference is statistically significant (p < 0.05) -- not due to random chance.")
else:
    print("This difference is not statistically significant.")

# --- Did unemployment actually recover by late 2020? ---
recovery_trend = df2.groupby('Date')['Estimated Unemployment Rate (%)'].mean()

plt.figure(figsize=(10, 5))
plt.plot(recovery_trend.index, recovery_trend.values, color='darkred', marker='o')
plt.axhline(y=before_data.mean(), color='green', linestyle='--',
            label=f'2019 average ({before_data.mean():.2f}%)')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.title('Unemployment Recovery Through Late 2020')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('recovery_trend.png', dpi=150, bbox_inches='tight')
plt.show()

latest_rate = df2[df2['Date'] == df2['Date'].max()]['Estimated Unemployment Rate (%)'].mean()
print(f"\nLatest available (Nov 2020) average rate: {latest_rate:.2f}%")
print(f"2019 pre-Covid average: {before_data.mean():.2f}%")
print(f"Remaining gap: {latest_rate - before_data.mean():.2f} percentage points")