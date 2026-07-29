import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------------
df = pd.read_csv('Iris.csv')
print(df.head())

# Check how many samples of each species we have
print(df['Species'].value_counts())

# Compare average measurements grouped by species
print(df.groupby('Species').mean(numeric_only=True))

# Print some basic info: column types, row count, memory usage
print(df.info())

# Print summary statistics: mean, min, max, etc. for each numeric column
print(df.describe())

# ------------------------------------------------------------------
# 2. Species distribution chart
# ------------------------------------------------------------------
species_counts = df['Species'].value_counts()

plt.figure(figsize=(6, 4))
plt.bar(species_counts.index, species_counts.values, color=['steelblue', 'darkorange', 'seagreen'])
plt.ylabel('Number of samples')
plt.title('Sample Count per Species')
plt.tight_layout()
plt.savefig('species_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# 3. Scatter plot: petal length vs petal width, colored by species
# ------------------------------------------------------------------
colors = {'Iris-setosa': 'blue', 'Iris-versicolor': 'orange', 'Iris-virginica': 'green'}

for species, color in colors.items():
    subset = df[df['Species'] == species]
    plt.scatter(subset['PetalLengthCm'], subset['PetalWidthCm'],
                label=species, color=color)

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Iris Species by Petal Size')
plt.legend()
plt.savefig('petal_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# 4. Split into training and testing sets
# ------------------------------------------------------------------
# X = the features (measurements) the model learns FROM
# y = the label (species) the model learns to PREDICT
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# ------------------------------------------------------------------
# 5. Train the KNN model
# ------------------------------------------------------------------
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", list(y_pred))
print("Actual:     ", list(y_test))

# ------------------------------------------------------------------
# 6. Evaluate: accuracy + classification report
# ------------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------------
# 7. Confusion matrix
# ------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.colorbar(label='Number of predictions')
plt.xticks(ticks=np.arange(len(model.classes_)), labels=model.classes_, rotation=45)
plt.yticks(ticks=np.arange(len(model.classes_)), labels=model.classes_)
plt.xlabel('Predicted species')
plt.ylabel('Actual species')
plt.title('Confusion Matrix')

for i in range(len(model.classes_)):
    for j in range(len(model.classes_)):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()


for species, color in colors.items():
    subset = df[df['Species'] == species]
    plt.scatter(subset['SepalLengthCm'], subset['SepalWidthCm'],
                label=species, color=color)

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Species by Sepal Size')
plt.legend()
plt.savefig('sepal_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

from sklearn.ensemble import RandomForestClassifier

# --- Compare KNN against Random Forest ---
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"\nKNN Accuracy: {accuracy * 100:.2f}%")
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

# --- Which measurement actually matters most? ---
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(importances.to_string(index=False))

plt.figure(figsize=(7, 5))
plt.bar(importances['Feature'], importances['Importance'], color='seagreen')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance (Iris)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Tune K instead of guessing 5 ---
k_values = range(1, 16)
k_accuracies = []

for k in k_values:
    knn_test = KNeighborsClassifier(n_neighbors=k)
    knn_test.fit(X_train, y_train)
    k_pred = knn_test.predict(X_test)
    k_accuracies.append(accuracy_score(y_test, k_pred))

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), k_accuracies, marker='o', color='steelblue')
plt.xlabel('K (number of neighbors)')
plt.ylabel('Accuracy on Test Set')
plt.title('KNN Accuracy vs K')
plt.xticks(list(k_values))
plt.tight_layout()
plt.savefig('k_tuning.png', dpi=150, bbox_inches='tight')
plt.show()

best_k = list(k_values)[k_accuracies.index(max(k_accuracies))]
print(f"\nBest K value: {best_k} with accuracy {max(k_accuracies) * 100:.2f}%")