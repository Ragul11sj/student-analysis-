import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("student_performance.csv")
print("Data Loaded:\n", df.head())

# Check for null values
print("\nMissing values:\n", df.isnull().sum())

# Convert categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Describe numerical data
print("\nSummary statistics:\n", df.describe())

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.show()

# Pairplot
sns.pairplot(df, hue="gender")
plt.savefig("pairplot.png")
plt.show()
