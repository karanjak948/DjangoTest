# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# After importing the libraries you need to install tha following:
# pip install pandas
# pip install matplotlib seaborn
# pip install scikit-learn

# After installing the libraries, you can load and explore the Iris dataset.

# Setting seaborn style
sns.set(style="whitegrid")

# Task 1: Load and Explore the Dataset
# Load the Iris dataset
try:
    from sklearn.datasets import load_iris
    iris_data = load_iris()
    # Converting the dataset to a pandas DataFrame
    df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    df['species'] = iris_data.target
    df['species'] = df['species'].map({i: name for i, name in enumerate(iris_data.target_names)})
except FileNotFoundError:
    print("Dataset not found. Please check the file path.")

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())  # Use print() instead of display()

# Explore the structure of the dataset
print("\nDataset Information:")
print(df.info())

# Checking for missing values
print("\nMissing Values in the Dataset:")
print(df.isnull().sum())

# Task 2: Basic Data Analysis
# Basic statistics of the dataset
print("\nBasic Statistics:")
print(df.describe())  # Use print() instead of display()

# Grouping by 'species' and calculating the mean of each numerical column
species_mean = df.groupby('species').mean()
print("\nMean values grouped by species:")
print(species_mean)  # Use print() instead of display()

# Task 3: Data Visualization
# 1. Line Chart: Average Sepal Length per species
plt.figure(figsize=(12, 8))
plt.plot(species_mean.index, species_mean['sepal length (cm)'], marker='o')
plt.title("Average Sepal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Sepal Length (cm)")
plt.show()

# 2. Bar Chart: Average Petal Length per Species
plt.figure(figsize=(12, 8))
sns.barplot(x='species', y='petal length (cm)', data=df, ci=None)
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram: Distribution of Sepal Length
plt.figure(figsize=(12, 8))
plt.hist(df['sepal length (cm)'], bins=10, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot: Sepal Length vs. Petal Length
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='viridis')
plt.title("Sepal Length vs. Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# Findings and Observations
observations = """
Findings and Observations:
1. The average sepal length differs across species, with 'virginica' having the largest mean value.
2. The bar chart indicates that 'virginica' also has the longest average petal length among species.
3. The histogram shows that sepal length is mostly concentrated between 5.0 and 6.0 cm.
4. The scatter plot reveals a positive relationship between sepal length and petal length, especially evident in 'virginica' species.
"""
print(observations)
