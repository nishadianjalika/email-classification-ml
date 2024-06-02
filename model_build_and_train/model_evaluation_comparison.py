import pandas as pd
import matplotlib.pyplot as plt

# Creating the DataFrame from the given data
data = {
    "Model": ["NaiveBayes", "Random Forest", "LSTM", "BERT"],
    "Validation F1 Score": [40.4, 59.82, 64.15, 58.63],
    "Testing F1 Score": [40.97, 59.79, 64.83, 59.32]
}

df = pd.DataFrame(data)

# Plotting the bar graph
plt.figure(figsize=(8, 5))
bar_width = 0.1
index = range(len(df))

plt.bar(index, df["Validation F1 Score"], bar_width, label="Validation F1 Score")
plt.bar([i + bar_width for i in index], df["Testing F1 Score"], bar_width, label="Testing F1 Score")

plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.title("Comparison of Validation and Testing F1 Scores for Models")
plt.xticks([i + bar_width / 2 for i in index], df["Model"], rotation=45)
plt.legend()

plt.tight_layout()
plt.show()
