#%%
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = "spotlight_frames.csv"
df = pd.read_csv(file_path)

# Clean the data: Remove NaN rows and reset index
df = df.dropna().reset_index(drop=True)

# Filter out unwanted accuracy types
filtered_df = df[~df["detaild nums"].isin(["medium acc", "short acc"])]

# Extract data for plotting
hyperparameter_values = filtered_df.columns[1:].astype(int)  # Convert column names to integers
filtered_acc_types = filtered_df["detaild nums"]
filtered_acc_values = filtered_df.iloc[:, 1:].astype(float)  # Convert accuracy values to float

# Plot the accuracy trends
plt.figure(figsize=(8, 8))
for i, acc_type in enumerate(filtered_acc_types):
    plt.plot(hyperparameter_values, filtered_acc_values.iloc[i], marker="o", label=acc_type)

plt.xlabel("Detaild nums (Hyperparameter)")
plt.ylabel("Accuracy (ACC)")
plt.title("Accuracy Trends vs. Hyperparameter Detaild Nums (Filtered)")
plt.legend()
plt.grid(linestyle="--", axis="y")
plt.show()
