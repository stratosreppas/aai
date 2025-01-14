import pandas as pd

# Read the CSV file
csv_file = 'octsvm.csv'
data = pd.read_csv(csv_file)

# Calculate the best parameter (c1) for the combination of dataset, flip percentage, and model with the best average accuracy
best_combination = data.groupby(['dataset', 'flip_percentage', 'tree_type', 'parameters'])['accuracy'].mean().idxmax()
best_c1 = best_combination[3]

print(f"The best parameter (c1) is: {best_c1}")

# Calculate the best accuracies per model, per dataset, per flip percentage
best_accuracies = data.groupby(['dataset', 'flip_percentage', 'tree_type'])['accuracy'].max().reset_index()

# Save the best accuracies to a CSV file
best_accuracies.to_csv('best_accuracies.csv', index=False)

print("Best accuracies per model, per dataset, per flip percentage have been saved to 'best_accuracies.csv'.")
