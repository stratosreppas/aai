import pandas as pd

# Read the CSV file
csv_file = 'results copy 2.csv'
data = pd.read_csv(csv_file)

# Calculate the best parameter (c1) for each combination of dataset, flip percentage, and tree type by taking the mean accuracy across all folds
best_parameters = data.groupby(['dataset', 'flip_percentage', 'tree_type', 'parameters'])['accuracy'].mean().reset_index()

# Find the best c1 parameter for each combination by selecting the parameter with the highest mean accuracy
best_parameters = best_parameters.loc[best_parameters.groupby(['dataset', 'flip_percentage', 'tree_type'])['accuracy'].idxmax()]

# Save the best parameters to a CSV file
best_parameters.to_csv('best_parameters.csv', index=False)

print("Best parameters per dataset, tree type, and flip percentage have been saved to 'best_parameters.csv'.")
