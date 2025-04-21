import pandas as pd
import os

# Use path relative to current file location
file_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'dataux_labeled.csv')

# Load dataset
df = pd.read_csv(file_path)

# Clean the dataset
df = df.dropna(subset=['text'])
df['dark_pattern_label'] = df['dark_pattern_label'].str.strip().str.replace(r'[^\w\s/]', '', regex=True)

# Save to cleaned file in Data folder
output_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'cleaned_dataux_labeled.csv')
df.to_csv(output_path, index=False)
