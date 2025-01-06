import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# Load your dataset
data = pd.read_csv('creditcard.csv')

# Features (X) and Target (y) columns
X = data.drop(columns=['Class'])  
y = data['Class']               

# Display original class distribution
print("Original class distribution:", Counter(y))

# Apply RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X, y)

# Display oversampled class distribution
print("Oversampled class distribution:", Counter(y_over))

# Save the oversampled dataset
oversampled_data = pd.DataFrame(X_over, columns=X.columns)
oversampled_data['Class'] = y_over
oversampled_data.to_csv('creditcard_oversampled_dataset.csv', index=False)
print("Oversampled dataset saved")
