from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from collections import Counter

data=pd.read_csv('creditcard.csv')

X=data.drop(columns=['Class'])
y=data['Class']

print("Original class distribution:",Counter(y))

undersample = RandomUnderSampler(sampling_strategy='majority')
X_under, y_under = undersample.fit_resample(X, y)
print("Undersampled class distribution:", Counter(y_under))

undersampled_data = pd.DataFrame(X_under, columns=X.columns)
undersampled_data['Class'] = y_under
undersampled_data.to_csv('creditcard_undersampled_dataset.csv', index=False)
print("Undersampled dataset saved")
