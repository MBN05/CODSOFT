import pandas as pd
from imblearn.combine import SMOTEENN
from collections import Counter
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('creditcard.csv')

datas = data.sample(n=5000, random_state=42)

# Features (X) and Target (y) columns
X = datas.drop(columns=['Class'])
y = datas['Class']  

# Display original class distribution
print("Original class distribution:", Counter(y))

# Apply SMOTEENN
smote_enn = SMOTEENN()
X_balanced, y_balanced = smote_enn.fit_resample(X, y)

# Display balanced class distribution
print("Balanced class distribution:", Counter(y_balanced))

# Save the balanced dataset
balanced_data = pd.DataFrame(X_balanced, columns=X.columns) 
balanced_data['Class'] = y_balanced 
balanced_data.to_csv('creditcard_balanced_dataset.csv', index=False)
print("Balanced dataset saved")

# Visualize class distribution
class_counts = Counter(y_balanced)
plt.bar(class_counts.keys(), class_counts.values(), color=['blue', 'orange'])
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.title("Balanced Class Distribution (SMOTEENN)")
plt.xticks([0, 1], labels=['Class 0', 'Class 1'])
plt.show()
