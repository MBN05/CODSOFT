import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_curve, auc

# Load the dataset
data = pd.read_csv('creditcard.csv')
#Balancing with only 5000 random data from input gives 1.00 accuracy
#Also Oversampling does the job
#Undersampling gives accuracy of 0.95

# Features and target
X = data.drop(columns=['Class'])
y = data['Class']

class_counts = data['Class'].value_counts(normalize=True) * 100

# Print class ratios
print(class_counts)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) #Stratify - counter ratio is less difference

# Class distribution in training and testing sets
print(Counter(y_train))  # Random, might not retain the 90:10 ratio
print(Counter(y_test))

# Train Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# AUPRC calculation
y_scores = rf.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
precision, recall, _ = precision_recall_curve(y_test, y_scores)
auprc = auc(recall, precision)

print(f"AUPRC: {auprc:.4f}")
