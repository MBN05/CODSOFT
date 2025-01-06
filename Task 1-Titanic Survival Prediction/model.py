import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
data = pd.read_csv('Titanic-Dataset.csv')

# Fill missing values with mean
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

# Label Encoding for categorical features
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
data['Embarked'] = le.fit_transform(data['Embarked'])
data['Cabin'] = le.fit_transform(data['Cabin'].fillna('Unknown'))

# Features and target
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin']]
y = data['Survived']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM Classifier
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
svm_accuracy = svm.score(X_test_scaled, y_test)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_accuracy = knn.score(X_test, y_test)

# Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_accuracy = dt.score(X_test, y_test)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_accuracy = rf.score(X_test, y_test)

print(f"SVM Accuracy: {svm_accuracy}")
print(f"KNN Accuracy: {knn_accuracy}")
print(f"Decision Tree Accuracy: {dt_accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")

survival_counts = data.groupby('Sex')['Survived'].sum()
total_counts = data['Sex'].value_counts()

print("Survival counts by Gender:")
print(survival_counts)
print("\nTotal counts by Gender:")
print(total_counts)

female_percentage = (survival_counts[1] / total_counts[1]) * 100  
male_percentage = (survival_counts[0] / total_counts[0]) * 100      

print(f"Percentage of Female Survivors: {female_percentage:.2f}%")
print(f"Percentage of Male Survivors: {male_percentage:.2f}%")
