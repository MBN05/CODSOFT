import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.svm import SVR
#from sklearn.neighbors import KNeighborsRegressor
#from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor

# Load dataset
data = pd.read_csv("IMDb_Movies_India.csv", encoding='latin1')

data['Year'] = data['Year'].str.extract(r'(\d{4})')

# Convert 'Year', 'Duration', and 'Votes' to numeric
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce')
data['Votes'] = pd.to_numeric(data['Votes'], errors='coerce')

# Handle missing values explicitly
if data['Year'].notnull().sum() > 0:
    data['Year'] = data['Year'].fillna(data['Year'].median())
else:
    data['Year'] = 0  # Default

if data['Duration'].notnull().sum() > 0:
    data['Duration'] = data['Duration'].fillna(data['Duration'].median())
else:
    data['Duration'] = 0

if data['Votes'].notnull().sum() > 0:
    data['Votes'] = data['Votes'].fillna(data['Votes'].median())
else:
    data['Votes'] = 0

data['Year'] = data['Year'].astype(int)

# Verify the data types again
print(data.dtypes)

# Fill numerical columns with their mean
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

# Fill categorical columns with a placeholder or mode
categorical_columns = data.select_dtypes(include=['object']).columns
data[categorical_columns] = data[categorical_columns].fillna('Missing')

# Encode categorical features
categorical_features = ['Name', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
for col in categorical_features:
    data[col] = data[col].astype('category').cat.codes

# Separate features and target
X = data.drop(columns=['Rating'])
y = data['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42) #0.69
#model = LinearRegression() #0.95
#model = GradientBoostingRegressor(random_state=42) #0.74
#model = SVR() #0.96
#model = KNeighborsRegressor(n_neighbors=5) #1.12
#model = XGBRegressor(random_state=42) #0.72
#model = LGBMRegressor(random_state=42) #0.67
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

best_actor = data.groupby('Actor 1')['Rating'].mean().idxmax()
print(f"The best actor (Actor 1) based on average ratings is: {best_actor}")

most_director = data['Director'].value_counts().idxmax()
print(f"The director who directed the most movies is: {most_director}")

highly_rated_movies = data[data['Rating'] > 9.0]
total_highly_rated_movies = len(highly_rated_movies)
print(f"Total number of movies with a rating greater than 9: {total_highly_rated_movies}")

top_movies_overall = data.nlargest(10, 'Rating')[['Name', 'Year', 'Rating']]
print("Top 10 movies overall:")
print(top_movies_overall)

best_year = data.groupby('Year')['Rating'].mean().idxmax()
print(f"The year with the best average rating is: {best_year}")
