import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

file_path = 'winequality-white.csv'

# Read the etiquettes from the first line
first_line = pd.read_csv(file_path, delimiter=';', nrows=0).columns.tolist()
column_names = [col.strip('"') for col in first_line]

# Read the data past the first line
X = pd.read_csv(file_path, delimiter=';', skiprows=1, names=column_names)

# Create class labels from the last column
y = X['quality'].apply(lambda x: 'good' if x > 5 else 'bad')
X = X.drop(columns='quality')
print(y.value_counts())

# Split the data into train and test, create an SVM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
svc = SVC(kernel='rbf', C=1, gamma=100)
svc.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svc.predict(X_test)

# Report accuracy, confusion matrix
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

