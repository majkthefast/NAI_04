"""
Klasyfikacja za pomocą Maszyny Wektorów Nośnych (SVM) dla Oceny Graczy NBA

Autor: Mikołaj Prętki; Mikołaj Hołdakowski

Ten skrypt implementuje klasyfikator SVM do oceny graczy NBA na podstawie wybranych cech. Zmienną docelową jest 'net_rating', sklasyfikowana jako pozytywna lub negatywna.

Funkcje:
1. preprocess_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series]
   - Wczytuje dane graczy NBA z pliku CSV.
   - Przetwarza dane, wybierając konkretne cechy i tworząc zmienną docelową.
   - Zwraca krotkę zawierającą cechy (X) i zmienną docelową (y).

2. train_svm(X_train: pd.DataFrame, y_train: pd.Series) -> SVC
   - Trenuje klasyfikator SVM na dostarczonych danych treningowych.
   - Wykorzystuje GridSearchCV do strojenia hiperparametrów.
   - Zwraca wytrenowany model SVM.

3. evaluate_svm(model: SVC, X_test: pd.DataFrame, y_test: pd.Series) -> None
   - Ocenia model SVM na zestawie testowym.
   - Wyświetla dokładność i macierz pomyłek.

Użycie:
1. Wczytaj dane: X, y = preprocess_data('all_seasons.csv')
2. Podziel dane: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
3. Trenuj model: svm_model = train_svm(X_train, y_train)
4. Oceń model: evaluate_svm(svm_model, X_test, y_test)

Referencje:
- Maszyny Wektorów Nośnych: https://scikit-learn.org/stable/modules/svm.html
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Read NBA player data
file_path = 'all_seasons.csv'
data = pd.read_csv(file_path)

# Preprocessing
# (You may need to adjust these based on your specific criteria for evaluating NBA players)
selected_features = ['age', 'player_height', 'player_weight', 'gp', 'pts', 'reb', 'ast']
X = data[selected_features]
y = data['net_rating'].apply(lambda x: 'positive' if x > 0 else 'negative')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# SVM with GridSearchCV for hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10], 'kernel': ['rbf']}
svm = SVC()
grid = GridSearchCV(svm, param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)

# Get the best SVM model from the grid search
best_svm = grid.best_estimator_

# Make predictions on the test set
y_pred_svm = best_svm.predict(X_test)

# Report accuracy and confusion matrix for SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm}")
print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# Print the best hyperparameters found by GridSearchCV
print("\nBest SVM Hyperparameters:")
print(grid.best_params_)