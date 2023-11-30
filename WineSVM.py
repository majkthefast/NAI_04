"""
Klasyfikator maszyny wektorów nośnych do przewidywania jakości wina

Autor: Mikołaj Prętki; Mikołaj Hołdakowski
Opis problemu: Ten skrypt używa klasyfikatora maszyny wektorów nośnych (SVM) do przewidzenia jakości wina na podstawie cech.
Użycie: python nazwa_skryptu.py

Zależności:
- pandas
- scikit-learn

Instrukcje:
1. Upewnij się, że zainstalowane są wymagane zależności.
2. Podaj poprawną ścieżkę do pliku 'winequality-white.csv'.
3. Uruchom skrypt.

Opcjonalne referencje:
- SVM w scikit-learn: https://scikit-learn.org/stable/modules/svm.html
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

def read_data(file_path):
    """Odczytuje dane z pliku CSV."""
    first_line = pd.read_csv(file_path, delimiter=';', nrows=0).columns.tolist()
    column_names = [col.strip('"') for col in first_line]
    X = pd.read_csv(file_path, delimiter=';', skiprows=1, names=column_names)
    return X, column_names

def create_labels(X):
    """Tworzy etykiety klas ('dobre'/'złe') na podstawie jakości wina."""
    y = X['quality'].apply(lambda x: 'good' if x > 5 else 'bad')
    X = X.drop(columns='quality')
    return X, y

def train_svm(X_train, y_train):
    """Trenuje klasyfikator maszyny wektorów nośnych (SVM)."""
    svc = SVC(kernel='rbf', C=1, gamma=100)
    svc.fit(X_train, y_train)
    return svc

def evaluate_svm(model, X_test, y_test):
    """Ocenia klasyfikator SVM i wyświetla dokładność oraz macierz pomyłek."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Dokładność: {accuracy}")
    print("\nMacierz Pomyłek:")
    print(confusion_matrix(y_test, y_pred))

def main():
    """Główna funkcja programu."""
    file_path = 'winequality-white.csv'

    # Odczytaj dane z pliku
    X, column_names = read_data(file_path)

    # Stwórz etykiety klas
    X, y = create_labels(X)
    print(y.value_counts())

    # Podziel dane na zestawy treningowy i testowy, stwórz SVM
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
    svc_model = train_svm(X_train, y_train)

    # Przewiń prognozy i raportuj dokładność, macierz pomyłek
    evaluate_svm(svc_model, X_test, y_test)

if __name__ == "__main__":
    main()
