"""
Klasyfikator drzewa decyzyjnego do przewidywania jakości wina

Autor: Mikołaj Prętki; Mikołaj Hołdakowski
Opis problemu: Ten skrypt używa klasyfikatora drzewa decyzyjnego do przewidzenia jakości wina na podstawie cech.
Użycie: python nazwa_skryptu.py

Zależności:
- pandas
- matplotlib
- scikit-learn

Instrukcje:
1. Upewnij się, że zainstalowane są wymagane zależności.
2. Podaj poprawną ścieżkę do pliku 'winequality-white.csv'.
3. Uruchom skrypt.

Opcjonalne referencje:
- Drzewa decyzyjne w scikit-learn: https://scikit-learn.org/stable/modules/tree.html
- Dokumentacja pandas: https://pandas.pydata.org/pandas-docs/stable/index.html
- Dokumentacja matplotlib: https://matplotlib.org/stable/contents.html
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text, DecisionTreeClassifier, plot_tree

def read_data(file_path):
    """Odczytuje dane z pliku CSV."""
    first_line = pd.read_csv(file_path, delimiter=';', nrows=0).columns.tolist()
    column_names = [col.strip('"') for col in first_line]
    X = pd.read_csv(file_path, delimiter=';', skiprows=1, names=column_names)
    return X, column_names

def create_labels(X):
    """Tworzy etykiety klas ('dobrze'/'źle') na podstawie jakości wina."""
    y = X['quality'].apply(lambda x: 'dobre' if x > 5 else 'złe')
    X = X.drop(columns='quality')
    return X, y

def train_decision_tree(X_train, y_train):
    """Trenuje klasyfikator drzewa decyzyjnego."""
    decisionTree = DecisionTreeClassifier(random_state=8)
    decisionTree.fit(X_train, y_train)
    return decisionTree

def evaluate_model(model, X_test, y_test):
    """Ocenia dokładność klasyfikatora na zestawie testowym."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Dokładność: {accuracy}")

def main():
    """Główna funkcja programu."""
    file_path = 'winequality-white.csv'

    # Odczytaj dane z pliku
    X, column_names = read_data(file_path)

    # Stwórz etykiety klas
    X, y = create_labels(X)
    print(y.value_counts())

    # Podziel dane na zestawy treningowy i testowy, stwórz drzewo decyzyjne
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
    decisionTree = train_decision_tree(X_train, y_train)

    # Ocen dokładność modelu
    evaluate_model(decisionTree, X_test, y_test)

    # Pokaż zasady drzewa decyzyjnego
    tree_rules = export_text(decisionTree, feature_names=column_names[:-1], max_depth=30)
    print("\nZasady drzewa decyzyjnego:")
    print(tree_rules)

    # Odkomentuj poniższe linie, aby zwizualizować drzewo decyzyjne
    # plt.figure(figsize=(84, 42))
    # plot_tree(decisionTree, feature_names=column_names[:-1], class_names=decisionTree.classes_.astype(str), filled=True, rounded=True, fontsize=5, label="root")
    # plt.savefig('decision_tree.png', format='png')
    # plt.show()

if __name__ == "__main__":
    main()
