"""
Klasyfikacja za pomocą Drzewa Decyzyjnego dla Oceny Graczy NBA

Autor: Mikołaj Prętki; Mikołaj Hołdakowski

Ten skrypt implementuje klasyfikator Drzewa Decyzyjnego do oceny graczy NBA na podstawie wybranych cech. Zmienną docelową jest 'net_rating', sklasyfikowana jako pozytywna lub negatywna.

Funkcje:
1. preprocess_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series]
   - Wczytuje dane graczy NBA z pliku CSV.
   - Przetwarza dane, wybierając konkretne cechy i tworząc zmienną docelową.
   - Zwraca krotkę zawierającą cechy (X) i zmienną docelową (y).

2. train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier
   - Trenuje klasyfikator Drzewa Decyzyjnego na dostarczonych danych treningowych.
   - Zwraca wytrenowany model Drzewa Decyzyjnego.

3. evaluate_decision_tree(model: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None
   - Ocenia model Drzewa Decyzyjnego na zestawie testowym.
   - Wyświetla dokładność i macierz pomyłek.

4. visualize_decision_tree(model: DecisionTreeClassifier, feature_names: List[str]) -> None
   - Wizualizuje model Drzewa Decyzyjnego za pomocą funkcji plot_tree.
   - Zapisuje wizualizację jako plik graficzny.

Użycie:
1. Wczytaj dane: X, y = preprocess_data('all_seasons.csv')
2. Podziel dane: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
3. Trenuj model: decisionTree = train_decision_tree(X_train, y_train)
4. Oceń model: evaluate_decision_tree(decisionTree, X_test, y_test)
5. Zwizualizuj drzewo: visualize_decision_tree(decisionTree, feature_names=['age', 'player_height', 'player_weight', 'gp', 'pts', 'reb', 'ast'])

Referencje:
- Decision Tree Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Funkcja do wczytywania danych graczy NBA z pliku CSV i przetwarzania ich.

    Parameters:
    - file_path (str): Ścieżka do pliku CSV z danymi.

    Returns:
    - pd.DataFrame: Ramka danych zawierająca przetworzone dane.
    """
    data = pd.read_csv(file_path)
    selected_features = ['age', 'player_height', 'player_weight', 'gp', 'pts', 'reb', 'ast']
    X = data[selected_features]
    y = data['net_rating'].apply(lambda x: 'positive' if x > 0 else 'negative')
    return X, y

def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """
    Funkcja do trenowania klasyfikatora Drzewa Decyzyjnego na danych treningowych.

    Parameters:
    - X_train (pd.DataFrame): Cechy treningowe.
    - y_train (pd.Series): Zmienna docelowa treningowa.

    Returns:
    - DecisionTreeClassifier: Wytrenowany model Drzewa Decyzyjnego.
    """
    decisionTree = DecisionTreeClassifier(random_state=8)
    decisionTree.fit(X_train, y_train)
    return decisionTree

def evaluate_decision_tree(model: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Funkcja do oceny modelu Drzewa Decyzyjnego na danych testowych.

    Parameters:
    - model (DecisionTreeClassifier): Wytrenowany model Drzewa Decyzyjnego.
    - X_test (pd.DataFrame): Cechy testowe.
    - y_test (pd.Series): Zmienna docelowa testowa.

    Returns:
    - None
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Wyświetlenie wyniku w formie count
    result_count = pd.Series(y_test).value_counts()
    print(result_count)

    # Macierz pomyłek
    confusion_mat = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(confusion_mat)

    # Rules
    tree_rules = export_text(model, feature_names=X_test.columns.tolist())
    print("\nDecision Tree Rules:")
    print(tree_rules)

    # Wizualizacja drzewa (opcjonalnie)
    # plt.figure(figsize=(12, 8))
    # plot_tree(model, feature_names=X_test.columns.tolist(), class_names=model.classes_.astype(str), filled=True, rounded=True)
    # plt.show()

# Przykładowe użycie
if __name__ == "__main__":
    file_path = 'all_seasons.csv'
    X, y = preprocess_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
    decisionTree = train_decision_tree(X_train, y_train)
    evaluate_decision_tree(decisionTree, X_test, y_test)
