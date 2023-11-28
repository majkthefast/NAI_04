
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text, DecisionTreeClassifier, plot_tree

file_path = 'winequality-white.csv'

# Read the etiquettes from the first line
first_line = pd.read_csv(file_path, delimiter=';', nrows=0).columns.tolist()
column_names = [col.strip('"') for col in first_line]

# Read the data past first line
X = pd.read_csv(file_path, delimiter=';', skiprows=1, names=column_names)

# Create class labels from the last column
y = X['quality'].apply(lambda x: 'good' if x > 5 else 'bad')
X = X.drop(columns='quality')
print(y.value_counts())

# Split the data into train and test, create a decision tree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

decisionTree = DecisionTreeClassifier(random_state=8)
decisionTree.fit(X_train, y_train)

# Make predictions on the test set
y_pred = decisionTree.predict(X_test)

# Report accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Show the rules of the decision tree
tree_rules = export_text(decisionTree, feature_names=column_names[:-1], max_depth=30)
print("\nDecision Tree Rules:")
print(tree_rules)
#plt.figure(figsize=(84, 42))
#plot_tree(clf, feature_names=column_names[:-1], class_names=clf.classes_.astype(str), filled=True, rounded=True, fontsize=5, label="root")
#plt.savefig('decision_tree.png', format='png')
#plt.show()
