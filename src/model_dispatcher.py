from sklearn import ensemble
from sklearn import tree

models = {
    'decision_tree':tree.DecisionTreeClassifier(),
    'rf':ensemble.RandomForestClassifier()
}
