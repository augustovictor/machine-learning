import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# Load data
iris = load_iris()

# Picking 3 entries from dataset
test_idx = [0, 50, 100]

# Training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Testing data
# Labels
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print test_target

print "Predicting:"
print clf.predict(test_data)

# TREE PDF GENERATE
from sklearn.externals.six import StringIO
import pydot

dot_data = StringIO()

tree.export_graphviz(
    clf, out_file=dot_data,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True, rounded=True,
    special_characters=True
)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_svg("iris.svg")

print test_data[2], test_target[2]

print iris.feature_names, iris.target_names

# Col names (atributes)
# print iris.feature_names

# Labels (classifiers)
# print iris.target_names

# Data row
# print iris.data[0]

# Classification
# print iris.target[50]
