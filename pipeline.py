# import a dataset
from sklearn import datasets
iris = datasets.load_iris()

# Features ( Input )
X = iris.data

# Labels ( Output )
y = iris.target

# f(x) = y

from sklearn.cross_validation import train_test_split
# Spliting data from dataset in the train and test group. 50% for each
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# Solution 1 - DecisionTree
# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

# Solution 2 - KNeighbor
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

# print predictions

from sklearn.metrics import accuracy_score

# How accurate the prediction was
print accuracy_score(y_test, predictions)
