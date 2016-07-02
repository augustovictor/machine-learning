from sklearn import tree

features = [
    [140, 1], # smooth
    [130, 1], # smooth
    [150, 0], # bumpy
    [170, 0]  # bumpy
]

labels = [
    0, # apple
    0, # apple
    1, # orange
    1  # orange
]

clf = tree.DecisionTreeClassifier()

# fit = find patterns and data
clf = clf.fit(features, labels)

# testing
print clf.predict([[160, 0]])
