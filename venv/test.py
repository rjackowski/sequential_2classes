import numpy as np
from sklearn.naive_bayes import CategoricalNB
# from scikit-learn import CategoricalNB
X_train = np.array([[0, 1], [0, 1], [0, 1]])
y_train = np.array([0, 0, 1])
clf = CategoricalNB(min_categories=[3,3])
# clf = CategoricalNB()
clf.fit(X_train, y_train)
X_test = np.array([[2,2]])
a = clf.predict_proba(X_test)

print(clf.classes_)
# clf.n_categories_
