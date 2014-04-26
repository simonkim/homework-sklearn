from sklearn import datasets
from sklearn import svm
iris = datasets.load_iris()
digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100.)

print('fitting model(data/target)')
clf.fit(digits.data[:-1], digits.target[:-1])
print('predicting target of image:')
print(digits.images[-1])
print(' - sample:')
print(digits.data[-1])


prediction = clf.predict(digits.data[-1])
print(' - prediction:')
print(prediction)
print('storing model')
import pickle
s = pickle.dumps(clf)
print('loading  model back')
clf2 = pickle.loads(s)

print('predicting target of image:')
print(digits.images[-2])
print(' - sample:')
print(digits.data[-2])
prediction = clf2.predict(digits.data[-2])
print(' - prediction:')
print(prediction)
