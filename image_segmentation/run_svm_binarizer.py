from helpers import *
from sklearn.svm import SVC
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

clf = SVC(kernel='linear', class_weight='balanced', verbose=True)

with open('signs.pkl', 'rb') as f:
	sign_set = pickle.load(f)

with open('backgrounds.pkl', 'rb') as f:
	background_set = pickle.load(f)

X = np.append(sign_set, background_set, axis=0)
y = np.append(np.ones((sign_set.shape[0], 1)), np.zeros((background_set.shape[0], 1)), axis=0)

clf.fit(X,y)

ypred = clf.predict(X)

print('F1 score:',f1_score(y, ypred))
