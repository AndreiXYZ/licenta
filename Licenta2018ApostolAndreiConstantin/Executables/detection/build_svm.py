from helpers import *
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
import pickle

with open('PKL-Dicts/signs.pkl', 'rb') as f:
	signs = pickle.load(f)

with open('PKL-Dicts/backgrounds.pkl', 'rb') as f:
	backgrounds = pickle.load(f)

#build x and y training
X = np.append(backgrounds['train'], signs['train'], axis=0)
y = np.append(np.zeros((backgrounds['train'].shape[0], 1)), np.ones((signs['train'].shape[0], 1)))

print('Total set size of', X.shape[0], ' of which ', signs['train'].shape[0], ' are signs.')

clf = SVC(kernel='linear', class_weight='balanced', verbose=True, C=0.8)
clf.fit(X, y)
y_pred = clf.predict(X)

#test results using confusion matrix on training set
print('Model score on train set:', f1_score(y,y_pred))
confusion_mat = confusion_matrix(y, y_pred)
print('Train acc.:', clf.score(X, y))
print('Confusion matrix for trainig set:')
print(confusion_mat)


#now run confusion matrix on validation set
#we proceed in a similar fashion to the training set

print('-'*30)
Xval = np.append(backgrounds['val'], signs['val'], axis=0)
yval = np.append(np.zeros((backgrounds['val'].shape[0],1)), np.ones((signs['val'].shape[0], 1)))
y_pred_val = clf.predict(Xval)
print('Total val set size of ', Xval.shape[0], ' of which ', signs['val'].shape[0], ' are signs.')
print('F1-score on test set: ', f1_score(yval, y_pred_val))
confusion_mat_val = confusion_matrix(yval, y_pred_val)

print('Test acc.:', clf.score(Xval, yval))
print('Confusion matrix for val set:')
print(confusion_mat_val)

#now the test set
print('-'*30)
Xtest = np.append(backgrounds['test'], signs['test'], axis=0)
ytest = np.append(np.zeros((backgrounds['test'].shape[0], 1)), np.ones((signs['test'].shape[0], 1)))
y_pred_test = clf.predict(Xtest)

print('Total test set size of', Xtest.shape[0], ' of which ', signs['test'].shape[0], ' are signs.')

print('F1-score on test set: ', f1_score(ytest, y_pred_test))
confusion_mat_test = confusion_matrix(ytest, y_pred_test)

print('Test acc.:', clf.score(Xtest, ytest))
print('Confusion matrix for test set:')
print(confusion_mat_test)
joblib.dump(clf, 'SVC8e-1.pkl')