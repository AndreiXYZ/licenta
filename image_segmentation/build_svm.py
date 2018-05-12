from helpers import *
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
model = build_feature_extractor('best_model_acc_98_1.h5')
background_set = extract_features('background_set_multi/train', model)

root_signs_train = '/home/apo/Licenta/German-Augmented/Training/'
root_signs_test = '/home/apo/Licenta/German-Augmented/Testing/'
#add all other signs into background set
for i in range(43):
	if i==18:
		continue
	other_sign_set = extract_features(os.path.join(root_signs_train, sign_folders[i]), model)
	background_set = np.append(background_set, other_sign_set, axis=0)

#build positive samples set
sign_set = extract_features('/home/apo/Licenta/German-Augmented/Training/00018', model)

#build x and y training
X = np.append(background_set, sign_set, axis=0)
y = np.append(np.zeros((background_set.shape[0], 1)), np.ones((sign_set.shape[0], 1)))

print('Total set size of', X.shape[0], ' of which ', sign_set.shape[0], ' are signs.')

clf = SVC(kernel='linear', class_weight='balanced', verbose=True)
clf.fit(X, y)
y_pred = clf.predict(X)

#test results using confusion matrix on training set
print('Model score on train set:', f1_score(y,y_pred))
confusion_mat = confusion_matrix(y, y_pred)
print('Train acc.:', clf.score(X, y))
print('Confusion matrix for trainig set:')
print(confusion_mat)

#now run confusion matrix on test set
background_set_test = extract_features('background_set_multi/test', model)
#we proceed in a similar fashion to the training set
for i in range(43):
	if i==18:
		continue
	other_sign_set = extract_features(os.path.join(root_signs_test, sign_folders[i]), model)
	background_set_test = np.append(background_set_test, other_sign_set, axis=0)

sign_set_test = extract_features('/home/apo/Licenta/German-Augmented/Testing/00018', model)

Xtest = np.append(background_set_test, sign_set_test, axis=0)
ytest = np.append(np.zeros((background_set_test.shape[0], 1)), np.ones((sign_set_test.shape[0], 1)))
y_pred_test = clf.predict(Xtest)

print('Total test set size of', Xtest.shape[0], ' of which ', sign_set_test.shape[0], ' are signs.')

print('F1-score on test set: ', f1_score(ytest, y_pred_test))
confusion_mat_test = confusion_matrix(ytest, y_pred_test)

print('Test acc.:', clf.score(Xtest, ytest))
print('Confusion matrix for test set:')
print(confusion_mat_test)
joblib.dump(clf, 'SVC18.pkl')