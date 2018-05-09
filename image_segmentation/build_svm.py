from helpers import *
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

model = build_feature_extractor('best_model_acc_98_1.h5')
background_set = extract_features('background_set_multi', model)

#build from entire dataset (train+test+validation)
sign_set_train = extract_features('/home/apo/Licenta/German-Augmented/Training/00018', model)
sign_set_test = extract_features('/home/apo/Licenta/German-Augmented/Testing/00018', model)
sign_set_validation = extract_features('/home/apo/Licenta/German-Augmented/Validation/00018', model)

sign_set = np.append(sign_set_train, sign_set_validation, axis=0)
sign_set = np.append(sign_set, sign_set_test, axis=0)
print(sign_set.shape)
print(background_set)
#build x and y training
X = np.append(background_set, sign_set, axis=0)
y = np.append(np.zeros((background_set.shape[0], 1)), np.ones((sign_set.shape[0], 1)))
print(y.shape)
print(X.shape)

clf = SVC(kernel='linear', class_weight='balanced', verbose=True)
clf.fit(X, y)
y_pred = clf.predict(X)

#test results using confusion matrix
unique, counts = np.unique(clf.predict(X), return_counts=True)
print('\n',dict(zip(unique, counts)))
confusion_mat = confusion_matrix(y, y_pred)
print('\nConfusion matrix\n',confusion_mat)
joblib.dump(clf, 'SVC18.pkl')