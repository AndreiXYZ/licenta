from helpers import *
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

train_accs = []
test_accs = []
model = build_feature_extractor('best_model_acc_98_1.h5')

root_signs_train = '/home/apo/Licenta/German-Augmented/Training'
root_signs_test = '/home/apo/Licenta/German-Augmented/Testing'

background_set_train = extract_features('background_set_multi/train', model)
background_set_test = extract_features('background_set_multi/test', model)

for i in range(43):
	#extract that sign's conv features
	sign_train_folder = os.path.join(root_signs_train, sign_folders[i])
	sign_set_train = extract_features(sign_train_folder, model)

	#define train and test tensor
	X_train = np.append(background_set_train, sign_set_train, axis=0)
	y_train = np.append(np.zeros((background_set_train.shape[0], 1)), np.ones((sign_set_train.shape[0], 1)), axis=0)

	#create the model and train it
	clf = SVC(kernel='linear', class_weight='balanced', verbose=True)
	clf.fit(X_train, y_train)

	#get predictions and check training accuracy
	y_pred = clf.predict(X_train)
	confusion_train = confusion_matrix(y_train, y_pred)
	train_acc = (confusion_train[0][0] + confusion_train[1][1]) / confusion_train.sum()
	train_accs.append(train_acc)
	
	#now run it on the test set
	#we have to build our X_test and y_test first, just like before
	sign_test_folder = os.path.join(root_signs_test, sign_folders[i])
	sign_set_test = extract_features(sign_test_folder, model)

	X_test = np.append(background_set_test, sign_set_test, axis=0)
	y_test = np.append(np.zeros((background_set_test.shape[0], 1)), np.ones((sign_set_test.shape[0], 1)))

	y_test_pred = clf.predict(X_test)
	confusion_test = confusion_matrix(y_test, y_test_pred)
	test_acc = (confusion_test[0][0] + confusion_test[1][1]) / confusion_test.sum()
	test_accs.append(test_acc)

	#finally, save our model to disk
	model_name = 'SVC' + str(i) + '.pkl'
	joblib.dump(clf, model_name)


#print train and test (average) accuracy
print('Train accuracies:')
print('avg=', sum(train_accs)/len(train_accs))
print('vals=', train_accs)

print('Test accuracies:')
print('avg=', sum(test_accs)/len(test_accs))
print('vals=', test_accs)