from helpers import *
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
import pickle

clf = joblib.load('SVMbinarizer.pkl')
#build background set
model = build_feature_extractor('best_model_acc_98_1.h5')

print('Building background set...')
background_set_test = extract_features('background_set_multi/test', model)

print('Background set size: ', background_set_test.shape)
#build signs set
root_signs_test = '/home/apo/Licenta/German-Augmented/Testing/'

print('Building signs set...')
for i in range(43):
	if i==0:
		sign_set_test = extract_features(os.path.join(root_signs_test, sign_folders[i]), model)
	else:
		sign_set_test = np.append(sign_set_test, extract_features(os.path.join(root_signs_test, sign_folders[i]), model), axis=0)

print('Sign set size:', sign_set_test.shape)

#save them as pkl files
with open('signs_test.pkl', 'wb') as f:
	pickle.dump(sign_set_test, f)

with open('backgrounds_test.pkl', 'wb') as f:
	pickle.dump(background_set_test, f)

#test classifier
Xtest = np.append(background_set_test, sign_set_test, axis=0)
ytest = np.append(np.zeros((background_set_test.shape[0], 1)), np.ones((sign_set_test.shape[0], 1)))

y_test_pred = clf.predict(Xtest)
print('Model score=', f1_score(ytest, y_test_pred))
print('Confusion matrix=', confusion_matrix(ytest, y_test_pred))