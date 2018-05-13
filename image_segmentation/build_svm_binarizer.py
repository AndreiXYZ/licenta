from helpers import *
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
import pickle

# #build background set
# model = build_feature_extractor('best_model_acc_98_1.h5')

# print('Building background set...')
# background_set = extract_features('background_set_multi/train', model)

# print('Background set size: ', background_set.shape)
# #build signs set
# root_signs_train = '/home/apo/Licenta/German-Augmented/Training/'

# print('Building signs set...')
# for i in range(43):
# 	if i==0:
# 		sign_set = extract_features(os.path.join(root_signs_train, sign_folders[i]), model)
# 	else:
# 		sign_set = np.append(sign_set, extract_features(os.path.join(root_signs_train, sign_folders[i]), model), axis=0)

# print('Sign set size:', sign_set.shape)

# #save them as pkl files
# with open('signs.pkl', 'wb') as f:
# 	pickle.dump(sign_set, f)

# with open('backgrounds.pkl', 'wb') as f:
# 	pickle.dump(background_set, f)

with open('backgrounds.pkl', 'rb') as f:
	background_set = pickle.load(f)

with open('signs.pkl', 'rb') as f:
	sign_set = pickle.load(f)

#build X and y for our classifier
X = np.append(background_set, sign_set, axis=0)
y = np.append(np.zeros((background_set.shape[0], 1)), np.ones((sign_set.shape[0], 1)))

del sign_set
del background_set

print('Training svm...')
#build our classifier, train it and see how it performs
clf = SVC(kernel='linear', class_weight='balanced', verbose=True)
clf.fit(X, y)

y_pred = clf.predict(X)

print('Model score=', f1_score(y, y_pred))
print('Confusion matrix=', confusion_matrix(y, y_pred))

#finally, save our model
joblib.dump(clf, 'SVMbinarizer.pkl')