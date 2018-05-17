from helpers import *
import pickle

model = build_feature_extractor('best_model_acc_98_1.h5')
root_signs_train = '/home/apo/Licenta/German-Augmented/Training/'
root_signs_test = '/home/apo/Licenta/German-Augmented/Testing/'

#building signs train set:
for i in range(43):
	if i==0:
		sign_set = extract_features(os.path.join(root_signs_train, sign_folders[i]), model)
	else:
		sign_set = np.append(sign_set, extract_features(os.path.join(root_signs_train, sign_folders[i]), model), axis=0)

print("Sign set train shape:",sign_set.shape)

with open("signs_train.pkl", "wb") as f:
	pickle.dump(sign_set, f)

#build signs test set:
for i in range(43):
	if i==0:
		sign_set_test = extract_features(os.path.join(root_signs_test, sign_folders[i]), model)
		y_test = np.zeros()
	else:
		sign_set_test = np.append(sign_set_test, extract_features(os.path.join(root_signs_test, sign_folders[i]), model), axis=0)

print("Sign set test shape:", sign_set_test.shape)
with open("signs_test.pkl", "wb") as f:
	pickle.dump(sign_set_test, f)