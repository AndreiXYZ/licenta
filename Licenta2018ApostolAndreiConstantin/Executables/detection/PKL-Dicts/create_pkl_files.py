from helpers import *
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
import pickle

model = build_feature_extractor('best_model_acc_98_1.h5')

root_backgrounds = '/home/apo/Licenta/image_segmentation/background_set_multi'
root_signs = '/home/apo/Licenta/German-Augmented'

backgrounds = {'train':[], 'test':[], 'val':[]}
signs = {'train':[], 'test':[], 'val':[]}

#Building the background set
print('Building background set...')
backgrounds['train'] = extract_features(os.path.join(root_backgrounds,'train'), model)
backgrounds['test'] = extract_features(os.path.join(root_backgrounds,'test'), model)
backgrounds['val'] = extract_features(os.path.join(root_backgrounds,'val'), model)
with open('backgrounds.pkl', 'wb') as f:
	pickle.dump(backgrounds, f)

print(backgrounds['train'].shape)
print(backgrounds['test'].shape)
print(backgrounds['val'].shape)
#Building the signs set

print('Building signs set...')
for i in range(43):
	if i==0:
		signs['train'] = extract_features(os.path.join(root_signs, 'Training',sign_folders[i]), model)
		signs['test'] = extract_features(os.path.join(root_signs, 'Testing',sign_folders[i]), model)
		signs['val'] = extract_features(os.path.join(root_signs, 'Validation',sign_folders[i]), model)
	else:
		signs['train'] = np.append(signs['train'], 
									extract_features(os.path.join(root_signs, 'Training', sign_folders[i]), model),
									axis=0)
		signs['test'] = np.append(signs['test'], 
									extract_features(os.path.join(root_signs, 'Testing', sign_folders[i]), model),
									axis=0)
		signs['val'] = np.append(signs['val'], 
									extract_features(os.path.join(root_signs, 'Validation', sign_folders[i]), model),
									axis=0)


with open('signs.pkl', 'wb') as f:
	pickle.dump(signs, f)

print(signs['train'].shape)
print(signs['test'].shape)
print(signs['val'].shape)