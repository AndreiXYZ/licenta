from helpers import *

clf = joblib.load('SVC18.pkl')

img = plt.imread('00042.ppm')
model = build_feature_extractor('best_model_acc_98_1.h5')

font = cv2.FONT_HERSHEY_DUPLEX

(winW, winH) = (100, 100)
for y,x,image in sliding_window(img, 32, (winW, winH)):
	features = pass_pipeline(image, model)
	if clf.predict(features) == 1:
		plt.imshow(image)
		plt.show()
		cv2.rectangle(img, (x,y), (x+winH, y+winW), (0,255,0), 3)
		cv2.putText(img, sign_dict[18], (x, y+150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

plt.imshow(img)
plt.show()