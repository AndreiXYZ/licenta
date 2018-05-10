from helpers import *

clf = joblib.load('SVMs/SVC18.pkl')

img = plt.imread('00042.ppm')
model = build_feature_extractor('best_model_acc_98_1.h5')

fullmodel = load_model('best_model_acc_98_1.h5')

font = cv2.FONT_HERSHEY_DUPLEX

(winW, winH) = (110, 110)
for y,x,image in sliding_window(img, 32, (winW, winH)):
	features = pass_pipeline(image, model)
	if clf.predict(features) == 1:
		processed_img = process_image(img)
		probability_dist = fullmodel.predict(processed_img)[0]
		probability_dict = {x:y for x,y in enumerate(probability_dist)}
		if(max(probability_dict.values()) < 0.8):
			continue
		plt.bar(probability_dict.keys(), probability_dict.values())
		plt.show()
		plt.imshow(image)
		plt.show()
		cv2.rectangle(img, (x,y), (x+winH, y+winW), (0,255,0), 3)
		cv2.putText(img, sign_dict[18], (x, y+150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

plt.imshow(img)
plt.show()