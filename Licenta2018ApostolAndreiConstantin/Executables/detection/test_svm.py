from helpers import *
import operator

clf = joblib.load('SVC8e-1.pkl')

img = plt.imread('00005.ppm')
imgShow = plt.imread('00005.ppm')

model = build_feature_extractor('best_model_acc_98_1.h5')

fullmodel = load_model('best_model_acc_98_1.h5')

font = cv2.FONT_HERSHEY_DUPLEX

ctr = 0
(winW, winH) = (120, 120)
for y,x,image in sliding_window(img, 64, (winW, winH)):
	ctr += 1
	features = pass_pipeline(image, model)
	if image.shape[0]!=image.shape[1]:
		continue
	if clf.predict(features) == 1:
		processed_img = process_image(image)
		probability_dist = fullmodel.predict(processed_img)[0]
		probability_dict = {x:y for x,y in enumerate(probability_dist)}
		predicted_class = max(probability_dict.items(), key=operator.itemgetter(1))[0]
		if(max(probability_dict.values()) < 0.8):
			continue
		print('Detected ', sign_dict[predicted_class], ' at x:', x+winW, ' y:', y+winH)
		cv2.rectangle(imgShow, (x,y), (x+winH, y+winW), (0,255,0), 3)
		cv2.putText(imgShow, sign_dict[predicted_class], (x, y+110), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

print(ctr)
plt.imshow(imgShow)
plt.show()