from helpers import *

#read image
img = plt.imread('00042.ppm')
#create a copy in order to place rectangles and text on it
img_copy = plt.imread('00042.ppm')

#load our feature extractor and full model (used for possible thresholding)
model = build_feature_extractor('best_model_acc_98_1.h5')
fullmodel = load_model('best_model_acc_98_1.h5')

#load all the SVMs
svmdir = '/home/apo/Licenta/image_segmentation/SVMs'
SVMs = []
for i in range(43):
	current_svm_path = os.path.join(svmdir,'SVC' + str(i) + '.pkl')
	current_svm = joblib.load(current_svm_path)
	SVMs.append(current_svm)

#define font
font = cv2.FONT_HERSHEY_DUPLEX

#run sliding window and all SVMs on the image
(winW, winH) = (110, 110)
for y,x,image in sliding_window(img, 32, (winW, winH)):
	features = pass_pipeline(image, model)
	for i,svm in enumerate(SVMs):
		if svm.predict(features) == 1:
			processed_img = process_image(img)
			probability_dist = fullmodel.predict(processed_img)[0]
			probability_dict = {x:y for x,y in enumerate(probability_dist)}
			# if(max(probability_dict.values()) < 0.86):
			# 	continue
			# plt.bar(probability_dict.keys(), probability_dict.values())
			# plt.show()
			# plt.imshow(image)
			# plt.show()
			cv2.rectangle(img_copy, (x,y), (x+winH, y+winW), (0,255,0), 3)
			cv2.putText(img_copy, sign_dict[i], (x, y+150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

plt.imshow(img_copy)
plt.show()