from helpers import *
import operator
from itertools import combinations

def detect_signs(img, feature_extractor, full_model, 
				 classifier, winW, winH, stride):
	'''Applies sliding window of size (winW, winH) over image to
	   detect and classify traffic signs.
	   Returns the image with labeled bounding boxes, total number of windows
	   and the coordinates of the detected signs along with their class index.
	'''
	rects = []
	font = cv2.FONT_HERSHEY_DUPLEX
	ctr = 0
	#create a copy of the image. (bounding boxes may interfere with detection)
	imgShow = np.array(img, copy=True)

	for y,x,image in sliding_window(img, stride, (winW, winH)):
		ctr += 1
		#perform required transformations and pass it to the cnn
		#to extract conv featuress
		features = pass_pipeline(image, feature_extractor)
		if image.shape[0] != image.shape[1]:
			continue
		#if the svm predicts a sign, threshold on the neural network
		#probability and add the detection to the rectangles list
		if classifier.predict(features) == 1:
			processed_img = process_image(image)
			probability_dist = full_model.predict(processed_img)[0]
			probability_dict = {x:y for x,y in enumerate(probability_dist)}
			if max(probability_dict.values()) < 0.8:
				continue
			predicted_class = max(probability_dict.items(), key=operator.itemgetter(1))[0]
			rects.append((x, y, x+winW, x+winH, predicted_class))

	#remove overlapping rectangles
	for r1, r2 in combinations(rects, 2):
		#if they have different classes, skip
		if r1[4]!=r2[4]:
			continue
		#if they have been already removed, skip
		if r1 not in rects or r2 not in rects:
			continue
		#if no overlap, skip
		if r1[0] > r2[0]+winW or r1[0]+winW < r2[0] or r1[1]+winH < r2[1] or r1[1] > r2[1]+winH:
			continue
		#remove overlapping rectangles and add the one
		#that results from merging them
		if r1 in rects:
			rects.remove(r1)
		if r2 in rects:
			rects.remove(r2)
		rects.append((min(r1[0], r2[0]), min(r1[1],r2[1]),
					  max(r1[2], r2[2]), max(r1[3],r2[3]),
					  r1[4]))
	#draw bounding boxes and add text
	for elem in rects:
		cv2.rectangle(imgShow, (elem[0],elem[1]), (elem[0]+winH, elem[1]+winW), (0,255,0), 3)
		cv2.putText(imgShow, sign_dict[elem[4]], (elem[0], elem[1]), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
	return imgShow, ctr, rects


if __name__ == "__main__":
	clf = joblib.load('SVC8e-1.pkl')
	extractor = build_feature_extractor('best_model_acc_98_1.h5')
	fullmodel = load_model('best_model_acc_98_1.h5')
	
	img1 = plt.imread('00005.ppm')
	img2 = plt.imread('00006.ppm')
	img3 = plt.imread('00628.ppm')
	img4 = plt.imread('00871.ppm')

	imgShow, ctr, rects = detect_signs(img1, feature_extractor=extractor, full_model=fullmodel, classifier=clf,
			 			 winW=180, winH=180, stride=64)
	print('Total windows=', ctr)
	print('Total signs detected=', rects)
	plt.imshow(imgShow)
	plt.show()

	imgShow, ctr, rects = detect_signs(img2, feature_extractor=extractor, full_model=fullmodel, classifier=clf,
			 			 winW=110, winH=110, stride=64)
	print('Total windows=', ctr)
	print('Total signs detected=', rects)
	plt.imshow(imgShow)
	plt.show()

	imgShow, ctr, rects = detect_signs(img3, feature_extractor=extractor, full_model=fullmodel, classifier=clf,
			 			 winW=100, winH=100, stride=40)
	print('Total windows=', ctr)
	print('Total signs detected=', rects)
	plt.imshow(imgShow)
	plt.show()

	imgShow, ctr, rects = detect_signs(img4, feature_extractor=extractor, full_model=fullmodel, classifier=clf,
			 			 winW=48, winH=48, stride=16)
	print('Total windows=', ctr)
	print('Total signs detected=', rects)
	plt.imshow(imgShow)
	plt.show()