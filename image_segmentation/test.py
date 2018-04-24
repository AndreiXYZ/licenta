from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

model = load_model('best_model_loss.h5')
img = plt.imread('r2.ppm')
img = cv2.resize(img, (32, 32))
img = np.reshape(img, [1,32,32,3])
img = img/255.0
t1 = time.time()
print(model.predict_classes(img))
print("Time elapsed:", time.time()-t1)
distribution = model.predict(img)[0]
print("Class probability distribution")
print(distribution)
distribution_dict = {x:distribution[x] for x in range(42)}
plt.bar(distribution_dict.keys(), distribution_dict.values())
plt.show()