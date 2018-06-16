import random
import os

traindir = 'train'
testdir = 'test'
valdir = 'val'

files_train = os.listdir(traindir)
files_test = os.listdir(testdir)

#move 3000 from training set
# for i in range(3000):
# 	idx = random.randint(0,len(files_train))
# 	os.rename(os.path.join(traindir, files_train[idx]),
# 		  os.path.join(valdir, files_train[idx]))
# 	del files_train[idx]
# 	print(idx)

#move 3000 from testing set
for i in range(3000):
	idx = random.randint(0,len(files_test))
	os.rename(os.path.join(testdir, files_test[idx]),
		  os.path.join(valdir, files_test[idx]))
	del files_test[idx]
	print(idx)