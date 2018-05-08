import matplotlib.pyplot as plt

with open("gt.txt", "r") as f:
	imgSizes = []
	for line in f.readlines():
		fields = line.split(";")
		leftCol = int(fields[1])
		topRow = int(fields[2])
		rightCol = int(fields[3])
		bottomRow = int(fields[4])
		imgSize = (rightCol - leftCol, bottomRow - topRow)
		imgSizes.append(imgSize)

sortedSizes = sorted(imgSizes)
print(sortedSizes)
plt.plot([x[0] for x in sortedSizes], [x[1] for x in sortedSizes])
plt.title('Traffic sign dimensions in images')
plt.xlabel('Width')
plt.ylabel('Height')
plt.show()