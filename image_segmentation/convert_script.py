import os
import sys

os.chdir(sys.argv[1])
os.getcwd()
for elem in os.listdir():
	name = elem.split('.')[0]
	name = name+".ppm"
	os.system('convert ' + elem + " " + name)
	os.system('rm ' + elem)