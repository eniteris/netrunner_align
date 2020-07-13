import glob
import os
import sys

scans = sorted(glob.glob("aligned/"+sys.argv[1]+".jpg"))
for f in scans:
	num = f.split("/")
	num = num[1].split(".")
	num = num[0]
	print(num)
	os.system("python3 border.py "+str(num))
