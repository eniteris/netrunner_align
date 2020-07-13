import glob
import os
import sys

scans = sorted(glob.glob("scans/"+sys.argv[1]+".jpg"))
for f in scans:
	num = f.split("/")
	num = num[1].split(".")
	num = num[0]
	os.system("python3 align.py "+str(num))
