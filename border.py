from matplotlib import pyplot as plt
import cv2
import numpy as np
import sys


BORDER_BLEED = 25 + 100
WHITE_CUTOFF = 200

cardnum = str(sys.argv[1])

align_f = "aligned/"+cardnum+".jpg"
out_f = "border/"+cardnum+".jpg"

align_img = cv2.imread(align_f, cv2.IMREAD_COLOR)

ow, oh, channels = align_img.shape
HB = int(0.06936*oh/2)
WB = int(0.05950*ow/2)

BORDER_BLEED = 25

align_img = np.pad(align_img,((HB,HB),(WB,WB),(0,0)),'constant',constant_values=255)
print(align_img.shape)

ow, oh, channels = align_img.shape
mask = np.zeros((ow, oh),dtype=np.uint8);

for i in range(0,BORDER_BLEED+HB):
	for y in range(0,oh):
		if(align_img[i,y,0] > WHITE_CUTOFF and align_img[i,y,1] > WHITE_CUTOFF and align_img[i,y,2] > WHITE_CUTOFF):
			mask[i,y] = 255
		if(align_img[ow-i-1,y,0] > WHITE_CUTOFF and align_img[ow-i-1,y,1] > WHITE_CUTOFF and align_img[ow-i-1,y,2] > WHITE_CUTOFF):
			mask[ow-i-1,y] = 255

for i in range(0,BORDER_BLEED+WB):
	for x in range(0,ow):
		if(align_img[x,i,0] > WHITE_CUTOFF and align_img[x,i,1] > WHITE_CUTOFF and align_img[x,i,2] > WHITE_CUTOFF):
			mask[x,i] = 255
		if(align_img[x,oh-i-1,0] > WHITE_CUTOFF and align_img[x,oh-1-i,1] > WHITE_CUTOFF and align_img[x,oh-1-i,2] > WHITE_CUTOFF):
			mask[x,oh-1-i] = 255

kernel = np.ones((3,3),np.uint8)
mask = cv2.dilate(mask,kernel,iterations=3)
#imReg = cv2.inpaint(align_img,mask,3,cv2.INPAINT_TELEA)
imReg = cv2.inpaint(align_img,mask,3,cv2.INPAINT_NS)

cv2.imwrite(out_f, imReg)


plt.figure("mask")
plt.imshow(mask)
#plt.figure("orig")
#plt.imshow(align_img[...,::-1])
#plt.figure("filled")
#plt.imshow(imReg[...,::-1])
plt.show()
