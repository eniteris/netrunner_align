from __future__ import print_function
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import urllib.request

MAX_FEATURES = 100000
GOOD_MATCH_PERCENT = 0.01	#0.15
BORDER_BLEED = 25
BLACK_CUTOFF = 50
WHITE_CUTOFF = 250
WIDTH = 1515
HEIGHT = 2121

cardnum = str(sys.argv[1])

refFilename = "nrdb/orig/"+cardnum+".jpg"
imFilename = "scans/"+cardnum+".jpg"
upFilename = "nrdb/upscale/"+cardnum+".jpg"
outFilename = "aligned/"+cardnum+".jpg"

frFilename = imFilename #might not be needed

def alignImages(im1, im2):

	# Convert images to grayscale
	im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
	
	# Detect ORB features and compute descriptors.
	orb = cv2.ORB_create(MAX_FEATURES)
	keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
	keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
	
	# Match features.
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(descriptors1, descriptors2, None)
	
	# Sort matches by score
	matches.sort(key=lambda x: x.distance, reverse=False)

	# Remove not so good matches
	numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
	matches = matches[:numGoodMatches]

	# Draw top matches
	imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
	cv2.imwrite("matches/"+cardnum+".jpg", imMatches)
	
	# Extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = keypoints1[match.queryIdx].pt
		points2[i, :] = keypoints2[match.trainIdx].pt
	
	# Find homography
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

	# Use homography
	height, width, channels = im2.shape
	im1Reg = cv2.warpPerspective(im1, h, (width, height))
	
	return im1Reg, h

def calculate_cdf(histogram):
	"""
	This method calculates the cumulative distribution function
	:param array histogram: The values of the histogram
	:return: normalized_cdf: The normalized cumulative distribution function
	:rtype: array
	"""
	# Get the cumulative sum of the elements
	cdf = histogram.cumsum()
 
	# Normalize the cdf
	normalized_cdf = cdf / float(cdf.max())
 
	return normalized_cdf
 
def calculate_lookup(src_cdf, ref_cdf):
	"""
	This method creates the lookup table
	:param array src_cdf: The cdf for the source image
	:param array ref_cdf: The cdf for the reference image
	:return: lookup_table: The lookup table
	:rtype: array
	"""
	lookup_table = np.zeros(256)
	lookup_val = 0
	for src_pixel_val in range(len(src_cdf)):
		lookup_val
		for ref_pixel_val in range(len(ref_cdf)):
			if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
				lookup_val = ref_pixel_val
				break
		lookup_table[src_pixel_val] = lookup_val
	return lookup_table
 
def match_histograms(src_image, ref_image):
	"""
	This method matches the source image histogram to the
	reference signal
	:param image src_image: The original source image
	:param image  ref_image: The reference image
	:return: image_after_matching
	:rtype: image (array)
	"""
	# Split the images into the different color channels
	# b means blue, g means green and r means red
	src_b, src_g, src_r = cv2.split(src_image)
	ref_b, ref_g, ref_r = cv2.split(ref_image)
 
	# Compute the b, g, and r histograms separately
	# The flatten() Numpy method returns a copy of the array c
	# collapsed into one dimension.
	src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0,256])
	src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0,256])
	src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0,256])	
	ref_hist_blue, bin_3 = np.histogram(ref_b.flatten(), 256, [0,256])	
	ref_hist_green, bin_4 = np.histogram(ref_g.flatten(), 256, [0,256])
	ref_hist_red, bin_5 = np.histogram(ref_r.flatten(), 256, [0,256])
 
	# Compute the normalized cdf for the source and reference image
	src_cdf_blue = calculate_cdf(src_hist_blue)
	src_cdf_green = calculate_cdf(src_hist_green)
	src_cdf_red = calculate_cdf(src_hist_red)
	ref_cdf_blue = calculate_cdf(ref_hist_blue)
	ref_cdf_green = calculate_cdf(ref_hist_green)
	ref_cdf_red = calculate_cdf(ref_hist_red)
 
	# Make a separate lookup table for each color
	blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
	green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
	red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)
 
	# Use the lookup function to transform the colors of the original
	# source image
	blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
	green_after_transform = cv2.LUT(src_g, green_lookup_table)
	red_after_transform = cv2.LUT(src_r, red_lookup_table)
 
	# Put the image back together
	image_after_matching = cv2.merge([
		blue_after_transform, green_after_transform, red_after_transform])
	image_after_matching = cv2.convertScaleAbs(image_after_matching)
 
	return image_after_matching


if __name__ == '__main__':
#download from nrdb
	refFilename = "nrdb/orig/"+cardnum+".jpg"
	if(os.path.isfile(outFilename) and len(sys.argv) == 2):
		print(cardnum+"\t"+"aligned")
		exit()

	if(os.path.isfile(refFilename)):
		print(cardnum+"\t"+"exists")
	else:
		url = "https://netrunnerdb.com/card_image/large/"+cardnum+".jpg"
		try:
			a = urllib.request.urlopen(url)
		except: 
			print(cardnum+"\t"+"404")
			exit()

		urllib.request.urlretrieve(url,refFilename)

# Read images
	print("Reading image to align : ", frFilename);	
	fr = cv2.imread(frFilename, cv2.IMREAD_COLOR)
	im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
#	im = cv2.bilateralFilter(im, 15, 75, 75) 

	if(os.path.isfile(upFilename)):
		print(cardnum+"\t"+"upscaled")
		imReference = cv2.imread(upFilename, cv2.IMREAD_COLOR)
	else:
# Rescale Image
		print("Reading reference image : ", refFilename)
		imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
		imReference = cv2.resize(imReference,(WIDTH,HEIGHT))
		cv2.imwrite(upFilename, imReference)

	ow, oh, channels = imReference.shape

# Aligning Image	
	print("Aligning images ...")
	# Registered image will be resotred in imReg. 
	# The estimated homography will be stored in h. 
	imReg, h = alignImages(fr, imReference)
	ow, oh, channels = imReference.shape

# Warp input image to homography
	imReg = cv2.warpPerspective(im,h,(oh,ow))
	warp = np.copy(imReg)

#Adjust Histograms
	imReference = match_histograms(imReference,imReg)

# Fill black border scan with template
	mask = np.zeros((ow, oh),dtype=np.uint8);
	for i in range(0,BORDER_BLEED):
		for y in range(0,oh):
			if(imReg[i,y,0] < BLACK_CUTOFF and imReg[i,y,1] < BLACK_CUTOFF and imReg[i,y,2] < BLACK_CUTOFF):
#				if(imReference[i,y,0] > BLACK_CUTOFF and imReference[i,y,1] > BLACK_CUTOFF and imReference[i,y,2] > BLACK_CUTOFF):
				mask[i,y] = 255
			if(imReg[ow-i-1,y,0] < BLACK_CUTOFF and imReg[ow-i-1,y,1] < BLACK_CUTOFF and imReg[ow-i-1,y,2] < BLACK_CUTOFF):
#				if(imReference[ow-i-1,y,0] > BLACK_CUTOFF and imReference[ow-i-1,y,1] > BLACK_CUTOFF and imReference[ow-i-1,y,2] > BLACK_CUTOFF):
				mask[ow-i-1,y] = 255
		for x in range(0,ow):
			if(imReg[x,i,0] < BLACK_CUTOFF and imReg[x,i,1] < BLACK_CUTOFF and imReg[x,i,2] < BLACK_CUTOFF):
#				if(imReference[x,i,0] > BLACK_CUTOFF and imReference[x,i,1] > BLACK_CUTOFF and imReference[x,i,2] > BLACK_CUTOFF):
				mask[x,i] = 255
			if(imReg[x,oh-i-1,0] < BLACK_CUTOFF and imReg[x,oh-1-i,1] < BLACK_CUTOFF and imReg[x,oh-1-i,2] < BLACK_CUTOFF):
#				if(imReference[x,oh-i-1,0] > BLACK_CUTOFF and imReference[x,oh-1-i,1] > BLACK_CUTOFF and imReference[x,oh-1-i,2] > BLACK_CUTOFF):
				mask[x,oh-1-i] = 255
	kernel = np.ones((3,3),np.uint8)
	mask = cv2.dilate(mask,kernel,iterations=30)
	mask = cv2.GaussianBlur(mask,(99,99),0)
#gain shenanigans
	imask = np.full((ow,oh),255,dtype=np.uint8)-mask
	imReg = np.round(imReg*(cv2.cvtColor(imask,cv2.COLOR_GRAY2BGR)/255) + imReference*(cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)/255)).astype(np.uint8)
#	imReg[mask > 0] = imReference[mask > 0]
	prep = np.copy(imReg)

#border extension
	in_mask = np.zeros((ow, oh),dtype=np.uint8);
	for i in range(0,BORDER_BLEED):
		for y in range(0,oh):
			if(imReg[i,y,0] < BLACK_CUTOFF and imReg[i,y,1] < BLACK_CUTOFF and imReg[i,y,2] < BLACK_CUTOFF):
				in_mask[i,y] = 1
			if(imReg[ow-1-i,y,0] < BLACK_CUTOFF and imReg[ow-1-i,y,1] < BLACK_CUTOFF and imReg[ow-1-i,y,2] < BLACK_CUTOFF):
				in_mask[ow-1-i,y] = 1
			if(imReg[i,y,0] > WHITE_CUTOFF and imReg[i,y,1] > WHITE_CUTOFF and imReg[i,y,2] > WHITE_CUTOFF):
				in_mask[i,y] = 1
			if(imReg[ow-1-i,y,0] > WHITE_CUTOFF and imReg[ow-1-i,y,1] > WHITE_CUTOFF and imReg[ow-1-i,y,2] > WHITE_CUTOFF):
				in_mask[ow-1-i,y] = 1
		for x in range(0,ow):
			if(imReg[x,i,0] < BLACK_CUTOFF and imReg[x,i,1] < BLACK_CUTOFF and imReg[x,i,2] < BLACK_CUTOFF):
				in_mask[x,i] = 1
			if(imReg[x,oh-1-i,0] < BLACK_CUTOFF and imReg[x,oh-1-i,1] < BLACK_CUTOFF and imReg[x,oh-1-i,2] < BLACK_CUTOFF):
				in_mask[x,oh-1-i] = 1
			if(imReg[x,i,0] > WHITE_CUTOFF and imReg[x,i,1] > WHITE_CUTOFF and imReg[x,i,2] > WHITE_CUTOFF):
				in_mask[x,i] = 1
			if(imReg[x,oh-1-i,0] > WHITE_CUTOFF and imReg[x,oh-1-i,1] > WHITE_CUTOFF and imReg[x,oh-1-i,2] > WHITE_CUTOFF):
				in_mask[x,oh-1-i] = 1
	in_mask = cv2.dilate(in_mask,kernel,iterations=5)
#	in_mask = in_mask*mask

	#imReg = cv2.inpaint(imReg,in_mask,3,cv2.INPAINT_TELEA)
#	imReg = cv2.inpaint(imReg,in_mask,3,cv2.INPAINT_NS)

	# Write aligned image to disk. 
	print("Saving aligned image : ", outFilename); 
	cv2.imwrite(outFilename, imReg)

#Display
	if(len(sys.argv) > 2):
		plt.figure("Orig")
		plt.imshow(im[...,::-1])
		plt.figure("Temp")
		plt.imshow(imReference[...,::-1])
		plt.figure("Warp")
		plt.imshow(warp[...,::-1])
		plt.figure("Replacement Mask")
		plt.imshow(mask)
		plt.figure("Post Replacement")
		plt.imshow(prep[...,::-1])
#		plt.figure("Inpaint Mask")
#		plt.imshow(in_mask)
#		plt.figure("Inpaint")
#		plt.imshow(imReg[...,::-1])
		plt.show()
