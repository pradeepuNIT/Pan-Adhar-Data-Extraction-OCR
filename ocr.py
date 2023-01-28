# east text detector
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import dlib
import time
import cv2
import pytesseract

def perform_ocr(image, config=None):
	text = pytesseract.image_to_string(image, config=config)
	cv2.imshow(text, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return text

def logo_detector(image, detector):
	detector = dlib.simple_object_detector(detector)
	boxes = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	b = boxes[0]
	(x, y, w, h) = (b.left(), b.top(), b.right(), b.bottom())
	return (x, y, w, h)


def find_ratio_change(orig_image, cropped_image):
	orig_h, orig_w = orig_image.shape[:2]
	cropped_h, cropped_w = cropped_image.shape[:2]
	rW = orig_w/float(cropped_w)
	rH = orig_h/float(cropped_h)
	return rW, rH


def crop_and_resize_image(image, logo_bottom, resize_shape):

	image = image[logo_bottom: , :]
	image_after_logo_crop = image.copy()

	(newW, newH) = resize_shape

	# resize the image and grab the new image dimensions
	resized_image = cv2.resize(image, (newW, newH))

	return image_after_logo_crop, resized_image
	# cv2.imshow("crop_resized", image)
	# cv2.waitKey(0)

def east_text_detector(image, east_detector):
	layerNames = [
			"feature_fusion/Conv_7/Sigmoid",
			"feature_fusion/concat_3"]
	# load the pre-trained EAST text detector
	print("[INFO] loading EAST text detector...")
	net = cv2.dnn.readNet(east_detector)

	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	H, W = image.shape[:2]
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	start = time.time()
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	end = time.time()
	# show timing information on text prediction
	print("[INFO] text detection took {:.6f} seconds".format(end - start))
	return scores, geometry


def remove_low_confident_boxes(scores, geometry, min_confidence):
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates that
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < min_confidence:
				continue
	 
			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
	 
			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
	 
			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
	 
			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
	 
			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	return rects, confidences
	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes


def form_groups(boxes, rW, rH, image):
	padding = 5
	groups = []
	# loop over the bounding boxes
	for (startX, startY, endX, endY) in sorted(boxes, key=lambda x: x[1]):
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = max(0, int(startX * rW) - padding)
		startY = max(0, int(startY * rH) - padding)
		endX = int(endX * rW) + padding
		endY = int(endY * rH) + padding

		if endY-startY > endX-startX:
			continue
		found_group = False
		for group in groups:
			if startY < group['endY']:
				group['endY'] = max(endY, group['endY'])
				group['startX'] = min(startX, group['startX'])
				group['endX'] = max(endX, group['endX'])
				found_group = True
				break
		if not found_group:
			groups.append({'startY': startY, 'endY':endY, 'startX': startX, 'endX': endX})
	return groups


# main(args.image, args.detector, (args.width, args.length), args.east,
# 	args["min_confidence"])
def main(image_path, detector, resize_shape, east_detector, min_confidence):
	image = cv2.imread(image_path)
	(x1, y1, x2, y2) = logo_detector(image, detector)
	image_after_logo_crop, resized_image = crop_and_resize_image(image, y2, resize_shape)
	rW, rH = find_ratio_change(image_after_logo_crop, resized_image)
	scores, geometry = east_text_detector(resized_image, east_detector)
	# min_confidence = args['min_confidence']
	rects, confidences = remove_low_confident_boxes(scores, geometry, min_confidence)
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	groups = form_groups(boxes, rW, rH, image_after_logo_crop)
	for group in groups:
		sampled = \
			image_after_logo_crop[group['startY']:group['endY'],
								  group['startX']:group['endX']]
		print(perform_ocr(sampled))
		


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", type=str,
		help="path to input image")
	ap.add_argument("-e", "--east", type=str,
		help="path to input EAST text detector")
	ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
		help="minimum probability required to inspect a region")
	ap.add_argument("-w", "--width", type=int, default=480,
		help="resized image width (should be multiple of 32)")
	ap.add_argument("-l", "--length", type=int, default=320,
		help="resized image height (should be multiple of 32)")
	ap.add_argument("-d", "--detector", required=True, help="Path to trained object detector")
	args = vars(ap.parse_args())
	main(args['image'], args['detector'], (args['width'], args['length']),
		 args['east'], args['min_confidence'])

# Adhar Card
'''
python3 ocr.py \
	-d adhar_detector.svm \
	-e frozen_east_text_detection.pb \
	-i adhar_images/adhar1.jpg
'''

# Pan Card
'''
python3 ocr.py \
	-d pan_detector.svm \
	-e frozen_east_text_detection.pb \
	-i pan_images/img2.jpg
'''

# Uk Residence
'''
python3 ocr.py \
	-d uk_detector.svm \
	-e frozen_east_text_detection.pb \
	-i uk_images/UK1.jpg
'''