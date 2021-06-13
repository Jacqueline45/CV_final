Reference: https://github.com/biubug6/Pytorch_Retinaface

input: [batch_size, 3, H, W] (need to resize all training images to same shape)
output: (box_regression, classification, landmarks_regressions)

	box_regression: [batch_size, H'*W', 4], x1, y1, x2, y2 of two diagonal corners of the box
	classification: [batch_size, H'*W', 2], the probability of is face or not
	landmark_regression: [batch_size, H'*W', 10], coordinates of (x, y) of five landmarks of a face

	H': feature map height, W': feature map weight (H'*W' will be the maximum num of faces able to be detected)

usage: initialize with a model config (written in config.py)
	
resnet50: 104 MB
mobilenet: 1.75 MB
