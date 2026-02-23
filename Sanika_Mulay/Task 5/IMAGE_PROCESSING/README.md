IMAGE PROCESSING MINI TASKS (Using OpenCV)

Objective:
To perform basic image processing operations using OpenCV such as edge detection, thresholding, and image augmentation techniques.

Edge Detection (Canny)

Canny Edge Detection is used to detect edges in an image.
It highlights object boundaries by identifying areas with strong intensity changes.

Steps Performed:

Convert image to grayscale

Apply Gaussian blur to remove noise

Apply Canny edge detection function

Display the detected edges

Purpose:
Edge detection helps in identifying shapes and boundaries in an image. It is useful for feature extraction and object recognition tasks.

Image Thresholding

Image thresholding is used to convert a grayscale image into a binary image (black and white).

Types Used:

Simple Binary Thresholding

Adaptive Thresholding (if applied)

Steps Performed:

Convert image to grayscale

Apply threshold function

Convert pixel values above threshold to white

Convert pixel values below threshold to black

Purpose:
Thresholding helps in separating foreground objects from the background.
It improves clarity for further image processing tasks.

Image Augmentation

Image augmentation is used to artificially increase the size of the dataset by applying transformations.

Techniques Used:

Flip:

Horizontal flip

Vertical flip
This creates a mirror version of the image.

Rotation:

Rotate image by certain degrees (e.g., 45°, 90°)
Helps model learn from different orientations.

Brightness Adjustment:

Increase or decrease brightness
Helps model handle different lighting conditions.

Purpose:
Image augmentation improves model generalization and reduces overfitting.