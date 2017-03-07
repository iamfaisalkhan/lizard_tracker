import numpy as np
import cv2

def _apply_roi(self, img):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `roi`. The rest of the image is set to black.
    """
    # Region of interest
    imshape = img.shape
    roi = np.array([[
                (100,100),
                (imshape[1]/2, imshape[0]/2), 
                (imshape[1]/2+10, imshape[0]/2), 
                (imshape[1]-70,imshape[0])
            ]], dtype=np.int32)

    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, roi, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def abs_sobel_thresh(img, orient='x', sobel_kernel=(1, 15), thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    orient_mask = [1, 0]
    if orient == 'y':
        orient_mask = [0, 1]
    
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient_mask[0], orient_mask[1])
    # 3) Take the absolute value of the derivative or gradient
    
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel) )
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled)
    # 6) Return this mask as your binary_output image
    binary_output[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    
    return binary_output

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
        # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    
    sobelxy = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    
    scaled_sobelxy = np.uint8(255 * sobelxy / np.max(sobelxy) )
    # 5) Create a binary mask where mag thresholds are met
    mask =  ((scaled_sobelxy >= thresh[0]) & (scaled_sobelxy <= thresh[1]) )
    # 6) Return this mask as your binary_output image
    binary_output = np.zeros_like(scaled_sobelxy)
    binary_output[mask] = 1
    
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi)):
        # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    gradient_dir = np.arctan2(abs_sobely, abs_sobelx)
    
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    binary_output = np.zeros_like(abs_sobelx, dtype=np.uint8)
    binary_output[(gradient_dir >= thresh[0]) & (gradient_dir <= thresh[1])] = 1
    
    return binary_output

def color_threshold(image, sthresh=(0, 255), vthresh=(0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[ (s_channel > sthresh[0]) & (s_channel <= sthresh[1])] =  1

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[ (v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1

    return output

def combined_threshold(image):

    # Choose a Sobel kernel size
    ksize = 9

    # Apply each of the thresholding functions
    mag_binary = mag_thresh(image, ksize, (0, 255))
    dir_binary = dir_threshold(image, ksize, (1.29, 1.38))
    color_binary = color_threshold(image, (0, 212), (55, 255))
    
    combined = np.zeros_like(dir_binary)
    combined[( (mag_binary == 1) & (dir_binary == 1) ) | color_binary == 1] = 255

    return combined

