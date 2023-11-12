"""
Canny Edge Detection Program

Author: Curtis Helsel
Date: October 2021

Description:
This Python script implements the Canny Edge Detection algorithm for computer 
vision. The program includes various stages such as Gaussian masking, 
convolution, non-maximum suppression, and hysteresis thresholding. It is 
designed to be customizable through command-line options and outputs the 
different stages of the implementation along with exporting them to files.

Dependencies:
- numpy
- conv2
- skimage
- matplotlib
"""

import argparse
import numpy as np
import cv2
from skimage import filters
from matplotlib import pyplot as plt

# Creates the one-dimensional Gaussian mask 
# with sigma value and kernel size provided
# For most of the images I tried, a sigma between 1-3
# produced the best results. As a default, mine is set to 3.0
def CreateGaussianMask(sigma, kernel_size):

    # Intial mask values as samples of the Gaussian Function
    # for example [-1, 0, 1]
    gaussian_mask = np.arange(-kernel_size / 2 + .5, kernel_size / 2)

    # For each sample in the mask compute the Gaussian value
    # e^(-x^2 / 2 * sigma^2) / (sqrt(2*pi) * sigma)
    for i, x in  enumerate(gaussian_mask):
        gaussian_mask[i] = np.exp(-(x ** 2) / (2 * (sigma ** 2)))
        gaussian_mask[i] /= np.sqrt(2 * np.pi) * sigma
    
    # Normalizing the mask
    gaussian_mask /= sum(gaussian_mask)

    return gaussian_mask

        
# Creates the one-dimensional Gaussian mask derivative with
# sigma value and kernel size provided
def CreateDerivativeGaussian(sigma, kernel_size):

    # Intial mask values as samples of the Gaussian Function
    # for example [-1, 0, 1]
    gaussian_derivative = np.arange(-kernel_size / 2 + .5, kernel_size / 2)

    # For each sample in the mask compute the Gaussian value 
    # for the first derivative of the Gaussian function
    # e^(-x^2 / 2 * sigma^2) * -x / (sqrt(2*pi) * sigma ^ 3)
    for i, x in  enumerate(gaussian_derivative):
        gaussian_derivative[i] = -x * np.exp(-(x ** 2) / (2 * (sigma ** 2))) 
        gaussian_derivative[i] /= (sigma ** 3) * np.sqrt(2 * np.pi) 

    return gaussian_derivative


# Convolves the provided image with the mask provided. 
def Convolve(mask, image):
    kernel_size = len(mask)
    half = int(kernel_size / 2)
    length, width = image.shape

    # Pad the image with zeros to retain the image size
    pad = int((kernel_size - 1) / 2)
    I_padded = np.pad(image, pad, 'constant', constant_values=0)
    
    # Empty image array to add output from convolution
    convolved_image = np.empty((length, width), dtype=np.double)

    # Convolve the image with the dot product of pixels
    # and gaussian mask
    for i in range(length):
        for j in range(width):
            start = j - half + pad
            end = j + half + pad + 1
            pixel_value = np.dot(I_padded[i + pad][start:end], mask)
            convolved_image[i][j] = pixel_value

    return convolved_image


# Calculates the magnitude for each pixel of the two images
def CalculateMagnitude(Ix, Iy):
    # Shape provides the pixel length and width of the image
    length, width = Ix.shape
    
    # Empty image array to add output from magnitude calculation
    magnitude = np.empty((length, width), dtype=np.double)

    # For each pixel, calculate the magnitude
    # sqrt(Ix(x,y)^2 + Iy(x,y)^2)
    for i in range(length):
        for j in range(width):
            magnitude[i][j] = np.sqrt((Ix[i][j] ** 2) + (Iy[i][j] ** 2))

    return magnitude


# If the center value along the normal direction is max, 
# keep the value otherwise set it to zero.  
def NonMaximumSupression(magnitude, Iy, Ix):

    # Value setup for comparsion of direction
    right = np.arctan2(0,1)                             # 0 degrees
    upright = np.arctan2(np.sqrt(2) / 2,np.sqrt(2) / 2) # 45 degrees
    downright = - upright                               # -45 degrees
    up = np.arctan2(1,0)                                # 90 degrees
    down = - up                                         # -90 degrees
    upleft = np.arctan2(np.sqrt(2) / 2,-np.sqrt(2) / 2) # 135 degrees
    downleft = -upleft                                  # -135 degrees
    left = np.arctan2(0,-1)                             # 180 degrees

    
    # Midpoints between angles for pixel determination
    upright_midpoint = midpoint(upright, right) # 22.5 degrees
    downright_midpoint = - upright_midpoint     # -22.5 degrees
    up_midpoint = midpoint(up, upright)         # 67.5 degrees
    down_midpoint = - up_midpoint               # -67.5 degrees
    upleft_midpoint = midpoint(upleft, up)      # 112.5 degrees
    downleft_midpoint = - upleft_midpoint       # -112.5 degrees
    left_midpoint = midpoint(left, upleft)      # 157.5 degrees

    # Gradient direction of each pixel 
    directions = np.arctan2(Iy, Ix)

    # Shape provides the pixel length and width of the image
    length, width = magnitude.shape

    # Empty image array to add output from 
    # non-maximum supression calculation
    image = np.empty((length, width), dtype=np.double)
    
    for i in range(1, length-1):
        for j in range(1, width-1):
            direction = directions[i][j]
            center = magnitude[i][j]

            # Between 0 and 45 degrees or -135 and -180 degrees
            if right >= direction <= upright or downleft <= direction >= -left:

                # If direction is greater than 22.5 or less than -157.5 degrees
                if direction > upright_midpoint or direction < -left_midpoint:
                    right = magnitude[i+1][j-1]
                    left = magnitude[i-1][j+1]
                else:
                    right = magnitude[i+1][j]
                    left = magnitude[i-1][j]

            # Between 45 and 90 degrees or -90 and -135 degrees
            elif upright > direction <= up or down <= direction > downleft:

                # If direction is greater than 67.5 or less than -112.5 degrees
                if direction > up_midpoint or direction < downleft_midpoint:
                    right = magnitude[i][j-1]
                    left = magnitude[i][j+1]
                else:
                    right = magnitude[i+1][j-1]
                    left = magnitude[i-1][j+1]

            # Between 90 and 135 degrees or -45 and -90 degrees
            elif up > direction <= upleft or downright <= direction > down:
                
                # If direction is greater than 112.5 or less than -67.5 degrees
                if direction > upleft_midpoint or direction < down_midpoint:
                    right = magnitude[i-1][j-1]
                    left = magnitude[i+1][j+1]
                else:
                    right = magnitude[i][j-1]
                    left = magnitude[i][j+1]

            # Between 135 and 180 degrees or 0 and -45 degrees
            elif upleft > direction <= left or right < direction > downright:
                
                # If direction is greater than 157.5 or less than -22.5 degrees
                if direction > left_midpoint or direction < -downright_midpoint:
                    right = magnitude[i-1][j]
                    left = magnitude[i+1][j]
                else:
                    right = magnitude[i-1][j-1]
                    left = magnitude[i+1][j+1]

            # check if center is greater than both left and right pixels
            if center > right and center > left:
                image[i][j] = center
            else:
                image[i][j] = 0

    return image

# finds the midpoint between two angles
def midpoint(high, low):
   
    midpoint = (high + low) / 2

    return midpoint

# Find if the pixel is within range and intesifies the edges
def ApplyHysteresisThreshold(image, low, high):

    # Apply the hysteresis threshould against the image
    # with the lower and upper bounds
    hysteresis = filters.apply_hysteresis_threshold(image, low, high)

    length, width = image.shape
    
    # Empty image array to add output from magnitude calculation
    hysteresis_image = np.empty((length, width), dtype=np.double)

    # For each pixel, mutltiple the output of the hysteresis
    # threshold with 255 to emphasize the image edges.
    for i in range(length):
        for j in range(width):
            hysteresis_image[i][j] = hysteresis[i][j] * 255

    return hysteresis_image

def main(FLAGS):
    
    fig = plt.figure()
    rows = 2
    columns = 4

    # Read in a gray scale image and store it as matrix I
    I = cv2.imread(FLAGS.image, cv2.IMREAD_GRAYSCALE)

    # Add original image to plot
    fig.add_subplot(rows, columns, 1)
    plt.gray()
    plt.imshow(I)
    plt.title("Original")
    plt.axis('off')

    # Transposing the images allows to convolve with 
    # the same function.
    I_transposed = np.transpose(I)

    # Create one-dimensional Gaussian Mask 
    G = CreateGaussianMask(FLAGS.sigma, FLAGS.kernel_size)

    # Create one-dimensional for the first derivative Gaussian
    G_derivative = CreateDerivativeGaussian(FLAGS.sigma, FLAGS.kernel_size)

    # Convolve the image in the x and y directions
    # Transposing the image back puts it in the
    # correct orientation
    Ix = Convolve(G, I)
    Iy = np.transpose(Convolve(G, I_transposed)) 


    # Write convolved images 
    cv2.imwrite('./x_convolved.jpg', Ix)
    cv2.imwrite('./y_convolved.jpg', Iy)

    # Add the Gaussian mask against X output to plot
    fig.add_subplot(rows, columns, 2)
    plt.imshow(Ix)
    plt.title("X convolved")
    plt.axis('off')

    # Add the Gaussian mask against X output to plot
    fig.add_subplot(rows, columns, 3)
    plt.imshow(Iy)
    plt.title("Y convolved")
    plt.axis('off')

    # Transpose Ix so it can be convolved with y mask
    # and transpose again to get correct orientation
    Ix_transposed = np.transpose(Ix)
    Ixy = np.transpose(Convolve(G_derivative, Ix_transposed))
    Ixy = (Ixy / np.max(Ixy)) * 255

    # Add the derivative Xy output to plot
    fig.add_subplot(rows, columns, 4)
    plt.imshow(Ixy)
    plt.title("Xy convolved")
    plt.axis('off')

    # Convolve with x mask derivative in the y direction
    Iyx = Convolve(G_derivative, Iy)
    Iyx = (Iyx / np.max(Iyx)) * 255

    # Add the derivative Yx output to plot
    fig.add_subplot(rows, columns, 5)
    plt.imshow(Iyx)
    plt.title("Yx convolved")
    plt.axis('off')

    # Write derivative convolved images 
    cv2.imwrite('./xy_convolved.jpg', Ixy)
    cv2.imwrite('./yx_convolved.jpg', Iyx)

    # Calculate the magnitude of the two convolved images
    Im = CalculateMagnitude(Ixy, Iyx)

    # Add the magnitude output to plot
    fig.add_subplot(rows, columns, 6)
    plt.imshow(Im)
    plt.title("Magnitude")
    plt.axis('off')

    # Write magnitude image 
    cv2.imwrite('./Imagnitude.jpg', Im)

    # Calculate the non-maximum supression of two images
    # against the magnitude
    Inms = NonMaximumSupression(Im, Iyx, Ixy)

    # Add the non-maximum supression output to plot
    fig.add_subplot(rows, columns, 7)
    plt.imshow(Inms)
    plt.title("Non-maximum Suppression")
    plt.axis('off')

    # Write non-maximum suppression image 
    cv2.imwrite('./Inms.jpg', Inms)

    # Calculate the hysteresis threshould of image
    canny_image = ApplyHysteresisThreshold(Inms, FLAGS.low, FLAGS.high)

    # Add the canny image output to plot
    fig.add_subplot(rows, columns, 8)
    plt.imshow(canny_image)
    plt.title("Canny")
    plt.axis('off')

    # Write canny image 
    cv2.imwrite('./canny_image.jpg', canny_image)

    # Display the plot of all images
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Canny Edge Detection')
    parser.add_argument('--image',
                        type=str, default='./253036.jpg',
                        help='Provide an image file path.')

    parser.add_argument('--sigma',
                        type=float, default=3.0,
                        help='Provide sigma value.')

    parser.add_argument('--kernel_size',
                        type=float, default=3,
                        help='Provide kernel size value to convolve .')

    parser.add_argument('--low',
                        type=float, default=25,
                        help='Provide low threshold value for hysteresis.')

    parser.add_argument('--high',
                        type=float, default=50,
                        help='Provide high  threshold value for hysteresis.')

    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
