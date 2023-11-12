# Canny Edge Detection Program

## Overview

This repository contains a Python implementation of the Canny Edge Detection algorithm for computer vision. The program includes several stages, such as Gaussian masking, convolution, non-maximum suppression, and hysteresis thresholding. The implementation utilizes `numpy`, `conv2`, `skimage`, and `matplotlib` libraries.

## Running the Program

To run the program with default values, use the following command:

\```bash
python canny.py
\```

Default values:
- Image: 253036.jpg
- Sigma: 3.0
- Kernel size: 3
- Low hysteresis threshold: 25
- High hysteresis threshold: 50

To customize values, use the `--help` option to see all available options:

```bash
python canny.py --help
```

## Implementation Details

### Gaussian Mask

The Gaussian mask is created by sampling values from the Gaussian function. The mask is centered around zero for the mean, and the values are normalized to maintain image intensity.

### Gaussian Derivative

The Gaussian derivative mask is obtained by sampling values and applying them against the first derivative of the Gaussian function. The resulting values are normalized to maintain intensity.

### Convolution

The image is convolved with the mask by calculating the dot product for each pixel. Transposing the image in the Y direction allows the convolution to work in both X and Y directions.

### Magnitude

The magnitude is calculated using the convolved images for X and Y directions.

### Non-maximum Suppression

This stage involves determining pixel values based on the direction of gradients. Non-maximum suppression is applied by comparing values of neighboring pixels.

### Hysteresis Threshold

A threshold filter is applied to the image, determining edges based on the specified low and high hysteresis thresholds.

## Results

Three different edge detection runs with varying sigma values (1, 3, 5) are provided in the outputs. The sigma value affects smoothing, influencing the program's ability to find edges. After experimentation, a sigma value of 3.0 and thresholds of 25 and 50 were found to work well for the default image.

## Dependencies

- `numpy`
- `opencv-python`
- `scikit-image`
- `matplotlib`

