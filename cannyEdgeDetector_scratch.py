# Import the necessary libraries
import random
import warnings
import numpy as np
from skimage import data
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import gaussian,sobel_h,sobel_v,sobel,laplace
from skimage.feature import canny
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

def peak_signal_to_noise_ratio(original_image, output_image):
  mean_squared_error = np.mean(np.square(original_image - output_image), dtype = np.float64)
  psnr = 10 * np.log10((1 ** 2) / mean_squared_error)
  return psnr


def normalize(image):
  # Do min-max normalisation
  normalizedImage = (image - image.min())/(image.max() - image.min())
  return normalizedImage

# Define a function to return a gaussian kernel
def gaussian_kernel(kernel_size, sigma):
  step_size = kernel_size // 2
  factor = 1 / (2 * np.pi * (sigma ** 2))
  gaussian_filter = np.zeros((kernel_size, kernel_size), np.float32)
  for x in range(-step_size, step_size + 1):
    for y in range(-step_size, step_size + 1):
      gaussian_filter[x + step_size][y + step_size] = np.exp(-((x**2 + y**2) / (2.0*(sigma**2)))) * factor
  return gaussian_filter


# Define a function for the convolution operation
def convolve(image, kernel):
  kernel = np.fliplr(np.flipud(kernel)) # double flip the kernel
  image_rows = image.shape[0]
  image_cols = image.shape[1]
  kernel_size = kernel.shape[0]
  step_size = kernel_size // 2
  outputImage = np.zeros((image_rows, image_cols))
  for i in range(step_size, image_rows - step_size):
    for j in range(step_size, image_cols - step_size):
      outputImage[i][j] = (kernel * image[i - step_size : i + step_size + 1, j - step_size : j + step_size + 1]).sum()
  return outputImage

def gradient_of_image(image):
  # Lets use a sobel filter for extracting the gradients
  # The gradients are extracted along the x axis and the y axis
  sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])#/8
  sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])#/8
  vertical_gradient =  convolve(image, sobel_x)#/4
  horizontal_gradient =  convolve(image,sobel_y)#/4


  sobelOutput = np.hypot(horizontal_gradient, vertical_gradient)
  gradient_direction = np.arctan2(horizontal_gradient,vertical_gradient)
  gradient_direction = np.degrees(gradient_direction)
  gradient_direction[gradient_direction < 0] += 360
  return (sobelOutput, gradient_direction)

def hysteresis(image, high_threshold_ratio, low_threshold_ratio):
  list_strong_points = [] # list to store the strong points

  # Get the thresholds
  high_threshold = image.max() * high_threshold_ratio
  low_threshold = image.max() * low_threshold_ratio

  # Go through each point and check if it is a strong point
  for row_number in range(0,image.shape[0]):
    for col_number in range(0, image.shape[1]):
      # if the pixel value is greater than the high threshold, set to 1
      if image[row_number][col_number] >= high_threshold:
        image[row_number][col_number] = 1
        list_strong_points.append((row_number, col_number))
      # if the pixel value is less than the low threshold, set to 0,
      # leave all the points as they are
      elif image[row_number][col_number] < low_threshold:
        image[row_number][col_number] = 0

  # Update the weak points if they are connected to strong points
  while(len(list_strong_points) != 0):
    current_strong_point = random.choices(list_strong_points)[0]
    if current_strong_point[0] in range(0,image.shape[0]) and current_strong_point[1] in range(0,image.shape[1]):
      new_neighbours = get_valid_neighbours(image,current_strong_point, high_threshold, low_threshold)
      list_strong_points.extend(new_neighbours)
    list_strong_points.remove(current_strong_point)
  return image

def non_maxima_suppression(gradientImage,gradientDirection):
  image_rows, image_cols = gradientImage.shape
  outputImage = np.zeros((image_rows, image_cols))
  
  # Iterate through the image and find out the values of the 
  # neighbours according to the direction of the gradient
  for row_number in range(1, image_rows - 1):
    for col_number in range(1, image_cols - 1):
      current_pixel = gradientImage[row_number][col_number]
      angle = gradientDirection[row_number][col_number]
      if (0 <= angle < 22.5 or 157.5 <= angle < 202.5 or 337.5 <= angle <= 360):
        neighbour_1 = gradientImage[row_number][col_number + 1]
        neighbour_2 = gradientImage[row_number][col_number - 1]
      elif (22.5 <= angle < 67.5 or 202.5 <= angle < 247.5):
        neighbour_1 = gradientImage[row_number + 1][col_number - 1]
        neighbour_2 = gradientImage[row_number - 1][col_number + 1]
      elif (67.5 <= angle < 112.5 or 247.5 <= angle < 292.5):
        neighbour_1 = gradientImage[row_number + 1][col_number]
        neighbour_2 = gradientImage[row_number - 1][col_number]
      elif (112.5 <= angle < 157.5 or 292.5 <= angle < 337.5):
        neighbour_1 = gradientImage[row_number + 1][col_number + 1]
        neighbour_2 = gradientImage[row_number - 1][col_number - 1]
      
      if (current_pixel >= neighbour_1 and current_pixel >= neighbour_2):
        outputImage[row_number][col_number] = current_pixel
    
  return outputImage

def get_valid_neighbours(image,current_pixel, high_threshold, low_threshold):
  x_index = current_pixel[0]
  y_index = current_pixel[1]
  new_strong_pixels = [] # Store the indices of the newly turned strong pixels

  # Check the neighborhood of the current strong pixel to see there are any weak pixels
  # If there are weak pixels, turn them into strong pixels and add them to the list
  if x_index + 1 < image.shape[0] and x_index - 1 >= 0 and y_index + 1 < image.shape[0] and y_index - 1 >= 0: # boundary check
    if y_index + 1 <= image.shape[1] and low_threshold <= image[x_index][y_index + 1] < high_threshold:
       image[x_index][y_index + 1] = 1
       new_strong_pixels.append((x_index, y_index + 1))

    if y_index - 1 >= 0 and low_threshold <= image[x_index][y_index - 1] < high_threshold:
       image[x_index][y_index - 1] = 1
       new_strong_pixels.append((x_index, y_index - 1))

    if x_index + 1 < image.shape[0] and low_threshold <= image[x_index + 1][y_index] < high_threshold:
       image[x_index + 1][y_index] = 1
       new_strong_pixels.append((x_index + 1, y_index))

    if x_index - 1 >= 0 and low_threshold <= image[x_index - 1][y_index] < high_threshold:
       image[x_index - 1][y_index] = 1
       new_strong_pixels.append((x_index - 1, y_index))

    if x_index - 1 >= 0 and y_index - 1 >= 0 and low_threshold <= image[x_index - 1][y_index - 1] < high_threshold:
       image[x_index - 1][y_index - 1] = 1
       new_strong_pixels.append((x_index - 1, y_index - 1))

    if x_index + 1 < image.shape[0] and y_index + 1 < image.shape[1] and low_threshold <= image[x_index + 1][y_index + 1] < high_threshold:
       image[x_index + 1][y_index + 1] = 1
       new_strong_pixels.append((x_index + 1, y_index + 1))

    if x_index - 1 >= 0 and y_index + 1 <= image.shape[1] and low_threshold <= image[x_index - 1][y_index + 1] < high_threshold:
       image[x_index - 1][y_index + 1] = 1
       new_strong_pixels.append((x_index - 1, y_index + 1))

    if x_index + 1 < image.shape[0] and y_index - 1 >= 0 and low_threshold <= image[x_index + 1][y_index - 1] < high_threshold:
       image[x_index + 1][y_index - 1] = 1
       new_strong_pixels.append((x_index + 1, y_index - 1))

  return new_strong_pixels

def myCannyEdgeDetector(image, Low_Threshold, High_Threshold):
    
    grayscaleImage = rgb2gray(image) # grayscaleImage has values 0.0 - 1.0
    kernel = gaussian_kernel(3,1)
    kernel = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype = np.float64)*(1/16)
    blurredImage_1 = convolve(grayscaleImage,kernel) # blurredImage has values 0.0 - 1.0. Might need to normalize. max = 0.086 min = 0.0
    gradientImage, gradientDirection = gradient_of_image(blurredImage_1)
    suppressedImage = non_maxima_suppression(gradientImage, gradientDirection)
    thresholdedImage = hysteresis(suppressedImage, 0.2, 0.1)
    cannyImage = canny(grayscaleImage,sigma = 1, low_threshold=Low_Threshold,high_threshold=High_Threshold)
    cannyImage[cannyImage == True] = 1.0
    cannyImage[cannyImage == False] = 0.0
    psnr = peak_signal_to_noise_ratio(cannyImage, thresholdedImage)
    ssim = structural_similarity(cannyImage, thresholdedImage)
    print("The PSNR value is : ",psnr)
    print("The SSIM Index is : ",ssim)
    fig,axes = plt.subplots(nrows=1, ncols=3,figsize = (6,6))
    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[0].axis("off")
    axes[1].imshow(thresholdedImage,cmap = 'gray')
    axes[1].set_title("My Canny Edge Detector")
    axes[1].axis("off")
    axes[2].imshow(cannyImage,cmap = 'gray')
    axes[2].set_title("Inbuilt Canny Edge Detector")
    axes[2].axis("off")
    plt.show()

warnings.filterwarnings("ignore")
image_address = input("Enter the image address (images/<picture_name>.jpg): ")
High_Threshold = float(input("Enter the high threshold :"))
Low_Threshold = float(input("Enter the low threshold :"))
image = imread(image_address)
myCannyEdgeDetector(image, Low_Threshold, High_Threshold)
