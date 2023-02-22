import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import cv2

def canny_edge_detection(image, sigma=1.4, threshold=0.3):
    # Convert the image to grayscale
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    cv2.imwrite('gray_image.jpg', gray_image)

    # Apply Gaussian blur to the image
    kernel_size = math.ceil(sigma * 6)
    if kernel_size % 2 == 0:
        kernel_size += 1
    padded_image = np.pad(gray_image, pad_width=kernel_size // 2, mode='symmetric')
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j] = math.exp(-((i - kernel_size // 2) ** 2 + (j - kernel_size // 2) ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)
    blurred_image = np.zeros(gray_image.shape)
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            blurred_image[i][j] = np.sum(kernel * padded_image[i:i+kernel_size, j:j+kernel_size])

    # Compute the gradients of the image
    dx = np.zeros(blurred_image.shape)
    dy = np.zeros(blurred_image.shape)
    padded_blurred_image = np.pad(blurred_image, pad_width=1, mode='symmetric')
    for i in range(blurred_image.shape[0]):
        for j in range(blurred_image.shape[1]):
            dx[i][j] = padded_blurred_image[i+1][j] - padded_blurred_image[i][j]
            dy[i][j] = padded_blurred_image[i][j+1] - padded_blurred_image[i][j]
    gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
    gradient_direction = np.arctan2(dy, dx) * 180 / np.pi
    gradient_direction[gradient_direction < 0] += 180

    # Non-maximum suppression
    suppressed_gradient_magnitude = np.zeros(gradient_magnitude.shape)
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            angle = gradient_direction[i][j]
            if (angle >= 0 and angle < 22.5) or (angle >= 157.5 and angle <= 180):
                if gradient_magnitude[i][j] > gradient_magnitude[i][j-1] and gradient_magnitude[i][j] > gradient_magnitude[i][j+1]:
                    suppressed_gradient_magnitude[i][j] = gradient_magnitude[i][j]
            elif (angle >= 22.5 and angle < 67.5):
                if gradient_magnitude[i][j] > gradient_magnitude[i-1][j-1] and gradient_magnitude[i][j] > gradient_magnitude[i+1][j+1]:
                    suppressed_gradient_magnitude[i][j] = gradient_magnitude[i][j]
            elif (angle >= 67.5 and angle < 112.5):
                if gradient_magnitude[i][j] > gradient_magnitude[i-1][j] and gradient_magnitude[i][j] > gradient_magnitude[i+1][j]:
                    suppressed_gradient_magnitude[i][j] = gradient_magnitude[i][j]
            elif (angle >= 112.5 and angle < 157.5):
                if gradient_magnitude[i][j] > gradient_magnitude[i-1][j+1] and gradient_magnitude[i][j] > gradient_magnitude[i+1][j-1]:
                    suppressed_gradient_magnitude[i][j] = gradient_magnitude[i][j]

    # Double thresholding
    high_threshold = threshold * suppressed_gradient_magnitude.max()
    low_threshold = high_threshold * 0.4
    edge_map = np.zeros(suppressed_gradient_magnitude.shape, dtype=np.int32)
    strong_edge_i, strong_edge_j = np.where(suppressed_gradient_magnitude >= high_threshold)
    weak_edge_i, weak_edge_j = np.where((suppressed_gradient_magnitude < high_threshold) & (suppressed_gradient_magnitude >= low_threshold))
    edge_map[strong_edge_i, strong_edge_j] = 255
    for i, j in zip(weak_edge_i, weak_edge_j):
        if ((i > 0 and edge_map[i-1][j] == 255) or 
            (i < edge_map.shape[0]-1 and edge_map[i+1][j] == 255) or
            (j > 0 and edge_map[i][j-1] == 255) or 
            (j < edge_map.shape[1]-1 and edge_map[i][j+1] == 255)):
            edge_map[i][j] = 255

    # Edge tracking by hysteresis
    i, j = np.where(edge_map == 255)
    for idx in range(len(i)):
        if edge_map[i[idx]][j[idx]] == 255:
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if i[idx]+di >= 0 and i[idx]+di < edge_map.shape[0] and j[idx]+dj >= 0 and j[idx]+dj < edge_map.shape[1]:
                        if suppressed_gradient_magnitude[i[idx]+di][j[idx]+dj] >= low_threshold and edge_map[i[idx]+di][j[idx]+dj] == 0:
                            edge_map[i[idx]+di][j[idx]+dj] = 255

    return edge_map

# img = cv2.imread('depositphotos_3046726-stock-photo-isolated-surreal-snail.jpg',0)
img = np.array(Image.open('depositphotos_3046726-stock-photo-isolated-surreal-snail.jpg').convert('L'))
edges = canny_edge_detection(img);
cv2.imshow('edges',edges)
#import image with numpy