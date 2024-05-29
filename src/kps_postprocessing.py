
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
# from court_detector import CourtDetector



def crop_image(img, x, y, crop_size=40):
    x, y = int(x), int(y)
    
    img_height, img_width = img.shape[:2]
    x_min = max(x - crop_size, 0)
    x_max = min(x + crop_size, img_width)
    y_min = max(y - crop_size, 0)
    y_max = min(y + crop_size, img_height)

    img_crop = img[y_min:y_max, x_min:x_max]
    return img_crop


def niblack_thresholding(image, window_size, k):
    mean = cv2.blur(image, (window_size, window_size))
    mean_of_squares = cv2.blur(image**2, (window_size, window_size))
    std_deviation = np.sqrt(mean_of_squares - mean**2)
    threshold = mean + k * std_deviation

    binary_image = (image > threshold).astype(np.uint8) * 255

    return binary_image


def image_quantization(image, k=3):

    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    quantized_pixels = centers[labels.flatten()]
    return quantized_pixels.reshape(image.shape)


def detect_intersections(skeletonized_img):
    # Detectar bordes en la imagen
    edges = cv2.Canny(skeletonized_img, 50, 150, apertureSize=3)

    # Detectar líneas en la imagen usando la transformada de Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=0, maxLineGap=9)

    intersections = []
    if lines is not None:
        for line1 in lines:
            for line2 in lines:
                if np.array_equal(line1, line2):
                    continue
                x1, y1, x2, y2 = line1[0]
                x3, y3, x4, y4 = line2[0]

                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if denom != 0:
                    Px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                    Py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
                    if 0 <= Px < skeletonized_img.shape[1] and 0 <= Py < skeletonized_img.shape[0]:
                        intersections.append((int(Px), int(Py)))

    # Eliminar intersecciones duplicadas o cercanas
    filtered_intersections = []
    for point in intersections:
        if not any(np.linalg.norm(np.array(point) - np.array(p)) < 10 for p in filtered_intersections):
            filtered_intersections.append(point)

    # Encontrar la intersección más cercana al centro
    center = (skeletonized_img.shape[1] // 2, skeletonized_img.shape[0] // 2)
    if filtered_intersections: closest_point = min(filtered_intersections, key=lambda point: np.linalg.norm(np.array(point) - np.array(center)))
    else: closest_point = center  # Si no se encuentra ninguna intersección, usar el centro
    return closest_point


def calculate_delta(crop_image, coord):
    # Dibujar un círculo azul en el centro
    center = (crop_image.shape[1] // 2, crop_image.shape[0] // 2)

    # Dibujar un círculo rojo en la coordenada guardada
    if coord:
        # Calcular la diferencia en x y y entre el centro y la coordenada guardada
        delta_x = coord[0] - center[0]
        delta_y = coord[1] - center[1]

    return (delta_x, delta_y)


def fix_keypoints(keypoints, frame):
    keypoints = keypoints.reshape(-1, 2)
    new_keypoints = []
    for i, kps in enumerate(keypoints):
        x, y = kps

        # Cropping the image around the keypoint
        cropped_img = crop_image(frame, x, y)

        # Image Quantization to reduce the number of colors
        quantized_img = image_quantization(cropped_img)

        # Converting image to grayscale
        gray_img = cv2.cvtColor(quantized_img, cv2.COLOR_BGR2GRAY)

        # Niblack Thresholding to binarize the image
        binary_img = niblack_thresholding(gray_img, 10, 0.5)

        # Using Zhang-Suen Thinning Algorithm to get the skeleton of the binary image
        skeleton = (skeletonize(binary_img/255)*255).astype(np.uint8)

        # Detecting the intersection point
        intersection = detect_intersections(skeleton)

        # Calculating the delta between the center and the intersection point   
        delta_x, delta_y = calculate_delta(cropped_img, intersection)
        
        # Appending the new keypoints
        new_keypoints.append(x + delta_x)
        new_keypoints.append(y + delta_y)

        # cv2.circle(frame, (int(x) + int(delta_x), int(y) + int(delta_y)), 2, (255, 0, 0), -1)

    return new_keypoints


# cd = CourtDetector('../models/court/keypoints_model.pth')
# frame = cv2.imread('../data/raw/court/test_images/test_image3.jpg')
    

# keypoints = cd.detect(frame)
# new_keypoints = fix_keypoints(keypoints, frame)

# plt.figure(figsize=(15, 15))
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()