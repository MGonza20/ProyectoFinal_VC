
# Obtain box center coordinates
def get_box_center(box):
		x1, y1, x2, y2 = box
		return ((x2 + x1) / 2, (y2 + y1) / 2)

# Obtain the box bottom center coordinates
def get_box_bottom_center(box):
		x1, y1, x2, y2 = box
		return ((x2 + x1) / 2, y2)
# Euclidean distance
def euclidean_distance(point1, point2):
		x1, y1 = point1
		x2, y2 = point2
		return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def convert_pixels_distance_to_meters(pixel_distance, reference_in_meters, reference_in_pixels):
		return (pixel_distance * reference_in_meters) / reference_in_pixels

def convert_meters_distance_to_pixels(meters_distance, reference_in_meters, reference_in_pixels):
		return (meters_distance * reference_in_pixels) / reference_in_meters

def get_box_height(box):
		x1, y1, x2, y2 = box
		return y2 - y1