
# Obtain box center coordinates
def get_box_center(box):
		x1, y1, x2, y2 = box
		return ((x2 + x1) / 2, (y2 + y1) / 2)
# Euclidean distance
def euclidean_distance(point1, point2):
		x1, y1 = point1
		x2, y2 = point2
		return ((x2 - x1)**2 + (y2 - y1)**2)**0.5