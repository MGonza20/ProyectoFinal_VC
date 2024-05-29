
# Obtain box center coordinates
def get_box_center(box):
		x1, y1, x2, y2 = box
		return ((x2 + x1) / 2, (y2 + y1) / 2)