
class Shapes(object):
	def __init__(self, label, points, group_id, shape_type, flags):
		   self.label = label
		   self.points = points
		   self.group_id = group_id
		   self.shape_type = shape_type
		   self.flags = flags

	def to_string_form(self):
	       fields = {
            'label': self.label,
            'points': self.points,
            'group_id': self.group_id,
            'shape_type': self.shape_type,
			'flags': self.flags,
        }
	       return fields



class Json(object):
	def __init__(self, version, flags, shapes, imagePath, imageData,imageHeight,imageWidth):
		   self.version = version
		   self.flags = flags
		   self.shapes = shapes
		   self.imagePath = imagePath
		   self.imageData = imageData
		   self.imageHeight = imageHeight
		   self.imageWidth = imageWidth


	def to_string_form(self):
	       fields = {
            'version': self.version,
            'flags': self.flags,
            'shapes': self.shapes,
            'imagePath': self.imagePath,
			'imageData': self.imageData,
			'imageHeight': self.imageHeight,
			'imageWidth': self.imageWidth,
        }
	       return fields



