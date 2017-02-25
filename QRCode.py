import cv2
# turn off OpenCL because of compatibility issues
cv2.ocl.setUseOpenCL(False)
import numpy


class QRCode:

    def __init__(self, path, dimensions, edge_buffer):
        # turn off OpenCL because of incompatibility issues
        cv2.ocl.setUseOpenCL(False)

        self.path = path
        self.image = cv2.imread(path)
        self.pixel_length = self.image.shape[0]
        self.pixel_width = self.image.shape[1]
        self.length, self.width = dimensions
        self.edge_buffer = edge_buffer
        self.depth = 0
        self._find_pattern_keypoints()
        self._set_display_2d_coordinates()
        self._set_mapping_2d_coordinates()
        self._set_3d_coordinates()

    def _find_pattern_keypoints(self):
        self.ORB = cv2.ORB_create()
        pattern = cv2.imread(self.path, 0)
        self.keypoints, self.des1 = self.ORB.detectAndCompute(pattern, None)
        self.shape = pattern.shape

    def _set_mapping_2d_coordinates(self):
        self.mapping_coordinates_2d = numpy.float32([[0, 0],
                                                     [0, self.pixel_length],
                                                     [self.pixel_width, self.pixel_length],
                                                     [self.pixel_width, 0]]).reshape(-1, 1, 2)

    def _set_display_2d_coordinates(self):
        self.display_coordinates_2d = numpy.float32(
            [[0 + self.edge_buffer, 0 + self.edge_buffer],  # top left
             [0 + self.edge_buffer, self.pixel_length - self.edge_buffer],  # bottom left
             [self.pixel_width - self.edge_buffer, self.pixel_length - self.edge_buffer],  # bottom right
             [self.pixel_width - self.edge_buffer, 0 + self.edge_buffer]  # top right
             ]).reshape(-1, 1, 2)

    def _set_3d_coordinates(self):
        pattern_half_length = self.length / 2
        pattern_half_width = self.width / 2

        self.coordinates_3d = numpy.array([
            [-pattern_half_length, pattern_half_width, 0],  # top left
            [-pattern_half_length, -pattern_half_width, 0],  # bottom left
            [pattern_half_length, -pattern_half_width, 0],  # bottom right
            [pattern_half_length, pattern_half_width, 0]  # top right
        ]).astype("float32")