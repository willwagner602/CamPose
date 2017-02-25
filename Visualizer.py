from PIL import Image as ImageFile
import numpy
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Visualizer:

    def __init__(self, image, qr_code):
        self.image = image
        self.qr_code = qr_code

        self.x = []
        self.y = []
        self.z = []

        self.figure = None
        self.axis = None
        self.image_array = None

    def _set_camera_faces(self):
        for face in self.image.camera_faces:
            for point in face:
                self.x.append(point[0])
                self.y.append(point[1])
                self.z.append(point[2])

    def _get_axis_limits(self):
        # get maximum x, y, z for setting up axis limits
        max_horizontal = max(self.x + self.y)
        min_horizontal = min(self.x + self.y)
        self.horizontal = max((abs(max_horizontal), abs(min_horizontal), 300))
        self.max_z = max(self.z)

    def _scale_pattern_image(self):
        # get pattern image and scale it to the appropriate size
        image_file = ImageFile.open(self.qr_code.path)
        image = image_file.rotate(-90).resize((self.qr_code.width, self.qr_code.length))
        white = 1
        black = 0
        visible = 1
        image_array = []
        for row in numpy.array(image):
            row_vals = []
            for pixel in row:
                if pixel == 0:
                    row_vals.append(numpy.array([white, white, white, visible]))
                elif pixel == 1:
                    row_vals.append(numpy.array([black, black, black, visible]))
            image_array.append(numpy.array(row_vals))
        self.image_array = numpy.array(image_array)

        self.x_level, self.y_level = numpy.ogrid[-self.image_array.shape[0] // 2:self.image_array.shape[0] // 2,
                                                 -self.image_array.shape[1] // 2:self.image_array.shape[1] // 2]
        self.stride = 3
        self.z_level = 0

    def _plot_pattern_and_camera(self):
        # plot the pattern template and camera
        self.figure = plt.figure()
        self.axis = self.figure.add_subplot(1, 2, 1, projection="3d")
        self.axis.plot_surface(self.x_level, self.y_level, self.z_level, rstride=self.stride, cstride=self.stride,
                               facecolors=self.image_array)
        self.axis.add_collection3d(Poly3DCollection(self.image.camera_faces, zorder=1))

    def _setup_axes(self):
        # setup axes as 10% bigger than largest value
        modifier = 1.1
        self.axis.set_ylim([-self.horizontal * modifier, self.horizontal * modifier])
        self.axis.set_xlim([-self.horizontal * modifier, self.horizontal * modifier])
        self.axis.set_zlim([0, self.max_z * modifier])  # the camera will never be below the QR code
        self.axis.set_xlabel("Distance in mm", fontsize=18)

    def _setup_source_comparison(self, image_path):
        # setup source image comparison
        image = plt.imread(image_path)
        rotated = ndimage.rotate(image, -90)
        arr = numpy.array(rotated)
        image_plot = self.figure.add_subplot(1, 2, 2)
        image_plot.get_xaxis().set_visible(False)
        image_plot.get_yaxis().set_visible(False)
        image_plot.imshow(arr)

    def draw_pose(self):
        # Use matplotlib to generate the 3d scene of the camera's pose compared to the pattern
        self._set_camera_faces()
        self._get_axis_limits()
        self._scale_pattern_image()
        self._plot_pattern_and_camera()
        self._setup_axes()
        self._setup_source_comparison(self.image.path)
        plt.show()
