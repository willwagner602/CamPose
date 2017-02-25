import cv2
# turn off OpenCL because of compatibility issues
cv2.ocl.setUseOpenCL(False)


class Image:

    def __init__(self, image_path):
        # turn off OpenCL because of incompatibility issues
        cv2.ocl.setUseOpenCL(False)

        self.path = image_path
        self.image = cv2.imread(image_path, 0)
        self.identifier = self._generate_image_identifier()

        # camera pose information for this image
        self.camera_center = None
        self.camera_position = None
        self.camera_faces = None
        self.euler_angles = None

        # image specific matrices
        self.rotation_matrix = None
        self.jacobian = None

        self.source_points = None
        self.image_matched_points = None

        self.destination_points = None
        self.destination_pattern_points = None

        self.pattern_matched_points = None
        self.pattern_corners = None

        # lists of points used for calibration
        self.qr_code_points = []
        self.image_points = []

    def _generate_image_identifier(self):
        path_length = len(self.path)
        if path_length < 64:
            return self._strip_slashes(self.path)
        else:
            return self._strip_slashes(self.path[path_length - 64:])

    def _strip_slashes(self, name):
        return name.replace("\\", "").replace("/", "")