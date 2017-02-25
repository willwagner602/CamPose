import cv2
# turn off OpenCL because of compatibility issues
cv2.ocl.setUseOpenCL(False)
import numpy

from Image import Image
from Visualizer import Visualizer


class Camera:

    def __init__(self, qr_code, image_paths, camera_dimensions=(10, 10, 10)):

        self.pattern_path = qr_code.path
        self.image_paths = image_paths
        self.images = []
        self.qr_code = qr_code
        self.camera_dimensions = camera_dimensions
        self.pattern = cv2.imread(self.pattern_path)

        # these values are somewhat arbitrary but produce a reasonable result for high resolution images of QR codes
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Arrays to store object points and image points from all the images.
        self.objpoints = {}  # 3d point in real world space
        self.imgpoints = {}  # 2d points in image plane.

        self.camera_matrix = None
        self.distortion_coefficients = None

    def _identify_pattern_corners(self, image):
        image_keypoints, des2 = self.qr_code.ORB.detectAndCompute(image.image, None)
        brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = brute_force_matcher.match(self.qr_code.des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        image.source_points = numpy.float32([self.qr_code.keypoints[match.queryIdx].pt
                                             for match in matches]).reshape(-1, 1, 2)
        image.destination_points = numpy.float32([image_keypoints[match.trainIdx].pt for
                                                  match in matches]).reshape(-1, 1, 2)

    def _generate_pattern_corner_list(self):
        self.pattern_points = numpy.float32([[0, 0],
                                        [0, self.pattern_shape[0]],
                                        self.pattern_shape,
                                        [self.pattern_shape[1], 0]]).reshape(-1, 1, 2)

    def map_object_points_to_image(self, image):
        M, mask = cv2.findHomography(image.source_points, image.destination_points, cv2.RANSAC, 5.0)
        image_pattern_corners = cv2.perspectiveTransform(self.qr_code.mapping_coordinates_2d, M)
        micro_points = cv2.cornerSubPix(image.image, image_pattern_corners, (11, 11), (-1, -1), self.criteria)

        # If found, add object points, image points (after refining them)
        if micro_points.any():
            image.qr_code_points = self.qr_code.coordinates_3d
            image.image_points = numpy.array(micro_points).astype('float32')

    def _find_object_and_image_points(self):
        for image_path in self.image_paths:
            image = Image(image_path)
            self._identify_pattern_corners(image)
            self.map_object_points_to_image(image)
            self.images.append(image)

    def _generate_calibration_values(self,):
        objpoints = self._collect_qr_points()
        imgpoints = self._collect_image_points()
        val, cam_matrix, dist_coeffs, rvec, tvec = cv2.calibrateCamera(objpoints,
                                                                       imgpoints,
                                                                       self.images[0].image.shape[::-1], None, None)
        self.camera_matrix = cam_matrix
        self.distortion_coefficients = dist_coeffs

    def _collect_qr_points(self):
        points = []
        for image in self.images:
            points.append(image.qr_code_points)
        return points

    def _collect_image_points(self):
        points = []
        for image in self.images:
            points.append(image.image_points)
        return points

    def calibrate_camera(self):
        self._find_object_and_image_points()
        self._generate_calibration_values()

    def _match_pattern_points_in_image(self, image):
        # find list of matching points across pattern and image
        image_ORB = cv2.ORB_create()
        image_keypoints, des2 = image_ORB.detectAndCompute(image.image, None)
        brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = brute_force_matcher.match(self.qr_code.des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        image.pattern_matched_points = numpy.float32([self.qr_code.keypoints[match.queryIdx].pt
                                                      for match in matches]).reshape(-1, 1, 2)
        image.image_matched_points = numpy.float32([image_keypoints[match.trainIdx].pt
                                                    for match in matches]).reshape(-1, 1, 2)

    def _transform_pattern_corners_onto_image(self, image):
        # identify the points which delineate the pattern in the image
        transformation_matrix, mask = cv2.findHomography(image.pattern_matched_points,
                                                         image.image_matched_points,
                                                         cv2.RANSAC, 5.0)
        image.destination_pattern_points = cv2.perspectiveTransform(self.qr_code.display_coordinates_2d,
                                                                    transformation_matrix)

    def _generate_3d_points_for_known_corners(self, image):
        #  generate 3d points for known corners
        pattern_half_height = self.qr_code.length / 2 # mm
        pattern_half_width = self.qr_code.width / 2  # mm
        pattern_points_3d = numpy.array([
            [-pattern_half_width, pattern_half_height, 0],  # top left
            [-pattern_half_width, -pattern_half_height, 0],  # bottom left
            [pattern_half_width, -pattern_half_height, 0],  # bottom right
            [pattern_half_width, pattern_half_height, 0]  # top right
        ])
        image.pattern_corners = pattern_points_3d

    def _identify_camera_position(self, image):
        # run solvepnp to get an initial position for the origin in camera coordinates
        value, rvec, tvec = cv2.solvePnP(image.pattern_corners, image.destination_pattern_points,
                                         self.camera_matrix, self.distortion_coefficients)
        image.rotation_matrix, image.jacobian = cv2.Rodrigues(rvec)
        camera_position = -numpy.matrix(image.rotation_matrix).T * numpy.matrix(tvec)
        self._set_camera_position(camera_position, image)

    def _set_camera_position(self, camera_position, image):
        # set camera xyz location
        cam_x_center = camera_position[0].max()
        cam_y_center = camera_position[1].max()
        cam_z_center = camera_position[2].max()
        image.camera_center = (cam_x_center, cam_y_center, cam_z_center)

    def _identify_camera_attitude(self, image):
        # setup camera attitude
        projection_matrix = numpy.array([
            [self.rotation_matrix[0][0], self.rotation_matrix[0][1], self.rotation_matrix[0][2], 0],
            [self.rotation_matrix[1][0], self.rotation_matrix[1][1], self.rotation_matrix[1][2], 0],
            [self.rotation_matrix[2][0], self.rotation_matrix[2][1], self.rotation_matrix[2][2], 0]
        ])
        (self.cam_mat,
         rot_mat,
         trans_vect,
         rot_mat_x,
         rot_mat_y,
         rot_mat_z,
         self.euler_angles) = cv2.decomposeProjectionMatrix(projection_matrix)

    def _get_camera_position(self,  image):
        # Identify the camera's position (x, y, z) with the pattern as (0, 0, 0) in world coordinates
        self._match_pattern_points_in_image(image)
        self._transform_pattern_corners_onto_image(image)
        self._generate_3d_points_for_known_corners(image)
        self._identify_camera_position(image)

    def _generate_camera_points(self, image):
        # Create the array of 3d points that represents the camera in the scene
        cam_x_center, cam_y_center, cam_z_center = image.camera_center
        cam_width, cam_height, cam_depth = self.camera_dimensions
        cam_width_half = cam_width / 2
        cam_height_half = cam_height / 2

        top_left = (cam_x_center + cam_width_half, cam_y_center, cam_z_center + cam_height_half)
        top_right = (cam_x_center - cam_width_half, cam_y_center, cam_z_center + cam_height_half)
        bottom_right = (cam_x_center - cam_width_half, cam_y_center, cam_z_center - cam_height_half)
        bottom_left = (cam_x_center + cam_width_half, cam_y_center, cam_z_center - cam_height_half)
        back = (cam_x_center, cam_y_center - cam_width, cam_z_center)

        front_face = [top_left, bottom_left, bottom_right, top_right]
        left_face = [top_left, bottom_left, back]
        bottom_face = [bottom_left, bottom_right, back]
        right_face = [bottom_right, top_right, back]
        top_face = [top_left, top_right, back]

        image.camera_faces = [front_face, left_face, right_face, top_face, bottom_face]

    def process_images(self):
        for image_path in self.image_paths:
            image = Image(image_path)
            self._get_camera_position(image)
            self._generate_camera_points(image)
            visualizer = Visualizer(image, self.qr_code)
            visualizer.draw_pose()
