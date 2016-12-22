"""
A short script that identifies camera pose from a QR code in the given list of images.
"""

import os

import cv2
import numpy
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def generate_camera_points(camera_center, camera_size):
    """
    Create the array of 3d points that represents the camera in the scene
    :param camera_center: camera (x, y, z) position in mm
    :param camera_size: camera device size (x, y) in mm
    :return: list of faces in a pyramid at attitude (0,0,0) with base representing axis originating at camera sensor,
    centered on camera_center. Faces are lists of tuples containing the points in each face.
    """
    cam_x_center, cam_y_center, cam_z_center = camera_center
    cam_width, cam_height = camera_size
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

    return [front_face, left_face, right_face, top_face, bottom_face]


def get_camera_position(pattern_path, pattern_width, pattern_buffer, image_path, camera_matrix, dist_coeffs):
    """
    Identify the camera's position (x, y, z) with the pattern as (0, 0, 0) in world coordinates
    :param pattern_path: the pattern at (0, 0, 0)
    :param pattern_width: the width of the pattern in mm
    :param pattern_buffer: the number of pixels between pattern and edge of image
    :param image_path: filepath to the image to analyze
    :param camera_matrix: opencv or self-generated camera matrix for the camera that took the image
    :param dist_coeffs: opencv or self-generated distortion coefficient matrix for the camera that took the image
    :return: camera center - camera position in (x, y, z)
             euler angles - angles describing camera attitude in (x, z, y) format (opencv default)
    """
    pattern = cv2.imread(pattern_path)
    image = cv2.imread(image_path)

    # find list of matching points across pattern and image
    orb = cv2.ORB_create()
    pattern_keypoints, des1 = orb.detectAndCompute(pattern, None)
    image_keypoints, des2 = orb.detectAndCompute(image, None)
    brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force_matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    pattern_matched_points = numpy.float32([pattern_keypoints[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    image_matched_points = numpy.float32([image_keypoints[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    # identify the points which delineate the pattern in the image
    transformation_matrix, mask = cv2.findHomography(pattern_matched_points, image_matched_points, cv2.RANSAC, 5.0)
    pattern_h, pattern_w, channels = pattern.shape
    edge_padding = pattern_buffer
    pattern_points = numpy.float32(
        [[0 + edge_padding, 0 + edge_padding],  # top left
         [0 + edge_padding, pattern_h - edge_padding],  # bottom left
         [pattern_w - edge_padding, pattern_h - edge_padding],  # bottom right
         [pattern_w - edge_padding, 0 + edge_padding]  # top right
        ]).reshape(-1, 1, 2)
    destination_pattern_points = cv2.perspectiveTransform(pattern_points, transformation_matrix)

    #  generate 3d points for known corners
    pattern_half_width = pattern_width / 2  #mm
    pattern_points_3d = numpy.array([
        [-pattern_half_width, pattern_half_width, 0],  # top left
        [-pattern_half_width, -pattern_half_width, 0],  # bottom left
        [pattern_half_width, -pattern_half_width, 0],  # bottom right
        [pattern_half_width, pattern_half_width, 0]  # top right
    ])

    # run solvepnp to get an initial position for the origin in camera coordinates
    value, rvec, tvec = cv2.solvePnP(pattern_points_3d, destination_pattern_points, camera_matrix, dist_coeffs)

    # translate vectors to
    rotation_matrix, jacobian = cv2.Rodrigues(rvec)
    cam_position = -numpy.matrix(rotation_matrix).T * numpy.matrix(tvec)

    # setup camera attitude
    projection_matrix = numpy.array([
        [rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], 0],
        [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], 0],
        [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], 0]
    ])
    (cam_mat, rot_mat, trans_vect, rot_mat_x,
     rot_mat_y, rot_mat_z, euler_angles) = cv2.decomposeProjectionMatrix(projection_matrix)

    # setup camera xyz location
    cam_x_center = cam_position[0].max()
    cam_y_center = cam_position[1].max()
    cam_z_center = cam_position[2].max()
    camera_center = (cam_x_center, cam_y_center, cam_z_center)

    return camera_center, euler_angles


def calibrate_camera(pattern_path, image_paths, pattern_size):
    """
    Produce camera and distortion matrixes for the given images
    :param pattern_path: path to pattern image
    :param image_paths: list of paths to images to use for calibration
    :param pattern_size: size of pattern (x, y) in mm
    :return: camera_matrix, distortion_coefficients: camera matrix and distortion coefficients in opencv formats
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    image_shapes = []

    orb = cv2.ORB_create()
    pattern = cv2.imread(pattern_path, 0)
    pattern_keypoints, des1 = orb.detectAndCompute(pattern, None)
    h, w = pattern.shape

    pattern_width, pattern_height = pattern_size
    pattern_half_width = pattern_width / 2
    pattern_half_height = pattern_height / 2

    object_points = numpy.array([
        [-pattern_half_width, pattern_half_height, 0],  # top left
        [-pattern_half_width, -pattern_half_height, 0],  # bottom left
        [pattern_half_width, -pattern_half_height, 0],  # bottom right
        [pattern_half_width, pattern_half_height, 0]  # top right
    ]).astype("float32")

    for image_path in image_paths:
        image = cv2.imread(image_path, 0)

        # find the pattern, then find the corners of the pattern in the image
        image_keypoints, des2 = orb.detectAndCompute(image, None)
        brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = brute_force_matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        source_points = numpy.float32([pattern_keypoints[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
        destination_points = numpy.float32([image_keypoints[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)
        pattern_points = numpy.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        image_pattern_corners = cv2.perspectiveTransform(pattern_points, M)

        micro_points = cv2.cornerSubPix(image, image_pattern_corners, (11, 11), (-1, -1), criteria)

        # If found, add object points, image points (after refining them)
        if micro_points.any():
            objpoints.append(object_points)
            imgpoints.append(numpy.array(micro_points).astype('float32'))
            image_shapes.append(image.shape[::-1])

    val, cam_matrix, dist_coeffs, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, image.shape[::-1], None, None)
    return cam_matrix, dist_coeffs


def draw_pose(camera_faces, template_size):
    """
    Use matplotlib to generate the 3d scene of the camera's pose compared to the pattern
    :param camera_faces: list of faces containing vertexes in (x, y, z) form
    :param template_size: template size in (x, y, z)
    """
    x = []
    y = []
    z = []
    for face in camera_faces:
        for point in face:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])

    # get maximum x, y, z for setting up axis limits
    max_horizontal = max(x + y)
    min_horizontal = min(x + y)
    horizontal = max((abs(max_horizontal), abs(min_horizontal), 300))
    max_z = max(z)

    # get pattern image and scale it to the appropriate size
    template_width, template_height, template_depth = template_size
    image_file = Image.open(pattern_path)
    image = image_file.rotate(-90).resize((template_width, template_height))
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
    image_array = numpy.array(image_array)

    x, y = numpy.ogrid[-image_array.shape[0] // 2:image_array.shape[0] // 2,
                       -image_array.shape[1] // 2:image_array.shape[1] // 2]
    stride = 3
    z_level = 0

    # plot the pattern template and camera
    figure = plt.figure()
    axis = figure.add_subplot(1, 2, 1, projection="3d")
    axis.plot_surface(x, y, z_level, rstride=stride, cstride=stride, facecolors=image_array)
    axis.add_collection3d(Poly3DCollection(camera_faces, zorder=1))

    # setup axes
    axis.set_ylim([-horizontal * 1.1, horizontal * 1.1])
    axis.set_xlim([-horizontal * 1.1, horizontal * 1.1])
    axis.set_zlim([0, max_z * 1.1])
    axis.set_xlabel("Distance in mm", fontsize=18)

    # setup source image comparison
    image = plt.imread(image_path)
    rotated = ndimage.rotate(image, -90)
    arr = numpy.array(rotated)
    image_plot = figure.add_subplot(1, 2, 2)
    image_plot.get_xaxis().set_visible(False)
    image_plot.get_yaxis().set_visible(False)
    image_plot.imshow(arr)

    plt.show()

if __name__ == "__main__":

    # turn off OpenCL because of incompatibility issues
    cv2.ocl.setUseOpenCL(False)

    base_path = os.getcwd()
    pattern = r"pattern.jpg"
    pattern_path = os.path.join(base_path, 'images', pattern)
    image_dir = os.path.join(os.getcwd(), 'images')
    images = [os.path.join(base_path, 'images', img) for img in os.listdir(image_dir) if 'IMG' in img]

    # qr code is 88mm x 88mm square
    qr_code_width = 88
    qr_code_depth = 0
    qr_code_edge_buffer = 40  # pixels

    # iphone 6 phone size
    camera_height = 138.1  # mm
    camera_width = 67  # mm

    # 9 isn't enough images for a good calibration, but better than other methods of guessing camera matrix
    print("Calibrating camera from images")
    cameraMatrix, distance_coefficients = calibrate_camera(pattern_path, images, (qr_code_width, qr_code_width))

    for image_path in images:
        cam_center, euler_angles = get_camera_position(pattern_path, qr_code_width, qr_code_edge_buffer, image_path,
                                                       cameraMatrix, distance_coefficients)
        print("Camera centered at {0:.1f}mm, {1:.1f}mm, {2:.1f}mm".format(cam_center[0], cam_center[1], cam_center[2]))
        camera_faces = generate_camera_points(cam_center, (camera_width, camera_height))
        draw_pose(camera_faces, (qr_code_width, qr_code_width, qr_code_depth))
