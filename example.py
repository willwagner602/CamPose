import os

from Camera import Camera
from QRCode import QRCode

if __name__ == "__main__":
    base_path = os.getcwd()
    pattern = r"pattern.jpg"
    pattern_path = os.path.join(base_path, 'images', pattern)
    image_dir = os.path.join(os.getcwd(), 'images')
    images = [os.path.join(base_path, 'images', img) for img in os.listdir(image_dir) if 'IMG' in img]

    # qr code is an 88mm x 88mm square
    qr_code_width = 88
    qr_code_edge_buffer = 40  # pixels
    qr_code = QRCode(pattern_path, (qr_code_width, qr_code_width), qr_code_edge_buffer)

    # iphone 6 phone size
    camera_height = 138.1  # mm
    camera_width = 67  # mm

    camera = Camera(qr_code, images, (camera_height, camera_width, 10))

    camera.calibrate_camera()
    camera.process_images()