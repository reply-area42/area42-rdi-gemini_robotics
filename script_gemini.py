from google import genai
from google.genai import types
import cv2
import numpy as np
import json
from secrets_API import API_KEY
import position_calculator as pc
from PIL import ImageColor, ImageDraw, ImageFont, Image

import logging_mp

# Configura il logging PRIMA di altri import
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)


import drawing_utils as du

from call_gemini import call_model


# PROMPT = """
#           Point to center no more than 3 items in the image. The label returned
#           should be an identifying name for the object detected.
#           The image you receive is a horizontal stack of RGB and Depth image.
#           The answer should follow the json format: [{"point": <point>,
#           "label": <label1>}, "depth": <depth>, ...]. The points are in [y,x] format
#           normalized to 0-1000. Depth is normalized to 0-1000.
#         """

# PROMPT = """
#         Detect max 3 objects. Return JSON: {"obj": name, 
#         "box_2d": [ymin, xmin, ymax, xmax], 
#         "polygon": [y1, x1, y2, x2, ...]}. Use normalized 0-1000 coordinates. No prose.
#         """

PROMPT = """
        Place a point on the screwdriver, then 15 points for the trajectory of
        moving the screwdriver to the top of the basket on the left.
        The points should be labeled by order of the trajectory, from '0'
        (start point at right hand) to <n> (final point)
        The answer should follow the json format:
        [{"point": <point>, "label": <label1>}, ...].
        The points are in [y, x] format normalized to 0-1000.
        """

# RGB_img = cv2.imread("images/RGB_img.png")
# Depth_img = cv2.imread("images/Depth_img.png")
# Depth_img = cv2.normalize(Depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# Depth_img = cv2.applyColorMap(Depth_img, cv2.COLORMAP_JET)


# combined = np.hstack((RGB_img, Depth_img)).astype(np.uint8)

# # Now encode
# success, encoded_image = cv2.imencode('.jpg', combined)


rgb_image = cv2.imread("images/RGB_img.png")

# Codifica solo l'immagine RGB per Gemini
success, encoded_image = cv2.imencode('.jpg', rgb_image)
# if not success:
#     logger_mp.error("Failed to encode rgb image")
#     exit(1)
# image_bytes = encoded_image.tobytes()

if not success:
    raise ValueError("Failed to encode image")
else:
    image_bytes = encoded_image.tobytes()


# cv2.imwrite("images/RGB_Depth_combined.png", combined)

# with open("images/RGB_img.png", 'rb') as f:
#     RGB_bytes = f.read()

# with open("images/Depth_img.png", 'rb') as f:
#     depth_bytes = f.read()


#image_response = call_model(image_bytes, PROMPT).text

# image_response = """[
#   {
#     "obj": "basket",
#     "box_2d": [121, 71, 788, 323],
#     "polygon": [121, 192, 137, 269, 172, 308, 238, 321, 357, 319, 503, 300, 608, 281, 711, 252, 765, 222, 783, 179, 747, 134, 693, 108, 621, 93, 532, 81, 417, 79, 316, 91, 235, 114, 169, 142, 133, 169]
#   },
#   {
#     "obj": "screwdriver",
#     "box_2d": [421, 362, 794, 418],
#     "polygon": [421, 365, 555, 379, 563, 371, 603, 369, 646, 372, 712, 375, 756, 381, 788, 396, 785, 413, 755, 416, 713, 410, 659, 404, 609, 399, 566, 394, 554, 385, 421, 370]
#   },
#   {
#     "obj": "phone",
#     "box_2d": [699, 0, 977, 78],
#     "polygon": [699, 38, 774, 61, 848, 75, 878, 73, 911, 62, 946, 44, 969, 14, 959, 0, 934, 0, 750, 4, 716, 17, 703]
#   }
# ]

# """

image_response = """
[
  { "point": [645, 781], "label": "0" },
  { "point": [611, 765], "label": "1" },
  { "point": [576, 747], "label": "2" },
  { "point": [542, 727], "label": "3" },
  { "point": [507, 706], "label": "4" },
  { "point": [471, 683], "label": "5" },
  { "point": [436, 659], "label": "6" },
  { "point": [402, 633], "label": "7" },
  { "point": [366, 606], "label": "8" },
  { "point": [330, 577], "label": "9" },
  { "point": [296, 547], "label": "10" },
  { "point": [262, 516], "label": "11" },
  { "point": [226, 484], "label": "12" },
  { "point": [193, 451], "label": "13" },
  { "point": [159, 417], "label": "14" },
  { "point": [126, 381], "label": "15" }
]

"""
trajectory_data = json.loads(image_response)
with open("response.json", "w") as f:
    json.dump(trajectory_data, f, indent=2)


objects = json.loads(image_response)
print("\n\n")
#print(image_response)
print("\n\n")

# RGB_img = cv2.imread("RGB_img.png")
# with Image.open("images/RGB_img.png") as img:
#     masks = du.parse_segmentation_polygons(image_response, img_height=img.height, img_width=img.width)
#     img = du.plot_segmentation_masks(img, masks)
#     du.plot_bounding_boxes(img, image_response)

with Image.open("images/RGB_img.png") as img: 
    du.draw_trajectory_points("images/RGB_img.png", "response.json", "images/trajectory_points_overlay.png")

    


# from camera.intrinsics import intrinsics
# _, fx, fy, cx, cy = intrinsics.get_intrinsics()

# fx=608.1531982421875
# fy=608.2805786132812
# cx=315.12713623046875
# cy=259.9116516113281

intrinsics = pc.CameraIntrinsics(None)

# K = intrinsics.to_matrix()



# # rs -> torso
# # <origin xyz="0.0576235 0.01753 0.42987" rpy="0 0.8307767239493009 0"/>
# # torso -> waist (CoM)
# # <origin xyz="-0.0039635 0 0.044" rpy="0 0 0"/>



# roll = 0.0                    
# pitch = 0.8307767239493009    
# yaw = 0.0                 

# tx = 0.05366   
# ty = 0.01753
# tz = 0.47387


# # Costruzione della matrice di rotazione e della trasformazione omogenea    
# R_cam_to_waist = pc.build_rotation_matrix(roll, pitch, yaw)    
# t_cam_to_waist = np.array([tx, ty, tz])    
# T_cam_to_waist = pc.build_T_cam_to_waist(R_cam_to_waist, t_cam_to_waist)


            

# # u_pixel, v_pixel = width // 2, height // 2       
# # depth_value = depth_image[int(round(v_pixel)), int(round(u_pixel))]
# depth_value = 600  # esempio di valore di profondità in millimetri
# u_pixel, v_pixel = 640 // 2, 480 // 2

# if depth_value > 0:
#     point_camera = pc.deproject_pixel(u_pixel, v_pixel, depth_value, intrinsics)    
#     print(f"point_camera: {point_camera*1000}")  # in millimetri

#     point_waist = pc.transform_to_waist(point_camera, T_cam_to_waist)  
#     print(f"point_waist: {point_waist*1000}") # in millimetri


# # target_pos = point_waist  # il tuo punto nel frame waist

# # # Orientamento: identità = polso allineato agli assi del waist
# # # Oppure copia l'orientamento corrente da tele_data
# # target_pose = np.eye(4)
# # target_pose[:3, 3] = target_pos
# # # target_pose[:3, :3] = desired_rotation  # se hai un orientamento specifico


