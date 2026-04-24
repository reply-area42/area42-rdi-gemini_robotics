from google import genai
from google.genai import types
import cv2
import numpy as np
import json
from secrets import API_KEY
import position_calculator as pc
from PIL import ImageColor, ImageDraw, ImageFont, Image

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

# RGB_img = cv2.imread("RGB_img.png")
# Depth_img = cv2.imread("Depth_img.png")
# Depth_img = cv2.normalize(Depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# Depth_img = cv2.applyColorMap(Depth_img, cv2.COLORMAP_JET)


# combined = np.hstack((RGB_img, Depth_img)).astype(np.uint8)

# # Now encode
# success, encoded_image = cv2.imencode('.jpg', combined)

# if not success:
#     raise ValueError("Failed to encode image")
# else:
#     image_bytes = encoded_image.tobytes()

# image_bytes = encoded_image.tobytes()

# cv2.imwrite("RGB_Depth_combined.png", combined)

# with open("RGB_img.png", 'rb') as f:
#     RGB_bytes = f.read()

# with open("Depth_img.png", 'rb') as f:
#     depth_bytes = f.read()


# image_response = call_model(image_bytes, PROMPT).text

image_response = """[
  {
    "obj": "basket",
    "box_2d": [121, 71, 788, 323],
    "polygon": [121, 192, 137, 269, 172, 308, 238, 321, 357, 319, 503, 300, 608, 281, 711, 252, 765, 222, 783, 179, 747, 134, 693, 108, 621, 93, 532, 81, 417, 79, 316, 91, 235, 114, 169, 142, 133, 169]
  },
  {
    "obj": "screwdriver",
    "box_2d": [421, 362, 794, 418],
    "polygon": [421, 365, 555, 379, 563, 371, 603, 369, 646, 372, 712, 375, 756, 381, 788, 396, 785, 413, 755, 416, 713, 410, 659, 404, 609, 399, 566, 394, 554, 385, 421, 370]
  },
  {
    "obj": "phone",
    "box_2d": [699, 0, 977, 78],
    "polygon": [699, 38, 774, 61, 848, 75, 878, 73, 911, 62, 946, 44, 969, 14, 959, 0, 934, 0, 750, 4, 716, 17, 703]
  }
]

"""

# objects = json.loads(image_response)
# print("\n\n")
# print(image_response)
# print("\n\n")

# # RGB_img = cv2.imread("RGB_img.png")
# with Image.open("RGB_img.png") as img:
#     masks = du.parse_segmentation_polygons(image_response, img_height=img.height, img_width=img.width)
#     img = du.plot_segmentation_masks(img, masks)
#     du.plot_bounding_boxes(img, image_response)
    


# import intrinsics 
# _, fx, fy, cx, cy = intrinsics.get_intrinsics()

fx=608.1531982421875
fy=608.2805786132812
cx=315.12713623046875
cy=259.9116516113281

intrinsics = pc.CameraIntrinsics(fx, fy, cx, cy)
print(f"\n[Intrinseci camera]")    
print(f"  fx={intrinsics.fx}, fy={intrinsics.fy}")
print(f"  cx={intrinsics.cx}, cy={intrinsics.cy}")    
print(f"\n  Matrice K:")    
K = intrinsics.to_matrix()

for i in range(3):        
    print(f"    [{K[i, 0]:8.2f}  {K[i, 1]:8.2f}  {K[i, 2]:8.2f}]")

# # rs -> torso
# # <origin xyz="0.0576235 0.01753 0.42987" rpy="0 0.8307767239493009 0"/>
# # torso -> waist (CoM)
# # <origin xyz="-0.0039635 0 0.044" rpy="0 0 0"/>



roll = 0.0                    
pitch = 0.8307767239493009    
yaw = 0.0                 

tx = 0.05366   
ty = 0.01753
tz = 0.47387


# Costruzione della matrice di rotazione e della trasformazione omogenea    
R_cam_to_waist = pc.build_rotation_matrix(roll, pitch, yaw)    
t_cam_to_waist = np.array([tx, ty, tz])    
T_cam_to_waist = pc.build_T_cam_to_waist(R_cam_to_waist, t_cam_to_waist)


            

# u_pixel, v_pixel = width // 2, height // 2       
# depth_value = depth_image[int(round(v_pixel)), int(round(u_pixel))]
depth_value = 600  # esempio di valore di profondità in millimetri
u_pixel, v_pixel = 640 // 2, 480 // 2  

if depth_value > 0:
    point_camera = pc.deproject_pixel(u_pixel, v_pixel, depth_value, intrinsics)    
    print(f"point_camera: {point_camera}")

    point_waist = pc.transform_to_waist(point_camera, T_cam_to_waist)  
    print(f"point_waist: {point_waist}")






















# import math
# import numpy as np



# # parameters

# pitch = 0.8307767239493009
# cy = math.cos(pitch)
# sy = math.sin(pitch)
# # Ry(pitch)
# Ry = np.array([[cy, 0, sy],
#                [0, 1, 0],
#                [-sy, 0, cy]], dtype=float)
# R_y = Ry


# t_rs_torso = np.array([0.0576235, 0.01753, 0.42987], dtype=float).reshape((3,))
# T_rs_torso = np.vstack([np.hstack([R_y, t_rs_torso.reshape(3,1)]), np.array([0,0,0,1], dtype=float)])


# t_torso_waist = np.array([-0.0039635, 0, 0.044], dtype=float)
# T_torso_waist = np.vstack([np.hstack([np.eye(3), t_torso_waist.reshape(3,1)]), np.array([0,0,0,1], dtype=float)])


# T_rs_waist = T_rs_torso @ T_torso_waist
# T_waist_rs = np.linalg.inv(T_rs_waist)
# T_rs_torso, T_torso_waist, T_rs_waist, T_waist_rs