from google import genai
from google.genai import types
import cv2
import numpy as np
import json
from secrets import API_KEY
import position_calculator as pc


PROMPT = """
          Point to no more than 3 items in the image. The label returned
          should be an identifying name for the object detected.
          The image you receive is a horizontal stack of RGB and Depth image.
          The answer should follow the json format: [{"point": <point>,
          "label": <label1>}, "depth": <depth>, ...]. The points are in [x, y] format
          normalized to [0-480 for vertical axis, 0-640 for horizontal axis]. Depth is normalized to 0-1000.
        """
client = genai.Client(api_key=API_KEY)

RGB_img = cv2.imread("RGB_img.png")
Depth_img = cv2.imread("Depth_img.png")
Depth_img = cv2.normalize(Depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
Depth_img = cv2.applyColorMap(Depth_img, cv2.COLORMAP_JET)


combined = np.hstack((RGB_img, Depth_img)).astype(np.uint8)

# Now encode
success, encoded_image = cv2.imencode('.jpg', combined)

if not success:
    raise ValueError("Failed to encode image")
else:
    image_bytes = encoded_image.tobytes()

image_bytes = encoded_image.tobytes()

# cv2.imwrite("RGB_Depth_combined.png", combined)

# # Load your image
# with open("RGB_img.png", 'rb') as f:
#     image_bytes = f.read()

# with open("Depth_img.png", 'rb') as f:
#     depth_bytes = f.read()


image_response = client.models.generate_content(
    model="gemini-robotics-er-1.6-preview",
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/png',
        ),
        PROMPT
    ],
    config=types.GenerateContentConfig(
        temperature=1.0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
    )
)

print(image_response.text)

objects = json.loads(image_response.text)


for el in objects:
    print(el)
    x, y = el["point"]
    depth = el["depth"]
    print(x,y,depth)
    cv2.circle(RGB_img, (x, y), 5, (0, 0, 255), -1)

cv2.imwrite("RealSense_Detection.png", RGB_img)
            




# import intrinsics 
# intrinsics = pc.CameraIntrinsics(fx, fy, cx, cy)
# print(f"\n[Intrinseci camera]")    
# print(f"  fx={intrinsics.fx}, fy={intrinsics.fy}")
# print(f"  cx={intrinsics.cx}, cy={intrinsics.cy}")    
# print(f"\n  Matrice K:")    
# K = intrinsics.to_matrix()

# for i in range(3):        
#     print(f"    [{K[i, 0]:8.2f}  {K[i, 1]:8.2f}  {K[i, 2]:8.2f}]")

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
            

# u_pixel, v_pixel = width // 2, height // 2       
# depth_value = depth_image[int(round(v_pixel)), int(round(u_pixel))]

# if depth_value > 0:
#     point_camera = pc.deproject_pixel(u_pixel, v_pixel, depth_value, intrinsics)    
#     print(f"point_camera: {point_camera}")

#     point_waist = pc.transform_to_waist(point_camera, T_cam_to_waist)  
#     print(f"point_waist: {point_waist}")