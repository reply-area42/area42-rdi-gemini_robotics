import pyrealsense2 as rs

import get_rs_info as gri

def get_intrinsics():

    pipeline = rs.pipeline()
    config = rs.config()
    ID = gri.get_rs_info()[0]
    config.enable_device(f"{ID}") # il serial number della tua D435i
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    pipeline.stop()

    return intrinsics, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy

if __name__ == "__main__":
    intrinsics = get_intrinsics()

    print(f"fx = {intrinsics.fx}")
    print(f"fy = {intrinsics.fy}")
    print(f"cx = {intrinsics.ppx}")  # ppx = principal point x
    print(f"cy = {intrinsics.ppy}")  # ppy = principal point y
    print(f"distortion model = {intrinsics.model}")
    print(f"distortion coeffs = {intrinsics.coeffs}")
    print(f"width = {intrinsics.width}")
    print(f"height = {intrinsics.height}")

