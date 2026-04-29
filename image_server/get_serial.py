import pyrealsense2 as rs

def list_serials():
    ctx = rs.context()               # query connected devices
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
        return

    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        product = dev.get_info(rs.camera_info.product_id)
        print(f"Device {i}: {name} | Product ID: {product} | Serial: {serial}")

if __name__ == "__main__":
    list_serials()
