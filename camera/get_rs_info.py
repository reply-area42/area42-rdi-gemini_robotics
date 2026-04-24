import pyrealsense2 as rs

def get_rs_info():

    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("Nessuna camera RealSense trovata.")
        return None
    else:
        IDs = []
        for i, dev in enumerate(devices):
            print(f"Camera {i}:")
            print(f"  Nome:          {dev.get_info(rs.camera_info.name)}")
            print(f"  Serial Number: {dev.get_info(rs.camera_info.serial_number)}")
            print(f"  Firmware:      {dev.get_info(rs.camera_info.firmware_version)}")
        IDs.append(dev.get_info(rs.camera_info.serial_number))
        return IDs
    
if __name__ == "__main__":
    get_rs_info()