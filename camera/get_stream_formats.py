import pyrealsense2 as rs

ctx = rs.context()
for dev in ctx.query_devices():
    print('Device:', dev.get_info(rs.camera_info.serial_number))
    for sensor in dev.query_sensors():
        print(' Sensor:', sensor.get_info(rs.camera_info.name))
        for profile in sensor.get_stream_profiles():
            try:
                vp = profile.as_video_stream_profile()
                res = str(vp.width()) + 'x' + str(vp.height())
                print('  ', str(profile.stream_type()).ljust(20), '|', res.ljust(10), '|', str(vp.fps()) + 'fps', '|', profile.format())
            except:
                pass
