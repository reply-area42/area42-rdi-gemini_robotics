import psutil
import socket


def find_interface_with_subnet(prefix="192.168.123."):
    addrs = psutil.net_if_addrs()
    for iface_name, iface_addresses in addrs.items():
        for addr in iface_addresses:
            if addr.family == socket.AF_INET and addr.address.startswith(prefix):
                return iface_name
    return None


if __name__ == "__main__":
    iface = find_interface_with_subnet()
    if iface is None:
        print("No interface found with IP 192.168.123.*")
    else:
        print(iface)
