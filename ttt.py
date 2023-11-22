import struct

hex_str = "713dcabe"
bytes_data = bytes.fromhex(hex_str)
float_value = struct.unpack('f', bytes_data)[0]

print(float_value)
