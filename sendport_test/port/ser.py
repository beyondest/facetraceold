import numpy as np

from port.CRC import *
import serial
from serial.serialposix import Serial




def Serial_communication(yaw, pitch, fps, ser:Serial,is_autoaim=1):
    """yaw and pitch are in radian numbers"""

    if is_autoaim == 1:
        f1 = bytes("$", encoding='utf8')
        f2 = 10
        f3 = float(yaw)
        f4 = float(pitch)
        f5 = fps
    else:
        f1 = bytes("$", encoding='utf8')
        f2 = 10
        f3 = 0
        f4 = 0
        f5 = 1
    pch_Message1 = get_Bytes(f1, is_datalen_or_fps=0)
    pch_Message2 = get_Bytes(f2, is_datalen_or_fps=1)
    pch_Message3 = get_Bytes(f3, is_datalen_or_fps=0)
    pch_Message4 = get_Bytes(f4, is_datalen_or_fps=0)
    pch_Message5 = get_Bytes(f5, is_datalen_or_fps=2)
    pch_Message = pch_Message1 + pch_Message2 + pch_Message3 + pch_Message4 + pch_Message5

    wCRC = get_CRC16_check_sum(pch_Message, CRC16_INIT)
    ser.write(struct.pack("=cBffHi", f1, f2, f3, f4, f5, wCRC)) 