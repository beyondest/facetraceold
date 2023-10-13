import numpy as np
import cv2
import math
from CRC import *
import serial



# 串口参数
serialPort = '/dev/ttyTHS0'  # 串口
baudRate = 115200  # 波特率
ser = serial.Serial(serialPort, baudRate, timeout=0.5)

def Serial_communication(yaw, pitch, fps, is_autoaim = 1):
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
    ser.write(struct.pack("=cBffHi", f1, f2, f3, f4, f5, wCRC))  #分别是帧头，长度，数据，数据，fps，校验