from gimbaltest_pack import *
from serial.serialposix import Serial
serialPort = '/dev/ttyTHS0'  
baudRate = 115200  

if __name__=='__main__':
    ser=Serial(serialPort,baudRate,timeout=0.5)
    loc_test(pitch=0,yaw=0.395)
    loc_test(yaw=-0.395,pitch=0.2)