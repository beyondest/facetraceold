from port.ser import Serial_communication
import time
import math
pi=math.pi

def turn_around(one_cycle_time_s:int=10,
                fps:int=10,
                yaw_start=0,
                yaw_end=0,
                pitch_start=0,
                pitch_end=0,
                begin_sleep_time_s=2,
                mode:str='yaw'
                ):
    times=one_cycle_time_s*fps
    each=2*pi/times
    sleep_time=1/fps
    
    #init
    Serial_communication(yaw_start,pitch_start,fps)
    time.sleep(begin_sleep_time_s)
    #turn
    for i in range(times):
        if mode=='yaw':
            yaw_start+=each
        elif mode=='pitch':
            pitch_start+=each
        Serial_communication(yaw_start,pitch_start,fps)
        time.sleep(sleep_time)
    #uninit
    Serial_communication(yaw_end,pitch_end,fps)
    
    
    
def loc_test(yaw:int=0.79,pitch:int=0.79,fps:int=20):
    for i in range(10):
        Serial_communication(yaw,pitch,fps)
        time.sleep(0.5)
        


        
        