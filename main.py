import tqdm
from camera import control
import img_operation as imo
import os_operation as oso
import cv2
import mydetect
import mydataloaders as mp
import numpy as np
import os
import time
from camera import mvsdk
from port.ser import Serial_communication
from serial.serialposix import Serial

wangyiw='yoloface.pt'
facew='face.pt'
yolov5sw='yolov5s.pt'
yolov5soldw='yolov5sold.pt'
img_path='people.jpg'

process_imgsz=(640,640)
camera_center=(320,320)

kp=1
ki=0.01
kd=0.01
pid_shape=(2,1)


#gimbal is whether to send to port
gimbal=False
save_img=True
save_img_path='./out'
show=False

serialPort = '/dev/ttyTHS0'  
baudRate = 115200  


def main():


    count=0
    '''camera init part'''
    hcamera=control.camera_init(mvsdk.CAMERA_MEDIA_TYPE_BGR8)
    control.isp_init(hcamera,2000)
    #out=control.save_video_camera_init(out_path,name='fuck2.mp4',codec='AVC1')
    camera_info=control.get_all(hcamera)
    control.print_getall(camera_info)
    control.camera_open(hcamera)
    pframebuffer_address=control.camera_setframebuffer()
    '''pid init'''
    pid=imo.PIDtrace(kp,ki,kd,pid_shape)


    #press esc to end
    while (cv2.waitKey(1) & 0xFF) != 27:
        t1=cv2.getTickCount()
        dst=control.grab_img(hcamera,pframebuffer_address)
        
        dst=cv2.resize(dst,process_imgsz,interpolation=cv2.INTER_LINEAR)
        
        dst,dia_list=mydetect.myrun(source=dst,weights=yolov5soldw,draw_img=True,classes=0)
        t2=cv2.getTickCount()
        fps=(t2-t2)/cv2.getTickFrequency()
        fps=20
        cv2.circle(dst,camera_center,10,(125,125,255),-1)
        if len(dia_list)>0:
            count+=1
            dst,center=imo.drawrec_and_getcenter(dia_list,dst,camera_center)
            pid_value=pid.update(camera_center,center)
            dst=imo.draw_pid_vector(dst,camera_center,pid_value)
            yaw=round(np.arctan2(pid_value[0],camera_center[0])[0],3)
            pitch=round(np.arctan2(pid_value[1],camera_center[1])[0],3)
            
            
            imo.add_text(dst,'fps',fps,(0,200))
            imo.add_text(dst,'yaw',yaw,(0,300))
            imo.add_text(dst,'pitch',pitch,(0,400))
            
            if gimbal:
                
                Serial_communication(yaw,pitch,fps)
                
            if save_img:
                write_path=os.path.join(save_img_path,f'{count}.jpg')
                cv2.imwrite(write_path,dst)
            
            print(f'**********yaw={yaw}*********')
            print(f'**********pitch={pitch}**********')
        #out.write(dst)
        if show:
            cv2.imshow('press esc to end',dst)
            
        
        print(f'**********fps={fps}***************')
        
        
        
    #cv2.destroyAllWindows()
    control.camera_close(hcamera,pframebuffer_address)
    #out.release()
if __name__=='__main__':
    if gimbal:
        ser=Serial(serialPort,baudRate,timeout=0.5)
    main()
