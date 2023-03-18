from django.shortcuts import render
import cv2
from django.http import StreamingHttpResponse
import yolov5,torch
import os
from ultralytics import YOLO
import random
from PIL import Image



data = ['person']
class_list=['person','cow','bottle']



# Create your views here.

def index(request):
    return render(request,'index.html')


#load model

 # Load model
# device = select_device(0)
# is_seg = '-seg' in str(WEIGHTS / 'yolov5m.pt')
# model = AutoBackend(WEIGHTS / 'yolov5m.pt', device=device, dnn=False, fp16=False)
# stride, names, pt = model.stride, model.names, model.pt
# imgsz = check_imgsz(imgsz, stride=stride)  # check image size

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model = YOLO("weights/yolov8n.pt", "v8")

frame_wid = 640
frame_hyt = 480

def stream():
    detection_colors = []
    for i in range(len(class_list)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
    detection_colors.append((b, g, r))
    cap=cv2.VideoCapture(0)
    while True:
        ret, frame=cap.read() 
        if not ret:
            print("Error: Image capturing failed")
            break
        detect_params = model.predict(source=[frame], conf=0.25, save=False)
        DP = detect_params[0].numpy()
        
        if len(DP) != 0:
            for i in range(len(detect_params[0])):
                
                
                boxes = detect_params[0].boxes
                box = boxes[i]  # returns one box
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]
                color=[255,200,0]
                if(clsID==0):
                    # code for IoT
                    # push notification to Arudino 
                    send_signals_to_box()
                    #  end of IoT
                    cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    color,
                    3,
                     ) 

                    # Display class name and confidence
                    font = cv2.FONT_HERSHEY_COMPLEX
                    cv2.putText(
                        frame,
                        "Object" + " " + str(round(conf, 3)) + "%",
                        (int(bb[0]), int(bb[1]) - 10),
                        font,
                        1,
                        (255, 255, 255),
                        2,
                    )



             

        # if(results.pandas().xyxy[0]['class']==0):
        #     print("person detected")
        # else:
        #     print("Other Object detected")
        image_bytes=cv2.imencode('.jpg',frame)[1].tobytes()
        # image_bytes =cv2.imencode('.jpg',frame)[1].tobytes()
        # yield(b'--frame\r\n' b'Content-type:image/jpeg\r\n\r\n'+image_bytes+b'r\n')
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')
        
        
           
def video_feed(request):
    return StreamingHttpResponse(stream(),content_type='multipart/x-mixed-replace; boundary=frame')

def send_signals_to_box():
    # code for IoT
    print("Sending signals to box")