import threading
import numpy as np
from copy import deepcopy
import cv2
import time
import random

# Setup threading variables
exitFlag = False
t_frameLock = threading.Lock()
t_resultLock = threading.Lock()
t_patchLock = threading.Lock()
t_fpsLock = threading.Lock()


# Define the shared variables
t_last_preds = np.asarray([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]).astype(float)
t_last_prediction = None
t_pred_avail = False
t_frame_avail = False
t_recent_frame = None
t_frame_faces = None
t_display_frame = None
t_dfps = 1
t_cfps = 1
t_pfps = -1
t_fheight = 0
t_fwidth = 0

# Custom rounding
def round_custom(x, base=20):
    return x#int(base * round(float(x)/base))

# Define the capture class
class captureThread (threading.Thread):
    def __init__(self, threadID, exp_constant, cc_string, cap_device):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.exp_constant = exp_constant
        self.cc_string = cc_string
        self.capture_device = cap_device
    def run(self):

        # Define globals
        global exitFlag, t_frameLock, t_resultLock, t_patchLock, t_last_preds, t_last_prediction
        global t_pred_avail, t_frame_avail, t_recent_frame, t_frame_faces, t_display_frame
        global t_fpsLock, t_cfps, t_dfps, t_pfps, t_fheight, t_fwidth

        # Open the video capture
        print("Opening video capture...")
        face_cascade = cv2.CascadeClassifier(self.cc_string)
        video_capture = cv2.VideoCapture(self.capture_device)

        if not video_capture.isOpened():
            print("Error opening video capture")
            quit()

        # Capture the frame height and width from the video stream
        EXPANSION_CONSTANT = self.exp_constant
        F_HEIGHT = int(video_capture.get(4))
        F_WIDTH  = int(video_capture.get(3))
        t_fheight = F_HEIGHT
        t_fwidth = F_WIDTH

        while not exitFlag:
            start_time = time.clock()
            # Get the frame from the capture card
            ret, frame = video_capture.read()
            frame = cv2.flip(frame, 1)

            if frame is not None:

                # Extract the faces from the frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(32,32),flags=cv2.CASCADE_SCALE_IMAGE)
                last_frame = []

                # Write displayable information to the global vars
                t_frameLock.acquire()
                t_frame_faces = deepcopy(faces)
                t_display_frame = deepcopy(frame)
                t_frameLock.release()

                # Extract the frame patches
                for (x,y,w,h) in faces:
                    x = round_custom(x)
                    y = round_custom(y)
                    w = round_custom(w)
                    h = round_custom(h)
                    patch = frame[max(0,y-EXPANSION_CONSTANT):min(y+h+EXPANSION_CONSTANT,F_HEIGHT),max(0,x-EXPANSION_CONSTANT):min(x+w+EXPANSION_CONSTANT,F_WIDTH),:]
                    last_frame.append(cv2.resize(patch,(32,32)))
                    break
                    
                t_patchLock.acquire()
                t_recent_frame = deepcopy(last_frame)
                t_frame_avail = True
                t_patchLock.release()

            total_time = time.clock() - start_time
            t_fpsLock.acquire()
            t_cfps = total_time
            t_fpsLock.release()

        # Close the capture
        video_capture.release()        


# Define the display class
class displayThread (threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
    def run(self):
        #Define Globals
        global exitFlag, t_frameLock, t_resultLock, t_patchLock, t_last_preds, t_last_prediction
        global t_pred_avail, t_frame_avail, t_recent_frame, t_frame_faces, t_display_frame
        global t_fpsLock, t_cfps, t_dfps, t_pfps, t_fheight, t_fwidth
         
        while not exitFlag:
            start_time = time.clock()
            t_frameLock.acquire()
            frame = deepcopy(t_display_frame)
            faces = deepcopy(t_frame_faces)
            t_frameLock.release()

            F_HEIGHT = t_fheight
            F_WIDTH = t_fwidth

            if frame is not None:
                #t_fpsLock.acquire()
                #cv2.putText(frame,str(round(1/t_dfps,2)), (5,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
                #cv2.putText(frame,str(round(1/t_cfps,2)), (5,60), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
                if t_pfps == -1:
                    cv2.putText(frame,'Loading Predictor...', (5,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                #elif t_pfps > 100:
                #    pass
                #else:
                #    cv2.putText(frame,str(round(1/t_pfps,2)), (5,90), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
                #t_fpsLock.release()

                # Draw the prediction quantifiers in upper right
                d_face = False
                if t_pred_avail:
                    
                    t_resultLock.acquire()
                    if faces is not None:
                        for (x, y, w, h) in faces:
                            d_face = True

                            EXPANSION_CONSTANT=50
                            
                            x = round_custom(x)
                            y = round_custom(y)
                            w = round_custom(w)
                            h = round_custom(h)
                            cv2.putText(frame,t_last_prediction[0],
                                        (max(0,x-EXPANSION_CONSTANT+30),
                                         max(0, y-EXPANSION_CONSTANT+30)),
                                         cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                         t_last_prediction[1], 2
                                        )
                            cv2.rectangle(frame,(max(0,x-EXPANSION_CONSTANT),max(0,y-EXPANSION_CONSTANT)),
                                                (min(x+w+EXPANSION_CONSTANT, F_WIDTH),
                                                 min(y+h+EXPANSION_CONSTANT, F_HEIGHT)),
                                                 t_last_prediction[1], 3
                                          )
                            
                            overlay = frame.copy()
                            overlay_alpha = 0.8
                            cv2.rectangle(overlay, (F_WIDTH-(260 if d_face else 100),0), (F_WIDTH, 125), (0,0,0), -1)
                            cv2.addWeighted(overlay, overlay_alpha, frame, 1-overlay_alpha, 0, frame)

                            #Draw the prediction bars
                            BASE_WIDTH = F_WIDTH - 100
                            BASE_EXT = -150
                            
                            cv2.line(frame,(BASE_WIDTH, 115), (int(BASE_WIDTH +  BASE_EXT*t_last_preds[7]), 115), (0,0,255) if np.argmax(t_last_preds) == 7 else (0, 255, 0), 2, 8, 0)
                            cv2.line(frame,(BASE_WIDTH, 100), (int(BASE_WIDTH +  BASE_EXT*t_last_preds[6]), 100), (0,0,255) if np.argmax(t_last_preds) == 6 else (0, 255, 0), 2, 8, 0)
                            cv2.line(frame,(BASE_WIDTH, 85), (int(BASE_WIDTH +  BASE_EXT*t_last_preds[5]), 85), (0,0,255) if np.argmax(t_last_preds) == 5 else (0, 255, 0), 2, 8, 0)
                            cv2.line(frame,(BASE_WIDTH, 70), (int(BASE_WIDTH +  BASE_EXT*t_last_preds[4]), 70), (0,0,255) if np.argmax(t_last_preds) == 4 else (0, 255, 0), 2, 8, 0)
                            cv2.line(frame,(BASE_WIDTH, 55), (int(BASE_WIDTH +  BASE_EXT*t_last_preds[3]), 55), (0,0,255) if np.argmax(t_last_preds) == 3 else (0, 255, 0), 2, 8, 0)
                            cv2.line(frame,(BASE_WIDTH, 40), (int(BASE_WIDTH +  BASE_EXT*t_last_preds[2]), 40), (0,0,255) if np.argmax(t_last_preds) == 2 else (0, 255, 0), 2, 8, 0)
                            cv2.line(frame,(BASE_WIDTH, 25), (int(BASE_WIDTH +  BASE_EXT*t_last_preds[1]), 25), (0,0,255) if np.argmax(t_last_preds) == 1 else (0, 255, 0), 2, 8, 0)
                            cv2.line(frame,(BASE_WIDTH, 10), (int(BASE_WIDTH +  BASE_EXT*t_last_preds[0]), 10), (0,0,255) if np.argmax(t_last_preds) == 0 else (0, 255, 0), 2, 8, 0)


                            break
                    t_resultLock.release()

                if not d_face:
                    overlay = frame.copy()
                    overlay_alpha = 0.8
                    cv2.rectangle(overlay, (F_WIDTH-(260 if d_face else 100),0), (F_WIDTH, 125), (0,0,0), -1)
                    cv2.addWeighted(overlay, overlay_alpha, frame, 1-overlay_alpha, 0, frame)

                cv2.putText(frame,"Neutral",   (F_WIDTH - 90,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255,255,255), 1)
                cv2.putText(frame,"Happiness", (F_WIDTH - 90,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0,255,255), 1)
                cv2.putText(frame,"Sadness",   (F_WIDTH - 90,45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255,144,30), 1)
                cv2.putText(frame,"Surprise",  (F_WIDTH - 90,60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255,255,0), 1)
                cv2.putText(frame,"Fear",      (F_WIDTH - 90,75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255,0,255), 1)
                cv2.putText(frame,"Disgust",   (F_WIDTH - 90,90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0,255,0), 1)
                cv2.putText(frame,"Anger",     (F_WIDTH - 90,105), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0,0,255), 1)
                cv2.putText(frame,"Contempt",  (F_WIDTH - 90,120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0,153,255), 1)

                
                
                
                cv2.imshow('RealtimeFR', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                exitFlag = True
                break

            total_time = time.clock() - start_time
            t_fpsLock.acquire()
            t_dfps = total_time
            t_fpsLock.release()

class predictionThread(threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
    def run(self):
        #Define Globals
        global exitFlag, t_frameLock, t_resultLock, t_patchLock, t_last_preds, t_last_prediction
        global t_pred_avail, t_frame_avail, t_recent_frame, t_frame_faces, t_display_frame
        global t_fpsLock, t_cfps, t_dfps, t_pfps, t_fheight, t_fwidth

        # Setup imports
        import tflearn
        import RealTimeMTModel

        model = RealTimeMTModel.get_model('model_resnet_cifar10-64000')

        NUM_SMOOTHING_FRAMES = 10

        while not exitFlag:
            start_time = time.clock()

            t_patchLock.acquire()
            recent_frame = deepcopy(t_recent_frame)
            t_patchLock.release()

            if recent_frame is not None and not len(recent_frame) == 0:
                patch = recent_frame[0]
                pred = model.predict(np.expand_dims(np.divide(patch.astype('float32'),255.0),axis=0))
                
                #pred = [[random.random() for _ in range(8)]]
                #pred /= np.max(np.abs(pred),axis=0)

                # Update prediction smoothing
                t_resultLock.acquire()
                t_last_preds = (np.asarray(pred[0])*(1/float(NUM_SMOOTHING_FRAMES)) + t_last_preds*((NUM_SMOOTHING_FRAMES-1)/float(NUM_SMOOTHING_FRAMES))).astype(float)
                t_last_prediction = RealTimeMTModel.get_emotion(t_last_preds)
                t_pred_avail = True
                t_resultLock.release()

            # Update time
            #time.sleep(0.25)
            total_time = time.clock() - start_time
            t_fpsLock.acquire()
            t_pfps = total_time
            t_fpsLock.release()

def main():

    global exitFlag, t_frameLock, t_resultLock, t_patchLock, t_last_preds, t_last_prediction
    global t_pred_avail, t_frame_avail, t_recent_frame, t_frame_faces, t_display_frame
	
    print("Building threads...")
    #t_cap = captureThread(1, 50, 'haarcascade_frontalface_default.xml', 0)
    t_cap = captureThread(1, 50, 'lbpcascade_frontalface_improved.xml', 0)
    t_disp = displayThread(2)
    t_pred = predictionThread(3)

    print("Starting threads...")
    t_cap.start()
    t_disp.start()
    t_pred.start()

    try:
        while not exitFlag:
            time.sleep(0.001)
    except KeyboardInterrupt:
        exitFlag = True

        
    t_disp.join()
    t_pred.join()
    t_cap.join()
    print("Done.")
    

if __name__ == "__main__": main()
        
