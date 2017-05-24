import numpy as np
import cv2
import time
import random

print("Importing tflearn...")
import tflearn

def GetEmotion(ival):
	emotion = np.argmax(ival)
	switcher = {
		0: ("Anger",(0,0,255)),
		1: ("Disgust",(0,255,0)),
		2: ("Fear",(255,0,255)),
		3: ("Happiness",(0,255,255)),
		4: ("Neutral",(0,0,0)),
		5: ("Sadness",(255,0,0)),
		6: ("Surprise",(255,255,0))
	}
	return switcher.get(emotion, ("N/A",(0,0,0)))

# First we load the network
print("Setting up neural networks...")
n = 18

# Real-time data preprocessing
print("Doing preprocessing...")
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True, mean=[83.11,86.75,110.358])

# Real-time data augmentation
print("Building augmentation...")
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([32, 32], padding=4)

#Build the model (for 32 x 32)
print("Shaping input data...")
net = tflearn.input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)

print("Carving Resnext blocks...")
net = tflearn.resnext_block(net, n, 16, 32)
net = tflearn.resnext_block(net, 1, 32, 32, downsample=True)
net = tflearn.resnext_block(net, n-1, 32, 32)
net = tflearn.resnext_block(net, 1, 64, 32, downsample=True)
net = tflearn.resnext_block(net, n-1, 64, 32)

print("Erroding Gradient...")
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
net = tflearn.fully_connected(net, 7, activation='softmax')
opt = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=opt,
                         loss='categorical_crossentropy')
						 
print("Structuring model...")
model = tflearn.DNN(net, checkpoint_path='model_resnet_cifar10',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)


# Load the model from checkpoint
print("Loading the model...")
model.load('model_resnet_cifar10-37000')


print("Opening video capture...")
face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error opening video capture")
    quit()

EXPANSION_CONSTANT=50
F_HEIGHT =  int(video_capture.get(4))
F_WIDTH = int(video_capture.get(3))



# Setup the smoothing
from collections import Counter
NUM_SMOOTHING_FRAMES=7 
current_frame = 0
last_preds = np.asarray([1.0,1.0,1.0,1.0,1.0,1.0,1.0]).astype(float)

pred_made = False
last_prediction = None

print("Capturing...")
while(True):
        start_time = time.clock()
        ret, frame = video_capture.read()
        if frame is not None:
            pred_made = False
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(32,32))
            for (x,y,w,h) in faces:
                patch = frame[max(0,y-EXPANSION_CONSTANT):min(y+h+EXPANSION_CONSTANT,F_HEIGHT),max(0,x-EXPANSION_CONSTANT):min(x+w+EXPANSION_CONSTANT,F_WIDTH),:]
                if patch is not None:
                        patch = cv2.resize(patch,(32,32))
                        pred = model.predict(np.expand_dims(patch.astype('float32'),axis=0))
                        #pred = [random.random() for _ in range(7)]
                        #pred /= np.max(np.abs(pred),axis=0)
                        last_preds = (np.asarray(pred[0])*(1/float(NUM_SMOOTHING_FRAMES)) + last_preds*((NUM_SMOOTHING_FRAMES-1)/float(NUM_SMOOTHING_FRAMES))).astype(float)
                        pred_made = True
                        last_prediction = GetEmotion(last_preds)
                        current_frame += 1

                        # Draw the prediction quantifiers in lower left
                        cv2.putText(frame,"Anger", (10,F_HEIGHT-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(frame,"Disgust", (10,F_HEIGHT-20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(frame,"Fear", (10,F_HEIGHT-30), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(frame,"Happiness", (10,F_HEIGHT-40), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(frame,"Neutral", (10,F_HEIGHT-50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(frame,"Sadness", (10,F_HEIGHT-60), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                        cv2.putText(frame,"Surprise", (10,F_HEIGHT-70), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)

                        BASE_WIDTH = 100
                        BASE_EXT = 150
                        cv2.line(frame,(BASE_WIDTH, F_HEIGHT-15), (int(BASE_WIDTH +  BASE_EXT*last_preds[0]), F_HEIGHT-15), (0,0,255) if np.argmax(last_preds) == 0 else (0, 255, 0), 2, 8, 0)
                        cv2.line(frame,(BASE_WIDTH, F_HEIGHT-25), (int(BASE_WIDTH +  BASE_EXT*last_preds[1]), F_HEIGHT-25), (0,0,255) if np.argmax(last_preds) == 1 else (0, 255, 0), 2, 8, 0)
                        cv2.line(frame,(BASE_WIDTH, F_HEIGHT-35), (int(BASE_WIDTH +  BASE_EXT*last_preds[2]), F_HEIGHT-35), (0,0,255) if np.argmax(last_preds) == 2 else (0, 255, 0), 2, 8, 0)
                        cv2.line(frame,(BASE_WIDTH, F_HEIGHT-45), (int(BASE_WIDTH +  BASE_EXT*last_preds[3]), F_HEIGHT-45), (0,0,255) if np.argmax(last_preds) == 3 else (0, 255, 0), 2, 8, 0)
                        cv2.line(frame,(BASE_WIDTH, F_HEIGHT-55), (int(BASE_WIDTH +  BASE_EXT*last_preds[4]), F_HEIGHT-55), (0,0,255) if np.argmax(last_preds) == 4 else (0, 255, 0), 2, 8, 0)
                        cv2.line(frame,(BASE_WIDTH, F_HEIGHT-65), (int(BASE_WIDTH +  BASE_EXT*last_preds[5]), F_HEIGHT-65), (0,0,255) if np.argmax(last_preds) == 5 else (0, 255, 0), 2, 8, 0)
                        cv2.line(frame,(BASE_WIDTH, F_HEIGHT-75), (int(BASE_WIDTH +  BASE_EXT*last_preds[6]), F_HEIGHT-75), (0,0,255) if np.argmax(last_preds) == 6 else (0, 255, 0), 2, 8, 0)

                if pred_made:
                        cv2.putText(frame,last_prediction[0], (max(0,x-EXPANSION_CONSTANT + 20), max(0, y-EXPANSION_CONSTANT + 20)), cv2.FONT_HERSHEY_PLAIN, 1, last_prediction[1], 1)
                        cv2.rectangle(frame,(max(0,x-EXPANSION_CONSTANT),
                                      max(0, y-EXPANSION_CONSTANT)),
                                      (x+w+EXPANSION_CONSTANT,
                                      y+h+EXPANSION_CONSTANT),
                                      last_prediction[1],2)
                
                break
                
            total_time = time.clock() - start_time
            cv2.putText(frame,str(round(1/total_time,2)), (5,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)    
            cv2.imshow('RealtimeFR', frame)
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
