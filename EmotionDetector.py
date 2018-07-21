from keras.models import model_from_json, load_model
from keras.optimizers import RMSprop
import numpy as np
import sys
from scipy.spatial import distance as dist
import dlib
from wide_resnet import WideResNet
from imutils import face_utils
import config
import os
import wmi
from time import sleep
from scipy.ndimage import zoom
import cv2
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
#model = model_from_json(open('./models/Face_model_architecture.json').read())
# model.load_weights('./models/Face_model_weights.h5')
model = load_model('./models/CNN.hdf5')
model_gender = load_model('./models/gender.hdf5')
modelage = WideResNet(64, depth=16, k=8)()
modelage.load_weights('./models/age.hdf5')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
recognizer = cv2.face.LBPHFaceRecognizer_create()
COUNTER = config.COUNTER
plt.axis([0, 10, 0, 1])
plt.ion()
path = config.facerecogpath
emo_vect_prec=[0]*7
j=0
color_vect=['r','g','b','c','m','k','y']
label=['angry','disgust','fear','happy','sad','surprise','neutral']
# optimizer = RMSprop(lr=0.001,
#                     rho=0.9,
#                     epsilon=1e-08, 
#                     decay=0.0)
# model.compile(optimizer = optimizer ,
#               loss = "categorical_crossentropy",
#               metrics=["accuracy"]) 
#               

def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    i=0
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = cv2.imread(image_path,0)
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        label= int(os.path.split(image_path)[1].split(".")[0])
        # Detect the face in the image
        faceCascade = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(cv2.resize(image[y: y + h, x: x + w], (64, 64), interpolation=cv2.INTER_AREA))
            labels.append(label)
    # return the images list and labels list
    return images, labels
    
if (config.trainfacerecog):
    images, labels = get_images_and_labels(path)
    recognizer.train(images, np.array(labels))
    recognizer.save("./models/facerecog.xml")
else:
    recognizer.load("./models/facerecog.xml")

def detect_face(frame):
        faceCascade = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(64, 64)
            )
        return gray, faces

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
 
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
 
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
 
    # return the eye aspect ratio
    return ear

def extract_face_emo(gray, detected_face):
        (x, y, w, h) = detected_face
        extracted_face = gray[y:y+h, 
                          x:x+w]
        extracted_face = zoom(extracted_face, (48. / extracted_face.shape[0], 
                                               48. / extracted_face.shape[1]))
        extracted_face = extracted_face.astype(np.float32)
        extracted_face /= 255.0
        return extracted_face.reshape(1,48,48,1)

def extract_face_gender(gray, detected_face):
        (x, y, w, h) = detected_face
        extracted_face = gray[y:y+h, 
                          x:x+w]
        extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0], 
                                               64. / extracted_face.shape[1]))
        extracted_face = extracted_face.astype(np.float32)
        extracted_face /= 255.0
        return extracted_face.reshape(1,64,64,1)
        
def extract_face_age(frame, detected_face):
    img_h, img_w, _ = frame.shape
    if detected_face is None:
        detected_face = [0, 0, img_w, img_h]
    (x, y, w, h) = detected_face
    x_a = x
    y_a = y 
    x_b = x + w
    y_b = y + h 
    cropped = frame[y_a: y_b, x_a: x_b]
    resized_img = cv2.resize(cropped, (64, 64), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    face_imgs = np.empty((1, 64, 64, 3))
    face_imgs[0]=resized_img
    return face_imgs

def weight_img(img):
    img = 3 * np.mean(img) / 170 * 100
    return int(img)

def change_brightness(a):
    if (sys.platform=='win32' or sys.platform=='win64'):
        wmi.WMI(namespace='wmi').WmiMonitorBrightnessMethods()[0].WmiSetBrightness(min(100,int(a)), 0)
    elif sys.platform.startswith('linux'):
        os.system("xbacklight -set "+min(100,int(a)))
    return 0
    
    
video_capture = cv2.VideoCapture(0)
print(video_capture.grab())
prev_vals = [100]
prev_mean = None
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
while video_capture.isOpened():

    # Capture frame-by-frame
#    sleep(0.8)
    ret, frame = video_capture.read()
    if not frame is None:
        if(ret):
            emo_vect=[0]*7
            emo_count=0
            img_val = weight_img(frame)
            # if deviation is high, reset prev_vals
            if abs(img_val - np.mean(prev_vals)) > 20:
                prev_vals = [img_val]
            else:
                prev_vals.append(img_val)
            # smooth by last values to prevent oscillation of brightness
            if len(prev_vals) < (config.averaging - 1):
                prev_vals = prev_vals[:(config.averaging - 1)]
    
            val_mean = int(np.mean(prev_vals))
            # only steps by 10 to prevent changing the
            # brightness in every evaluation
            if val_mean > config.sensible_threshold:
                val_mean = int(val_mean // config.steps * config.steps)
            # do not disable screen light
            elif val_mean < config.low_threshold:
                val_mean = config.low_light
            print("setting brightness to", val_mean)
            if prev_mean != val_mean:
                change_brightness(val_mean)
                prev_mean = val_mean
            
            # detect faces
            gray, faces = detect_face(frame)
            face_index = 0
            rects = detector(gray, 0)
            for rect in rects:
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
    
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    # average the eye aspect ratio together for both eyes
                    ear = (leftEAR + rightEAR) / 2.0
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                    if ear < config.EYE_AR_THRESHOLD:
                        COUNTER += 1
                        
                    else:
                        COUNTER = 0
    
            # predict output
            for face in faces:
                (x, y, w, h) = face
                if w > 40:
                    if COUNTER >= config.EYE_AR_CONSEC_FRAMES:
                        cv2.putText(frame, "drowsy", (x+int((3/4)*w),y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1, cv2.LINE_AA)
                    # extract features
                    extracted_face_emotion = extract_face_emo(gray, face) 
        
        
                    emotion_prediction = model.predict(extracted_face_emotion)
                    emotion_probability = np.max(emotion_prediction)
                    prediction_result = np.argmax(emotion_prediction)
                    frame[face_index * 48: (face_index + 1) * 48, -49:-1, :] = extracted_face_emotion*255.0 #cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)

                    for i in range(len(emo_vect)): emo_vect[i] += emotion_prediction.tolist()[0][i]
                    emo_count+=1

                    # annotate main image with a label
                    if prediction_result == 0:
                        cv2.putText(frame, "angry",(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_probability * np.asarray((255, 0, 0)), 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_probability * np.asarray((255, 0, 0)), 1)
                    elif prediction_result == 1:
                        cv2.putText(frame, "disgust",(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_probability *np.asarray((0, 255, 0)), 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_probability *np.asarray((0, 255, 0)), 1)
                    elif prediction_result == 2:
                        cv2.putText(frame, "fear",(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_probability * np.asarray((125, 125, 0)), 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (x, y), (x+w, y+h),emotion_probability * np.asarray((125, 125, 0)), 1)
                    elif prediction_result == 3:
                        cv2.putText(frame, "happy",(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_probability * np.asarray((255, 255, 0)), 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_probability * np.asarray((255, 255, 0)), 1)
                    elif prediction_result == 4:
                        cv2.putText(frame, "sad",(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_probability * np.asarray((0, 0, 255)), 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_probability * np.asarray((0, 0, 255)), 1)
                    elif prediction_result == 5:
                        cv2.putText(frame, "surprise",(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_probability * np.asarray((0, 255, 255)), 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_probability * np.asarray((0, 255, 255)), 1)
                    else :
                        cv2.putText(frame, "neutral",(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_probability *np.asarray((255, 255, 255)), 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_probability *np.asarray((255, 255, 255)), 1)
                    
                    extracted_face_gender = extract_face_gender(gray, face) 
        
        
                    gender_prediction = model_gender.predict(extracted_face_gender)
                    gender_probability = np.max(gender_prediction)
                    prediction_result = np.argmax(gender_prediction)
                    print(prediction_result)
                    if prediction_result == 0:
                        cv2.putText(frame, "female",(x,y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1, cv2.LINE_AA)
                    elif prediction_result == 1:
                        cv2.putText(frame, "male",(x,y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1, cv2.LINE_AA)
                    if (config.age_flag):
                        extracted_face_age = extract_face_age(frame, face) 
                        ageresults = modelage.predict(extracted_face_age)
                        ages = np.arange(0, 101).reshape(101, 1)
                        predicted_ages = ageresults[1].dot(ages).flatten()
                        cv2.putText(frame, str(int(predicted_ages[0])),(x+int((3/4)*w),y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1, cv2.LINE_AA)
                    
                    extracted_face_recog = cv2.resize(gray[y: y + h, x: x + w], (64, 64), interpolation=cv2.INTER_AREA)
                    name_predicted, conf = recognizer.predict(extracted_face_recog)
                    if (conf>=config.face_recog_threshold):
                        cv2.putText(frame, str(name_predicted),(int(x+w/2),int(y+h+h/10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "unknown", (int(x+w/2),int(y+h+h/10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1, cv2.LINE_AA)
            cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        # Display the resulting frame
            cv2.imshow('Video', frame)
    if (config.graph_flag):
        j+=1
        if (emo_count!=0):
            emo_vect_norm=  [x / emo_count for x in emo_vect]
        else:
            emo_vect_norm=emo_vect
        print(emo_vect_prec)
        print(emo_vect_norm)
        if (j>=2):
            for i in range(7):
                plt.plot([j-1,j], [emo_vect_prec[i],emo_vect_norm[i]], '.'+color_vect[i]+'-', label=label[i])
        plt.xlim(xmin=max(j-100,0), xmax=j+100)
        if (j==2):
            plt.legend()
        plt.show()
        emo_vect_prec=emo_vect_norm
        plt.pause(0.02)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break