# emotiondetection

Real-time emotion (anger, joy, surprise, disgust, happiness...), age, gender, drowsiness and face recognition, by using the webcam camera of the computer's user.
All the models were trained from scratch and use various deep neural networks (CNN) models. Models are not included, but you can retrain them using the code included in recon.py and gender_model.py.
Face detection and extraction is done through a cascade classifier. Face recogntion uses an LBPH classifier. Drowsiness works by extracting key facial features using a pretrained model, and tracking the evolution of the facial features' position. The other classifiers use various CNN sturctures.
