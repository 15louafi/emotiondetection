sleep = 10  # seconds, time between photo detection

averaging = 10  # last n photos to be considered


averaging_ignore = 20  # jump to measured brigthness if
# difference to average brighness is higher than this value
# for example:
# average = 30, measured = 80
# -> 80 - 30 > 20 (averaging_ignore)
# -> set brightness to 80

steps = 10  # int, steps in which the light is changed,
# for example: brightness 30, 40, 50, .. for steps = 10
sensible_threshold = 20  # under this value steps will be ignored
# for example:
# ..,5,6,7,8,9,10,20,30,40,.. for steps=10 and sensible_threshold = 10

low_threshold = 5  # all light values under threshold will be set to low_light
low_light = 3 # lowest light value


EYE_AR_THRESHOLD = 0.3
EYE_AR_CONSEC_FRAMES = 25
 
COUNTER = 0

face_recog_threshold=0.5

facerecogpath = './facerecog'

trainfacerecog= 1
age_flag = True

graph_flag = False