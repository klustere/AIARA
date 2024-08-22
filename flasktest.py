from flask import Flask, request, jsonify
import os
import subprocess
import speech_recognition as sr
from flask_cors import CORS
from sys import argv,exit
import cv2
from deepface import DeepFace
import numpy as np
import whisper
import google.generativeai as genai

from os import listdir

import logging

# Set the logging level to suppress the detailed output
logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.WARNING)
import warnings
# Suppress the warnings
warnings.filterwarnings("ignore")
from os import environ
from logging import getLogger,ERROR

# Silence the TensorFlow deprecation warning
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
getLogger('tensorflow').setLevel(ERROR)


GOOGLE_API_KEY="" #Add your own key
os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
modelGem = genai.GenerativeModel('gemini-1.5-pro-latest')
modelWhis = whisper.load_model("base")
# modelYOLO = YOLO("./final/runs/classify/train6/weights/best.pt")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

app = Flask(__name__)
CORS(app)

# Set the directory where the uploaded files will be stored
upload_dir = 'web_capture/'

# Initialize the processed data variable
processedData = ''

@app.route('/uploadVideo', methods=['POST'])
def upload_video():
    # Get the uploaded file from the request
    video_file = request.files['videoFile']

    # Save the file to the upload directory
    global video_path
    video_path = os.path.join(upload_dir, video_file.filename)
    video_file.save(video_path)
    
    global voice_process
    voice_process  = voiceProcess(video_path)
    # Process the video file

    # Remove the uploaded file
    # os.remove(video_path)

    # Return the processed text as a JSON response
    return jsonify({'processedText': voice_process})
    



def gemini(s):
    name, emotion, prompt = s.split('|')
    # print(s)
    if emotion == "neutral" or "NONE":
        promptmod='''Your role: A friendly social chatbot. Your response format: Emotion of your answer-either happy/sad/neutral/laughing(laughing only when telling jokes) 
+ greeting/congratulating/comforting/consoling + your generated response for query. Do not use any emojis. 
First word of your reply should just contain the hypothetical emotional state of your response followed by the response. The response should be short and sweet, unless explicitly mentioned in the query.for translation based prompts, type other languages as english text.
My name:'''+name+""". my query:"""+prompt
    else:
        promptmod='''Your role: A friendly social chatbot. 
Your response format: emotion of your response 
+ greeting/congratulating/comforting/consoling + your generated response for query. Do not use any emojis.
First word of your reply should just contain the hypothetical emotional state of your response followed by the response. The response should be short and sweet, unless explicitly mentioned in the query.
My name:'''+name+""". my emotion:sad. my query:"""+prompt    
    response=modelGem.generate_content(promptmod)
    s=response.text
    index = s.find("-")
    s = s[:index] + "|" + s[index+1:]
    # Replace "-" with "|"
    
    return s

def voiceProcess(video_path):
    result = modelWhis.transcribe(video_path)
    return result['text']

def process_video(video_path):
    # names = listdir("./final/data/test")
    # print(names)
    class_counts = {}
    cap = cv2.VideoCapture(video_path)
    minutes = 0
    seconds = 0
    fps = 0.5
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break  # Break the loop when the video ends
        t_msec = 1000*fps*(minutes*60 + seconds)
        cap.set(cv2.CAP_PROP_POS_MSEC, t_msec)
        # Convert frame to grayscale
        seconds += 1
        if seconds > 60:
            seconds = 0
            minutes += 1

        face_list=['tharun','sudeesh','pradeep','shyam','adi','neel','muthu']

        for i in face_list:
            if (DeepFace.verify(frame,'training_faces/'+i+'.jpeg')['distance']<0.5):
                class_counts[i]=1
        # except:
        #     pass
        # predictions = modelYOLO(frame,verbose=False)
        # predicted_class = names[predictions[0].probs.top1]  # Assuming predictions contain class probabilities
        # class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Initialize a list to store the detected emotions
        emotions = []

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = frame[y:y + h, x:x + w]

            # Perform emotion analysis on the face ROI
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                emotions.append(emotion)
            except:
                # Ignore any errors that occur during emotion analysis
                pass
    #
    # Print the dominant emotion and the list of detected emotions
    max_count_class = "NONE"
    dominant_emotion = "NONE"
    try:
        max_count_class = max(class_counts, key=class_counts.get)
        print(max_count_class)
        dominant_emotion = max(set(emotions), key=emotions.count)
        # print(f"Detected Emotion: {dominant_emotion} {emotions}")
    except:
        pass
    # Release the capture
    cap.release()    
    return max_count_class,dominant_emotion


@app.route('/getProcessedText', methods=['GET'])
def get_processed_text():
    processedData = finalout(video_path, voice_process)
    return processedData

def finalout(video_path, voice_process):
    processed_output = process_video(video_path)
    print(voice_process)
    kk =f"{processed_output[0]}|{processed_output[1]}|{voice_process}"
    final_ = gemini(kk)
    kk = f"{voice_process}|{final_}"
    print(kk)
    return kk

if __name__ == '__main__':
    app.run(host='localhost', port=3000, debug=True)