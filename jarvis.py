import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import os
import time
import dateutil.parser
import playsound
import speech_recognition
from gtts import gTTS
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import google.getEvents as gc


def speak(text):
    tts = gTTS(text=text, lang="en")
    filename = "voice.mp3"
    tts.save(filename)
    playsound.playsound(filename)

def get_audio():
    said = ""
    while said == "":
        recognizer = speech_recognition.Recognizer()
        with speech_recognition.Microphone() as source:
            audio = recognizer.listen(source)
        print("It's your turn")
        try:
            said = recognizer.recognize_google(audio)
            print("You: " + str(said))
        except Exception as e:
            print("Please talk to me!")

    return said


with open("model/data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("model/model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def main():
    
    tag = ""

    while tag != "greeting":
        inp = get_audio()
        results = model.predict([bag_of_words(inp, words)])[0]
        print(results)
        results_index = numpy.argmax(results)
        tag = labels[results_index]

    
    print("      /\  \       /\  \         /\  \         /\__\          ___        /\  \     ")
    print("      \:\  \     /::\  \       /::\  \       /:/  /         /\  \      /::\  \    ")
    print("  ___ /::\__\   /:/\:\  \     /:/\:\  \     /:/  /          \:\  \    /:/\ \  \   ")
    print(" /\  /:/\/__/  /::\~\:\  \   /::\~\:\  \   /:/__/  ___      /::\__\  _\:\~\ \  \  ")
    print(" \:\/:/  /    /:/\:\ \:\__\ /:/\:\ \:\__\  |:|  | /\__\  __/:/\/__/ /\ \:\ \ \__\ ")
    print("  \::/  /     \/__\:\/:/  / \/_|::\/:/  /  |:|  |/:/  / /\/:/  /    \:\ \:\ \/__/ ")
    print("   \/__/           \::/  /     |:|::/  /   |:|__/:/  /  \::/__/      \:\ \:\__\   ")
    print("                   /:/  /      |:|\/__/     \::::/__/    \:\__\       \:\/:/  /   ")
    print("                  /:/  /       |:|  |        ~~~~         \/__/        \::/  /    ")
    print("                  \/__/         \|__|                                   \/__/     ")

    service = gc.authenticate_google()

    for intent in data['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']

    speak(random.choice(responses))

    chat()

def chat():
    
    while True:
        inp = get_audio()

        results = model.predict([bag_of_words(inp, words)])[0]
        print(results)
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        
        if results[results_index] > 0.7:
            if tag == 'calendar':
                events = gc.get_events(3, service)

                if not events:
                    speak('No upcoming events found.')
                for event in events:
                    print(event['start'])
                    start = event['start'].get('dateTime', event['start'].get('date'))
                    startFormatted = dateutil.parser.parse(start).strftime('%c')
                    print(startFormatted + " " + event['summary'])
                    speak(startFormatted + " " + event['summary'])

            else:
                for intent in data['intents']:
                    if intent['tag'] == tag:
                        responses = intent['responses']

                speak(random.choice(responses))
                if tag == "goodbye":
                    # Kill process if we're saying goodbye to jarvis
                    break
        else:
            speak("What the fuck are you saying? To me?")


# Definição de variável global para autenticação de Google Calendar
service = None

# Load de arquivo de controle de controle de conversa
with open("intents.json") as file:
    data = json.load(file)

main() 