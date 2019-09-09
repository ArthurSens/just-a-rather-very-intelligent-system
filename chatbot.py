import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import playsound
import speech_recognition
from gtts import gTTS
import dateutil.parser
import numpy
import random
import google.getEvents as gc
import glob

def speak(text):
    tts = gTTS(text=text, lang=glob.lang)
    filename = "voice.mp3"
    tts.save(filename)
    playsound.playsound(filename)

def get_audio():
    said = ""
    while said == "":
        recognizer = speech_recognition.Recognizer()
        with speech_recognition.Microphone() as source:
            audio = recognizer.listen(source, )
        print("It's your turn")
        try:
            said = recognizer.recognize_google(audio, language=glob.lang)
            print("You: " + str(said))
        except Exception as e:
            print("Please talk to me!")

    return said

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def intentsChat(inp):
    results = glob.model.predict([bag_of_words(inp, glob.words)])[0]

    results_index = numpy.argmax(results)
    tag = glob.labels[results_index]
    
    if results[results_index] > 0.7:
        if tag == 'calendar':
            events = gc.get_events(3, glob.service)

            if not events:
                speak('No upcoming events found.')
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                startFormatted = dateutil.parser.parse(start).strftime('%c')
                print(startFormatted + " " + event['summary'])
                speak(startFormatted + " " + event['summary'])
            
            glob.new_context = "intents"
        else:
            for intent in glob.data['intents']:
                if intent['tag'] == tag:
                    responses = intent['responses']
                    glob.new_context = intent['context_set']

            speak(random.choice(responses))
            

    else:
        speak("What the fuck are you saying? To me?")
        glob.new_context = 'intents'

def changeLanguageChat(inp):
    results = glob.model.predict([bag_of_words(inp, glob.words)])[0]
    
    results_index = numpy.argmax(results)
    tag = glob.labels[results_index]
    
    if results[results_index] > 0.7:
        for intent in glob.data['intents']:
            if intent['tag'] == tag:
                responses = intent['responses']
                glob.new_context = intent['context_set']
                glob.lang = tag

        speak(random.choice(responses))

    else:
        speak("I don't know that language, so we will keep talking in the same we are talking now, okay?")
        glob.new_context = "intents"









def chat(inp):    
    if glob.context == "intents":
        return intentsChat(inp)
    elif glob.context == "changeLanguage":
        return changeLanguageChat(inp)


