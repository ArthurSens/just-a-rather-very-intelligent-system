import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import os
import argparse
import time
import dateutil.parser
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import google.getEvents as gc


import chatbot as chatbot
import google.getEvents as gc
import glob

def main():
    
    tag = ""

    while tag != "greeting":
        inp = chatbot.get_audio()
        results = glob.model.predict([chatbot.bag_of_words(inp, glob.words)])[0]
        results_index = numpy.argmax(results)
        tag = glob.labels[results_index]

    
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

    for intent in glob.data['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']

    chatbot.speak(random.choice(responses))

    chat()

def chat():

    while True:            
        print("Contexto antigo: " + str(glob.context))
        print("Contexto novo: " + str(glob.new_context))

        if glob.context != glob.new_context:
            glob.context = glob.new_context            
            glob.words, glob.labels, glob.training, glob.output, glob.model, glob.data = load_model(glob.context, glob.lang)
    
        inp = chatbot.get_audio()
        chatbot.chat(inp)

        if glob.new_context == "goodbye":
            break





parser = argparse.ArgumentParser(
        description="Inspired by MCU J.A.R.V.I.S, a simple chatbot with voice"
    )
parser.add_argument('--credentials_path', metavar='', help='Path for the google calendar credential\'s file', default='credentials.json')
parser.add_argument('--language', metavar='', help='Initial language used by Jarvis', default='en')
credentials_path = parser.parse_args().credentials_path
glob.lang = parser.parse_args().language


def load_model(m, language):

    with open("model/" + language + "/" + m + "/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    model.load("model/" + language + "/" + m + "/model.tflearn")

    with open("intents/" + language + "/" + m + ".json") as file:
        data = json.load(file) 

    return words, labels, training, output, model, data


# Definição de variável global para autenticação de Google Calendar
glob.service = gc.authenticate_google(credentials_path)

# Load de arquivo de controle de controle de conversa
glob.words, glob.labels, glob.training, glob.output, glob.model, glob.data = load_model('intents', glob.lang)

main() 