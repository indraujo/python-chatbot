import nltk
from nltk import stem
#from nltk.stem.lancaster import LancasterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
stemmer = StemmerFactory()

import numpy as np
import tflearn
import tensorflow
from tensorflow.python.framework import ops
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    print("try coooy")
    f = open("data.pickle","rb")
    words,labels,training,output = pickle.load(f)
except:
    print("except cooooy")
    words  = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        #print(intent)
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            print(wrds)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words   = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words   = sorted(list(set(words)))
    print(words)

    labels  = sorted(labels)

    training = []
    output   = []

    out_empty = [0 for _ in range(len(labels))]

    for x,doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training    = np.array(training)
    output      = np.array(output)

    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training,output),f)

print("Ini data Training :",training)
print("Ini data Output :", output)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation="softmax")
net = tflearn.regression(net)
print("ini net",net)
model = tflearn.DNN(net)
print(model)
try:
    print("Open model.tflearn dulu")
    model.load("model.tflearn")
except:
    print("Buat model.tflearn dulu")
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    print("Selesai Buat model.tflearn dulu")
    
def bag_of_words(s,words):
    print("Jumlah Kata :",len(words))
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def chat():
    print("Start talking with the bot!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp,words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]
        print(results)
        print(labels)
        print(tag)

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag']==tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("I don't understand dude, please try again")
chat()