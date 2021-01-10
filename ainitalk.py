import nltk
from nltk import stem
from nltk.stem.lancaster import LancasterStemmer

import numpy as np
import tflearn
import tensorflow
from tensorflow.python.framework import ops
import random
import json
import pickle

class askaini:
    def __init__(self):
        self.stemmer = LancasterStemmer()

        with open("intents.json") as file:
            self.data = json.load(file)

        try:
            print("try coooy")
            f = open("data.pickle","rb")
            self.words,self.labels,self.training,self.output = pickle.load(f)
        except:
            print("except cooooy")
            self.words  = []
            self.labels = []
            self.docs_x = []
            self.docs_y = []

            for intent in self.data["intents"]:
                #print(intent)
                for pattern in intent["patterns"]:
                    self.wrds = nltk.word_tokenize(pattern)
                    print(self.wrds)
                    self.words.extend(self.wrds)
                    self.docs_x.append(self.wrds)
                    self.docs_y.append(intent["tag"])

                if intent["tag"] not in self.labels:
                    self.labels.append(intent["tag"])

            words   = [self.stemmer.stem(w.lower()) for w in self.words if w != "?"]
            words   = sorted(list(set(words)))
            print(words)

            labels  = sorted(self.labels)

            training = []
            output   = []

            out_empty = [0 for _ in range(len(labels))]

            for x,doc in enumerate(self.docs_x):
                bag = []

                wrds = [self.stemmer.stem(w) for w in doc]

                for w in words:
                    if w in wrds:
                        bag.append(1)
                    else:
                        bag.append(0)

                output_row = out_empty[:]
                output_row[labels.index(self.docs_y[x])] = 1

                training.append(bag)
                output.append(output_row)

            self.training    = np.array(training)
            output      = np.array(output)

            with open("data.pickle","wb") as f:
                pickle.dump((words,labels,training,output),f)

        print("Ini data Training :",self.training)
        print("Ini data Output :", self.output)

        ops.reset_default_graph()

        net = tflearn.input_data(shape=[None,len(self.training[0])])
        net = tflearn.fully_connected(net,8)
        net = tflearn.fully_connected(net,8)
        net = tflearn.fully_connected(net,len(self.output[0]), activation="softmax")
        net = tflearn.regression(net)
        print("ini net",net)
        self.model = tflearn.DNN(net)
        print(self.model)
        try:
            print("Open self.model.tflearn dulu")
            self.model.load("self.model.tflearn")
        except:
            print("Buat self.model.tflearn dulu")
            self.model.fit(self.training, self.output, n_epoch=1000, batch_size=8, show_metric=True)
            self.model.save("self.model.tflearn")
            print("Selesai Buat self.model.tflearn dulu")
        
    def bag_of_words(self,s,words):
        print("Jumlah Kata :",len(words))
        bag = [0 for _ in range(len(words))]
        s_words = nltk.word_tokenize(s)
        s_words = [self.stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i,w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return np.array(bag)

    def askaini(self,message):
        results = self.model.predict([self.bag_of_words(message,self.words)])[0]
        results_index = np.argmax(results)
        tag = self.labels[results_index]
        print(results)
        print(self.labels)
        print(tag)

        if results[results_index] > 0.7:
            for tg in self.data["intents"]:
                if tg['tag']==tag:
                    self.responses = tg['responses']
            answer = random.choice(self.responses)
            return answer
        else:
            answer = "I don't understand dude, please try again"
            return answer