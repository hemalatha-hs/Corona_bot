import nltk
import numpy
import pickle
import random
import os
import json
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.keras.models import Sequential # tensorflow.keras is now available
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')

stemmer = LancasterStemmer()

#loading the json data
# with open("WHO.json") as file:	
json_path = os.path.join(os.getcwd(), "Corona_bot", "WHO.json")
with open(json_path) as file:			 
	data = json.load(file)
	
#print(data["intents"])
try:
	with open("data.pickle", "rb") as f:
		words, l, training, output = pickle.load(f)
except:
	
	# Extracting Data
	words = []
	l = []
	docs_x = []
	docs_y = []
	
# converting each pattern into list of words using nltk.word_tokenizer 
	for i in data["intents"]: 
		for p in i["patterns"]:
			wrds = nltk.word_tokenize(p)
			words.extend(wrds)
			docs_x.append(wrds)
			docs_y.append(i["tag"])

			if i["tag"] not in l:
				l.append(i["tag"])
	# Word Stemming		 
	words = [stemmer.stem(w.lower()) for w in words if w != "?"]		 
	words = sorted(list(set(words)))
	l = sorted(l)									 
	
	# This code will simply create a unique list of stemmed 
	# words to use in the next step of our data preprocessing
	training = []
	output = []
	out_empty = [0 for _ in range(len(l))]
	for x, doc in enumerate(docs_x):
		bag = []

		wrds = [stemmer.stem(w) for w in doc]

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)
		output_row = out_empty[:]
		output_row[l.index(docs_y[x])] = 1

		training.append(bag)
		output.append(output_row)
		
	# Finally we will convert our training data and output to numpy arrays 
	training = numpy.array(training)	 
	output = numpy.array(output)
	with open("data.pickle", "wb") as f:
		pickle.dump((words, l, training, output), f)

# Developing a Model	 
model = Sequential()
model.add(Dense(8, input_shape=(len(training[0]),), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(output[0]), activation='softmax'))  # Output layer

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Training & Saving the Model
model.fit(training, output, epochs=1000, batch_size=8, verbose=1)	 
model.save("model.h5")  # Save as Keras model

# making predictions
def bag_of_words(s, words):							 
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)


def chat():
    print("""Start talking with the bot and ask your
    queries about Corona-virus(type quit to stop)!""")
    
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Reshape the input to be 2D (add a batch dimension)
        results = model.predict(numpy.array([bag_of_words(inp, words)]))[0]  
        results_index = numpy.argmax(results)
        
        #print(results_index)
        tag = l[results_index]
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("I am sorry but I can't understand")

chat()