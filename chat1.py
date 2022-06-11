import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn

#libreria para hablar
import pyttsx3 


from tensorflow.python.framework import ops
import random
import json
import pickle



#Abrimo en archivo aceptando caracteres utf-8
with open('rep.json', encoding='utf-8') as archi:
    data = json.load(archi)


listaPalabras = []
tags = []
ladoX1 = []
nachosX = []



for paraRJSON in data["contenedorP"]:

    for respuesta in paraRJSON["preguntas"]:


        palabraSep = nltk.word_tokenize(respuesta)


        listaPalabras.extend(palabraSep)
        ladoX1.append(palabraSep)

        nachosX.append(paraRJSON["tag"])

    if paraRJSON["tag"] not in tags:
        print("ESTA PALABRA NO ESTÁ : ",paraRJSON)
        tags.append(paraRJSON["tag"])


#Limpiar 
listaPalabras = [stemmer.stem(w.lower()) for w in listaPalabras if w != "?"]

listaPalabras = sorted(list(set(listaPalabras)))

tags = sorted(tags)

entrenos = []
salidaPalabras = []

vaciaSal = [0 for _ in range(len(tags))]

for x, doc in enumerate(ladoX1):
    bolsaP = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in listaPalabras:
        if w in wrds:
            bolsaP.append(1)
        else:
            bolsaP.append(0)

    salidaPalabras_row = vaciaSal[:]
    salidaPalabras_row[tags.index(nachosX[x])] = 1

    entrenos.append(bolsaP)
    salidaPalabras.append(salidaPalabras_row)


entrenos = numpy.array(entrenos)
salidaPalabras = numpy.array(salidaPalabras)


#reiniciar el tensorflow
ops.reset_default_graph()

#realizar una red neuronal
#Ingreso de los datos a evaluar, sin ninguna forma
rdNeu = tflearn.input_data(shape=[None, len(entrenos[0])])

rdNeu = tflearn.fully_connected(rdNeu, 88)
rdNeu = tflearn.fully_connected(rdNeu, 88)
rdNeu = tflearn.fully_connected(rdNeu, 88)
rdNeu = tflearn.fully_connected(rdNeu, 88)

rdNeu = tflearn.fully_connected(rdNeu, len(salidaPalabras[0]), activation="softmax")

rdNeu = tflearn.regression(rdNeu)

model = tflearn.DNN(rdNeu)

model.load("model.tflearn")

model.fit(entrenos, salidaPalabras, n_epoch=8000, batch_size=100, show_metric=True)
model.save("model.tflearn")


#algoritmo de la bolsa de palabras
def paraPalab(s, listaPalabras):
    bolsaP = [0 for _ in range(len(listaPalabras))]

    tokPalabra = nltk.word_tokenize(s)
    tokPalabra = [stemmer.stem(word.lower()) for word in tokPalabra]

    for se in tokPalabra:
        for i, w in enumerate(listaPalabras):
            if w == se:

                #a la iteracion si es similar la palabra/token declarar ese indice como 1 o sea verdadero
                bolsaP[i] = 1
            
    return numpy.array(bolsaP)



def chatBot():

    engine = pyttsx3.init()
    


    while True:

        

        inp = input("Yo>")
        

        results = model.predict([paraPalab(inp, listaPalabras)])
        results_index = numpy.argmax(results)
        tag = tags[results_index]

        for tg in data["contenedorP"]:
            if tg['tag'] == tag:
                busqResp = tg['respuestas']

        print("Categoria : ", tag)

        #If categoria es la de despedida, entonces, que se despida y termine el flujo
        if tag == "despedidas":
            
            valorDesp = random.choice(busqResp)
            

            print("Asistente UMG>",valorDesp)
            
            engine.say(valorDesp)
            engine.runAndWait()
            
            break


        #Reproducir texto a voz
        

        #Mostrar una respuesta aletaria
        valorDeRan = random.choice(busqResp);

        print("Asistente UMG>",valorDeRan)

        engine.say(valorDeRan)
        engine.runAndWait()



chatBot()