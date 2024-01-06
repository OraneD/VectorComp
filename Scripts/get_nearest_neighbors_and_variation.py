from os import listdir
from gensim.models import KeyedVectors
from numpy import array
from sklearn.neighbors import NearestNeighbors

wordsToTest=("peinture","composition","musique",
             "cellule","évolution",
             "antiquité","renaissance","révolution",
             "poésie","théâtre",
             "algèbre","géométrie","calcul"
             "éthique","logique",
             "électromagnétisme","mécanique","thermodynamique",
             "phonétique","syntaxe",
             "marché","investissement","inflation",
             "démocratie","dictature")
k=25

modelPaths=[f"../Vectors/{model}"for model in listdir("../Vectors")]
kNearestNeighbors={}

for modelPath in modelPaths:
    #Loading vectors
    vectors=KeyedVectors.load_word2vec_format(modelPath,binary=False)
    #Preparing results dictionary
    kNearestNeighbors[modelPath]={}
    #COnverting vectors to matrix as numpy array
    vectorsList=[]
    wordsList=[]
    for key in vectors.vocab.keys():
        vectorsList.append(vectors[key])
        wordsList.append(key)
    vectorsMatrix=array(vectorsList)
    #Fitting NearestNeighbors model from sklearn
    nn=NearestNeighbors(n_neighbors=k,metric="cosine")
    nn.fit(vectorsMatrix)
    #Getting values for word
    for word in wordsToTest:
        vectorForWord=vectors[word]
        distances,indexes=nn.kneighbors([vectorForWord])
        kNearestNeighbors[modelPath][word]=zip([wordsList[indexes[0][i]]for i in range(1,k+1)],distances)

