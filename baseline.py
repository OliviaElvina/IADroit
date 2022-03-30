from extract import *

import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import naive_bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.pipeline import make_pipeline

from lime.lime_text import LimeTextExplainer


#prépare la dataset en la splittant en train et test et en la vectorisant
def prepareDataset(dataSet):

    #division en train/test
    train_X, test_X, train_y, test_y = train_test_split(
        dataSet[args.feature].fillna(' '),
        dataSet["label"],
        test_size=0.20,
        random_state=42,
        stratify=dataSet["label"])


    #on vectorise les données d'entrée
    if args.vector == "tfidf":
        vectorizer = TfidfVectorizer(ngram_range=(1,args.nGram))
    else:
        vectorizer = CountVectorizer(binary=True, max_df=0.8, min_df=5, ngram_range=(1,args.nGram))

    train_vectors = vectorizer.fit_transform(train_X)
    test_vectors = vectorizer.transform(test_X)

    return train_X, test_X, train_vectors, test_vectors, train_y, test_y, vectorizer




#permet de trouver les meilleurs paramètres des modèles choisis
def crossVal(dataSet, nameModels, modelsEval, train_vectors, train_y):

    models = []

    param_grid_models = [
        {'alpha': np.logspace(0,5,6)},
        {"n_neighbors" : np.arange(1,20), 'metric':['euclidean','manhattan']},
        {"kernel" : ["linear", "rbf", "sigmoid"], "C" : np.logspace(-1,1,3)}
    ]

    for name, model, param in zip(nameModels, modelsEval, param_grid_models):
        grid = GridSearchCV(model, param, cv = 5)
        grid.fit(train_vectors, train_y)
        models.append(grid.best_estimator_)

    return models



#permet de donner les métriques d'un modèle
def evaluate(pred, y):

    macroF1 = metrics.f1_score(y, pred, average = "macro")
    microF1 = metrics.f1_score(y, pred, average = "micro")
    F1weighted = metrics.f1_score(y, pred, average = "weighted")
    acc = metrics.accuracy_score(y, pred)

    return macroF1, microF1, F1weighted, acc




#permet de lancer les différents modèles et de les évaluer => retourne un dataFrame panda avec les résultats
def runModel(dataSet, nameModels, models, train_vectors, train_y, test_vectors, test_y):

    dicPredictTest = defaultdict(list)
    dicPredictTrain = defaultdict(list)
    dicResults = defaultdict(list)


    for name, model in zip(nameModels, models):
        model.fit(train_vectors, train_y)
        dicPredictTrain[name] = model.predict(train_vectors)
        dicPredictTest[name] = model.predict(test_vectors)


    for name in nameModels:
        trainMacroF1, trainMicroF1, trainF1weighted, trainAcc = evaluate(dicPredictTrain[name], train_y)
        testMacroF1, testMicroF1, testF1weighted, testAcc = evaluate(dicPredictTest[name], test_y)
        dicResults[name] = {
            "Macro-F1 Train": trainMacroF1,
            "Micro-F1 Train" : trainMicroF1,
            "F1-weighted Train" : trainF1weighted,
            "Accuracy Train" : trainAcc,
            "Macro-F1 Test": testMacroF1,
            "Micro-F1 Test" : testMicroF1,
            "F1-weighted Test" : testF1weighted,
            "Accuracy Test" : testAcc
            }


    if args.model == "all":
        results = pd.DataFrame(data=dicResults)
        # results.to_excel("resultats.xlsx", encoding="utf-8")
        # results.to_csv("resultats.csv")
    else:
        results = dicResults[args.model]


    return results



#Interprétation d'un classificateur
def interpret(test_X, train_y, test_y, model, nameModel, vectorizer):

    pipe = make_pipeline(vectorizer, model)
    class_names = sorted(set(train_y))
    explainer = LimeTextExplainer(class_names=class_names)
    dicInterpret = defaultdict(list)

    for idx in range(10):
        exp = explainer.explain_instance(list(test_X)[idx], pipe.predict_proba, num_features=20)
        dicInterpret["Document "+str(idx)] = {
            "Proba : " : pipe.predict_proba([list(test_X)[idx]]),
            "True class : " : list(test_y)[idx],
            "Features prédictives : " : exp.as_list()
        }
        exp.save_to_file(".\Interpretation\\"+nameModel+"_Document "+str(idx)+".html")

    interpretResults = pd.DataFrame(data=dicInterpret)

    return interpretResults



#fonction principale appelée dans la partie main
def main():


    #listes des noms des modèles et des modèles associés
    nameModels = [
        "Bayes",
        # "KNN",
        # "SVM"
    ]

    modelsEval = [
        naive_bayes.MultinomialNB(),
        # KNeighborsClassifier(),
        # SVC()
    ]


    #on charge la dataset
    dataSet = pd.read_csv('data.csv', sep="\t")

    #Répartition de la dataset
    print("Répartition des labels : \n")
    print(dataSet["label"].value_counts(normalize=True))
    print("\n")


    train_X, test_X, train_vectors, test_vectors, train_y, test_y, vectorizer = prepareDataset(dataSet)

    #On choisit les meilleurs paramètres pour chaque modè
    models = crossVal(dataSet, nameModels, modelsEval, train_vectors, train_y)

    #Ajout du MLP sur lequel on ne peut pas faire de GridSearchCV
    nameModels.extend(["MLP", "Dummy", "RandomForest"])
    models.extend([
        MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1, max_iter=500),
        DummyClassifier(strategy="most_frequent"),
        RandomForestClassifier(max_depth=7, n_estimators=200)
        ])


    results = runModel(dataSet, nameModels, models, train_vectors, train_y, test_vectors, test_y)
    print("Métriques des différents modèles : \n")
    print(results)


    if args.sortie == "interpret":
        if args.modelInterpret == "SVM":
            print("Interprétation Lime impossible avec SVM")
        else:
            interpretResults = interpret(test_X, train_y, test_y, models[nameModels.index(args.modelInterpret)], args.modelInterpret, vectorizer)
            print(interpretResults)







if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices = ["all", "Bayes", "KNN", "SVM", "MLP", "Dummy"], help="Choix du modèle")
    parser.add_argument("feature", choices = ["all", "faits", "motifs", "jugement", "faits_motifs", "motifs_jugement"], help = "Choix de la partie du texte sur laquelle appliquer le modèle")
    parser.add_argument("vector", choices = ["tfidf", "CountVect"], help = "Choix de la méthode de vectorisation")
    parser.add_argument("nGram", choices = np.arange(1,7), type=int, help = "Choix du ngram_range")
    parser.add_argument("-s", "--sortie", choices = ["interpret"], help = "Choix de sortir l'interprétation avec Lime")
    parser.add_argument("-mI", "--modelInterpret", choices = ["Bayes", "KNN", "SVM", "MLP", "Dummy"])
    args = parser.parse_args()

    main()
