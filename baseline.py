# from extract import *

import numpy as np
import pandas as pd
from collections import defaultdict
import random

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


#prepare la dataset en la splittant en train et test et en la vectorisant
def prepareDataset(dataSet):

    #division en train/test
    train_X, test_X, train_y, test_y = train_test_split(
        dataSet[args.feature].fillna(' '),
        dataSet["label"],
        test_size=0.20,
        random_state=42,
        stratify=dataSet["label"])


    #on vectorise les donnees d'entree
    if args.vector == "tfidf":
        vectorizer = TfidfVectorizer(ngram_range=(1,args.nGram))
    else:
        vectorizer = CountVectorizer(binary=True, max_df=0.8, min_df=5, ngram_range=(1,args.nGram))

    train_vectors = vectorizer.fit_transform(train_X)
    test_vectors = vectorizer.transform(test_X)

    return train_X, test_X, train_vectors, test_vectors, train_y, test_y, vectorizer




#permet de trouver les meilleurs parametres des modeles choisis
def crossVal(dataSet, nameModels, modelsEval, train_vectors, train_y):

    models = modelsEval.copy()

    param_grid_models = [
        {'alpha': np.logspace(0,5,6)},
        {"n_neighbors" : np.arange(1,20), 'metric':['euclidean','manhattan']},
        {"kernel" : ["linear", "rbf", "sigmoid"], "C" : np.logspace(-1,1,3)}
    ]

    if args.model == "all":

        for name, model, param in zip(nameModels, modelsEval, param_grid_models):
            grid = GridSearchCV(model, param, cv = 5)
            grid.fit(train_vectors, train_y)
            models[nameModels.index(name)] = grid.best_estimator_


    elif args.model in ["Bayes", "KNN", "SVM"]:
        model = modelsEval[nameModels.index(args.model)]
        param = param_grid_models[nameModels.index(args.model)]
        grid = GridSearchCV(model, param, cv = 5)
        grid.fit(train_vectors, train_y)
        models[nameModels.index(args.model)] = grid.best_estimator_

    return models



#permet de donner les metriques d'un modele
def evaluate(pred, y):

    macroF1 = metrics.f1_score(y, pred, average = "macro")
    microF1 = metrics.f1_score(y, pred, average = "micro")
    F1weighted = metrics.f1_score(y, pred, average = "weighted")
    acc = metrics.accuracy_score(y, pred)
    F1 = metrics.f1_score(y, pred, average = None)

    return macroF1, microF1, F1weighted, acc, F1




#permet de lancer les differents modeles et de les evaluer => retourne un dataFrame panda avec les resultats
def runModel(dataSet, nameModels, models, train_vectors, train_y, test_vectors, test_y):

    dicPredictTest = defaultdict(list)
    dicPredictTrain = defaultdict(list)
    dicResults = defaultdict(list)


    if args.model == "all":

        for name, model in zip(nameModels, models):
            model.fit(train_vectors, train_y)
            dicPredictTrain[name] = model.predict(train_vectors)
            dicPredictTest[name] = model.predict(test_vectors)


        for name in nameModels:
            trainMacroF1, trainMicroF1, trainF1weighted, trainAcc, trainF1 = evaluate(dicPredictTrain[name], train_y)
            testMacroF1, testMicroF1, testF1weighted, testAcc, testF1 = evaluate(dicPredictTest[name], test_y)
            tempF1 = []
            for valeur in testF1:
                tempF1.append(round(valeur, 3))

            dicResults[name] = {
                "Macro-F1 Train": round(trainMacroF1, 3),
                "Micro-F1 Train" : round(trainMicroF1, 3),
                "F1-weighted Train" : round(trainF1weighted, 3),
                "Accuracy Train" : round(trainAcc, 3),
                "Macro-F1 Test": round(testMacroF1, 3),
                "Micro-F1 Test" : round(testMicroF1, 3),
                "F1-weighted Test" : round(testF1weighted, 3),
                "Accuracy Test" : round(testAcc,3),
                "F1 Test" : tempF1
                }

        results = pd.DataFrame(data=dicResults)
        # pd.set_option('display.max_columns', None)
        results.to_excel("C:/Users/olivi/OneDrive/Bureau/Baseline/Resultats/"+args.feature+"_"+args.vector+"_"+str(args.nGram)+".xlsx", encoding="utf-8")

    else:
        model = models[nameModels.index(args.model)]
        model.fit(train_vectors, train_y)
        predictTrain = model.predict(train_vectors)
        predictTest = model.predict(test_vectors)

        trainMacroF1, trainMicroF1, trainF1weighted, trainAcc, trainF1 = evaluate(predictTrain, train_y)
        testMacroF1, testMicroF1, testF1weighted, testAcc, testF1 = evaluate(predictTest, test_y)

        tempF1 = []
        for valeur in testF1:
            tempF1.append(round(valeur, 3))

        dicResults[args.model] = {
            "Macro-F1 Train": round(trainMacroF1, 3),
            "Micro-F1 Train" : round(trainMicroF1, 3),
            "F1-weighted Train" : round(trainF1weighted, 3),
            "Accuracy Train" : round(trainAcc, 3),
            "Macro-F1 Test": round(testMacroF1, 3),
            "Micro-F1 Test" : round(testMicroF1, 3),
            "F1-weighted Test" : round(testF1weighted, 3),
            "Accuracy Test" : round(testAcc,3),
            "F1 Test" : tempF1
            }

        results = dicResults


    return results



#Interpretation d'un classificateur
def interpret(test_X, train_y, test_y, model, nameModel, vectorizer):

    pipe = make_pipeline(vectorizer, model)
    class_names = sorted(set(train_y))
    explainer = LimeTextExplainer(class_names=class_names)
    dicInterpret = defaultdict(list)

    alea = [random.randint(0, 100) for x in range(5)]

    for idx in alea:
        exp = explainer.explain_instance(list(test_X)[idx], pipe.predict_proba, num_features=20)
        dicInterpret["Document "+str(idx)] = {
            # "Proba : " : pipe.predict_proba([list(test_X)[idx]]),
            "True class : " : list(test_y)[idx],
            "Features predictives : " : exp.as_list()
        }
        exp.save_to_file(".\Interpretation\\"+nameModel+"_"+args.feature+"_"+args.vector+"_"+str(args.nGram)+"_Document "+str(idx)+"_"+list(test_y)[idx]+"_"+".html")

    interpretResults = pd.DataFrame(data=dicInterpret)

    return interpretResults



#fonction principale appelee dans la partie main
def main():


    #listes des noms des modeles et des modeles associes
    nameModels = [
        "Bayes",
        "KNN",
        "SVM"
    ]

    modelsEval = [
        naive_bayes.MultinomialNB(),
        KNeighborsClassifier(),
        SVC()
    ]


    #on charge la dataset
    dataSet = pd.read_csv('data.csv', sep="\t")

    #Repartition de la dataset
    # print("Repartition des labels : \n")
    # print(dataSet["label"].value_counts(normalize=True))
    # print("\n")


    train_X, test_X, train_vectors, test_vectors, train_y, test_y, vectorizer = prepareDataset(dataSet)

    #On choisit les meilleurs parametres pour chaque mode
    models = crossVal(dataSet, nameModels, modelsEval, train_vectors, train_y)

    #Ajout du MLP sur lequel on ne peut pas faire de GridSearchCV
    nameModels.extend(["MLP", "Dummy", "RandomForest"])
    models.extend([
        MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1, max_iter=500),
        DummyClassifier(strategy="most_frequent"),
        RandomForestClassifier(max_depth=7, n_estimators=200)
        ])


    results = runModel(dataSet, nameModels, models, train_vectors, train_y, test_vectors, test_y)
    print("Metriques des differents modeles : \n")
    # pd.set_option('display.max_columns', None)
    print(results)
    print(models)

    if args.sortie == "interpret":
        if args.modelInterpret == "SVM":
            print("Interpretation Lime impossible avec SVM")
        else:
            interpretResults = interpret(test_X, train_y, test_y, models[nameModels.index(args.modelInterpret)], args.modelInterpret, vectorizer)
            print(interpretResults)



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices = ["all", "Bayes", "KNN", "SVM", "MLP", "Dummy", "RandomForest"], help="Choix du modele")
    parser.add_argument("feature", choices = ["all", "faits", "motifs", "jugement", "faits_motifs", "motifs_jugement"], help = "Choix de la partie du texte sur laquelle appliquer le modele")
    parser.add_argument("vector", choices = ["tfidf", "CountVect"], help = "Choix de la methode de vectorisation")
    parser.add_argument("nGram", choices = np.arange(1,7), type=int, help = "Choix du ngram_range")
    parser.add_argument("-s", "--sortie", choices = ["interpret"], help = "Choix de sortir l'interpretation avec Lime")
    parser.add_argument("-mI", "--modelInterpret", choices = ["Bayes", "KNN", "SVM", "MLP", "Dummy", "RandomForest"])
    args = parser.parse_args()

    main()
