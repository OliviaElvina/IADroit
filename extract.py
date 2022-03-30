from bs4 import BeautifulSoup as bs
import pandas as pd
import os
from collections import defaultdict



#permet d'obtenir tous les noms de fichier d'un répertoire
def getFileNames(repertoire):
    namesFic = os.listdir(repertoire)
    return namesFic


#permet l'ouverture d'un fichier et sa conversion en objet BeautifulSoup
def openFic(nomFic, repertoire):
    with open("./"+repertoire+"/"+nomFic, 'r', encoding="utf-8") as ficHtml:
    #with open(".\\"+repertoire+"\\"+nomFic, 'r', encoding="utf-8") as ficHtml:
        soup = bs(ficHtml, "lxml")
        soup.encode("utf-8")
        body = soup.find("body")
    return soup, body



#permet d'obtenir tous les labels de décision des fichiers
def getLabel(repertoire):
    labels = []
    namesFic = getFileNames(repertoire)
    for name in namesFic:
        pageSoup, bodySoup = openFic(name, repertoire)
        labels.append(pageSoup.find("meta", attrs={'name' : 'output'})['content'])
    return labels



def dataToCSV(repertoire):

    """
        Fonction permettant de mettre tous les textes dans un csv comportant 6 colonnes :
        le numéro de l'affaire, tout le texte, que les faits, que les motifs, que les jugements, et le label
    """

    namesFic = getFileNames(repertoire)
    dataDict = defaultdict(list)

    with open("data.csv", "w", encoding="UTF-8") as ficData:
        entetes = ["all", "faits", "motifs", "jugement", "faits_motifs", "motifs_jugement", "label"]
        ficData.write("\t".join(entetes) + "\n")

        for i, name in enumerate(namesFic):
            tabData = []
            pageSoup, bodySoup = openFic(name, repertoire)
            label = pageSoup.find("meta", attrs={'name' : 'output'})['content']

            texteAll = [t.strip() for t in bodySoup.text.split("\n") if t.strip()]
            texteAll = " || ".join(texteAll)
            tabData.append(texteAll)

            tabContent = ["faits", "motifs", "jugement"]
            for content in tabContent:
                texteContent = bodySoup.find_all('div', class_=content)
                texte = []

                for el in texteContent:
                    texte.append(el.text.strip().replace("\n", " | "))

                texte = " || ".join(texte)
                tabData.append(texte)

            tabData.append(tabData[1]+"||"+tabData[2])
            tabData.append(tabData[2]+"||"+tabData[3])


            if label != "" and label != "S":
                tabData.append(label)
                ficData.write("\t".join(tabData) + "\n")


def dataToDic(repertoire):

    """
        Fonction permettant de mettre tous les textes dans un csv comportant 6 colonnes :
        le numéro de l'affaire, tout le texte, que les faits, que les motifs, que les jugements, et le label
    """

    namesFic = getFileNames(repertoire)
    dataDict = defaultdict(list)


    for i, name in enumerate(namesFic):

        dicCol = {}
        pageSoup, bodySoup = openFic(name, repertoire)
        label = pageSoup.find("meta", attrs={'name' : 'output'})['content']

        texteAll = [t.strip() for t in bodySoup.text.split("\n") if t.strip()]
        texteAll = " || ".join(texteAll)
        dicCol["all"] = texteAll

        tabContent = ["faits", "motifs", "jugement"]
        for content in tabContent:
            texteContent = bodySoup.find_all('div', class_=content)
            texte = []

            for el in texteContent:
                texte.append(el.text.strip().replace("\n", " | "))

            texte = " || ".join(texte)
            dicCol[content] = texte

        dicCol["faits_motifs"] = dicCol["faits"]+"||"+dicCol["motifs"]
        dicCol["motifs_jugement"] = dicCol["motifs"]+"||"+dicCol["jugement"]


        if label != "" and label != "S":
            dicCol["label"] = label
            dataDict[i] = dicCol


    return dataDict


# print(dataToDic("Jugements"))
dataToCSV("Jugements")
#getData("Jugements", "all")
