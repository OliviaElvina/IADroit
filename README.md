# IADroit

Paramètres de baseline.py :
  - "model" => Choix du modèle à lancer parmi ["all", "Bayes", "KNN", "SVM", "MLP", "Dummy", "RandomForest"]
  - "feature" => Choix de la partie du texte sur laquelle appliquer le modèle parmi ["all", "faits", "motifs", "jugement", "faits_motifs", "motifs_jugement"]
  - "vector" => Choix de la méthode de vectorisation parmi ["tfidf", "CountVect"]
  - "nGram" => Choix du rang des ngrammes entre 1 et 7
  - "-s" ou "--sortie" => Choix de sortir l'interprétation avec Lime via "interpret"
  - "-mI" ou "--modelInterpret" => Choix du modèle à interpréter parmi ["Bayes", "KNN", "SVM", "MLP", "Dummy", "RandomForest"]
