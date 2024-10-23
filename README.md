#### Objectif du projet

Le but de ce projet est de produire une application à partir d'un modèle de machine learning qui prédit le salaire des data scientists en fonction de leurs caractéristiques. Il s'agit d'un problème de régression.

#### Les étapes du projet

1. **Collecte et récupération des données**

Selon [CHALONS, G., 2022](https://kaizen-solutions.net/kaizen-insights/articles-et-conseils-de-nos-experts/cycle-de-vie-projet-machine-learning-8-etapes/), 
"Les projets de Machine Learning sont des processus longs, du fait qu'ils nécessitent notamment de collecter une grande quantité de données afin d'apporter une réponse robuste, fiable et stable." Cependant, nous n'avons pas eu à effectuer cette tâche, car nous utilisons des données propres disponibles sur Kaggle. Vous pouvez accéder au dataset et à sa description via ce [**lien**](https://www.kaggle.com/datasets/abhinavshaw09/data-science-job-salaries-2024).

2. **Exploration et Visualisation des données**

Cette étape, ainsi que la suivante, nous a permis de mieux comprendre l'environnement étudié, d'éviter certains biais et de fournir des données de qualité aux algorithmes de machine learning choisis. Elle m'a permis de bien visualiser la distribution des salaires, d'identifier les doublons et les asymétries de la distribution des salaires. Elle m'a aussi montré qu'il n'y avait pas de valeurs manquantes dans le jeu de données.

3. **Préparation des données**

Comme dans tout projet informatique, la qualité des données entrantes impacte fortement les résultats : "Garbage In, Garbage Out". Nous avons supprimé les doublons et corrigé les asymétries identifiées ([voir le notebook ](https://github.com/Senfidel/Machine-learning/blob/main/Script.ipynb)) durant l'étape précédente. Nous avons également enrichi les données en convertissant les codes des pays en noms complets grâce à la librairie pycountry et recodé plusieurs autres variables pour améliorer leur lisibilité par l'utilisateur.

4. **Choix et implémentation des modèles**

Nous avons testé plusieurs algorithmes de régression : **k-NN Regression, Polynomial Regression, ElasticNet, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor** et **SVR**, en comparant leurs capacités de généralisation. Notre choix de ces algotithmes est en partie motivé par une volonté de mise en pratique et de compréhension d'un grand nombre d'algorithmes de regression.

5. **Entraînement, évaluation et validation des modèles**

Afin d'éviter d'introduire des biais statistiques menant généralement à un sous-apprentissage ou à un surapprentissage des modèles, nous avons divisé le jeu de données en deux parties. 
La première partie (80 % des données) a servi à entraîner les modèles, tandis que la deuxième partie (20 % des données) a été utilisée pour les tester. Nous avons effectué cette séparation de manière aléatoire et reproductible. Pour optimiser les hyperparamètres, nous avons choisi des grilles de recherche plus ou moins simples en fonction du temps d'apprentissage des algorithmes utilisés, afin d'optimiser le temps d'apprentissage. En effet, les algorithmes à entraîner étaient relativement nombreux et cette phase du projet est très gourmande en ressources de calcul (CPU, GPU) et en temps. Nous avons dû interrompre l'optimisation des hyperparamètres lors du premier essai car l'ordinateur avait tourné pendant plus de 24 heures sans épuiser les combinaisons possibles des grilles que nous lui avions soumises. Nous avons ensuite exploré differents hyperparamettres sur Orange, ce qui nous a permi de trouver plus facilement les grilles optimales. Après avoir trouvé les grilles optimales, nous sommes retournés sur python pour selectionner les combinaisons des hyperparamètres qui minimisaient le plus le MSE (Mean Squared Error) de chaque algorithme. Après l'optimisation des hyperparamètres, nous avons comparé la capacité des algorithmes à prédire les données test et sélectionné celui qui avait le MSE le plus faible. C'est ce modèle (le modèle SVR) que nous avons déployé en production.  Pour obtenir le salaire réel dans ce modèle final, nous avons transformé les prédictions qui étaient en logarithmes en exponentielles. La MSE mesure la moyenne des écarts au carré entre les valeurs prédites et les valeurs réelles dans un ensemble de données. 
Il est vrai que ces valeurs brutes sont moins aisée à interpreter que celle de la RMSE qui est est mesurée dans les mêmes unités que la variable cible (elle est la raçine carrée du MSE) mais c'est la première qui était plus utilisée dans les tutos que nous avons regardé.

6. **Déploiement du modèle**

Nous avons créé une application Streamlit dans laquelle nous avons encapsulé les différents modèles entraînés et validés avec la méthode K-Fold. Pour faciliter le déploiement, ces modèles ont été introduits dans des pipelines et enregistrés. Dans la première page de l'application, nous présentons le contexte du projet (description sommaire et quelques visualisations). Dans la deuxième page, nous présentons des prédictions des algorithmes sur un échantillon des données test. La dernière page est réservée à l'utilisation du modèle SVR pour la prédiction des salaires. Chat GPT mous a été d’une aide précieuse dans ce projet, notamment pour la création du pipeline des modèles.
