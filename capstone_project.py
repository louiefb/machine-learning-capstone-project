import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("profiles.csv")

#####################
#####EXPLORATION#####
#####################

#reading columns
print(df.columns)

#exploring some of the columns
print(df.drinks.unique())
print(df.drugs.unique())
print(df.education.unique())
for i in range(15):
    print(df.ethnicity.unique()[i]) #this helped made entries to this column human-readable
print(df.income.unique())
print(len(df.last_online.unique()))
print(df.last_online.head(5)) #given that there's 30k+ entries, I looked at only the first five to see what the data looked like
print(df.orientation.unique())
print(df.smokes.unique())
print(len(df.speaks.unique()))
print(df.speaks.head(10)) #given that there's 7k+ entries, I looked at only the first ten to see what the data looked like
print(df.status.unique())
for i in df.religion.unique():
    print(i)
for i in df.location.unique():
    print(i)

#feature engineering: drinks, drugs, smokes
df["drinks"] = df["drinks"].map({"not at all":0, "rarely":1, "socially":2, "often":3, "very often":4, "desperately":5})
df["drugs"] = df["drugs"].map({"never":0, "sometimes":1, "often":2})
df["smokes"] = df["smokes"].map({"no":0, "trying to quit":1, "sometimes":2, "when drinking":3, "yes":4})
print(len(df) - len((df.drinks + df.drugs + df.smokes).dropna())) #when we take out all the entries with NaNs from drinks, drugs, smokes, we remove 17,451 entries (or we have ~70% left)

#custom feature: generation
generation = []
for i in range(len(df)):
    if 18 <= df["age"].iloc[i] <= 32:
        generation.append(0)
    elif 32 < df["age"].iloc[i] <= 47:
        generation.append(1)
    else:
        generation.append(2)
df["generation"] = generation

#custom feature: languages, based on speaks
df["languages"] = df["speaks"].apply(lambda row: 0 if pd.isnull(row) else len(row.split(", "))) #converted NaNs to 0 since there's only 50 of them (considered outliers)

#custom feature: total length of characters of all essays for each user
import math
df["essay0_length"] = df["essay0"].apply(lambda row: len(row) if type(row) == str else 0)
df["essay1_length"] = df["essay1"].apply(lambda row: len(row) if type(row) == str else 0)
df["essay2_length"] = df["essay2"].apply(lambda row: len(row) if type(row) == str else 0)
df["essay3_length"] = df["essay3"].apply(lambda row: len(row) if type(row) == str else 0)
df["essay4_length"] = df["essay4"].apply(lambda row: len(row) if type(row) == str else 0)
df["essay5_length"] = df["essay5"].apply(lambda row: len(row) if type(row) == str else 0)
df["essay6_length"] = df["essay6"].apply(lambda row: len(row) if type(row) == str else 0)
df["essay7_length"] = df["essay7"].apply(lambda row: len(row) if type(row) == str else 0)
df["essay8_length"] = df["essay8"].apply(lambda row: len(row) if type(row) == str else 0)
df["essay9_length"] = df["essay9"].apply(lambda row: len(row) if type(row) == str else 0)
df["total_text_length"] = df.apply(lambda row: row.essay0_length + row.essay1_length + row.essay2_length + row.essay3_length + row.essay4_length + row.essay5_length + row.essay6_length + row.essay7_length + row.essay8_length + row.essay9_length, axis=1)
#i used the code below to look at the "total_text_length" outlier
print(df["total_text_length"].max())
#for i in df[df["total_text_length"] == df["total_text_length"].max()]["essay0"].tolist():
    #print(i)
#now, instead of getting rid the outlier, this entry was trimmed to match the next longest essay
df[["total_text_length"]].iloc[27528] = 59113

#customer feature: religious seriousness
religious_seriousness = []
for i in range(len(df)):
    if type(df["religion"][i]) == str:
        if "laughing" in df["religion"][i]:
            religious_seriousness.append(0)
        elif "not too serious" in df["religion"][i]:
            religious_seriousness.append(1)
        elif "somewhat serious" in df["religion"][i]:
            religious_seriousness.append(2)
        elif "very serious" in df["religion"][i]:
            religious_seriousness.append(4)
        else:
            religious_seriousness.append(3) #those without the modifying words are considered simplye "serious"
    else:
        religious_seriousness.append(1) #Assumption: NaNs are treated as "not too serious" cuz it makes sense that those who didn't even answer this question may not take religion too seriously
df["religious_seriousness"] = religious_seriousness

#custom feature: indication of how serious profile is about finding love ("love" counter in essay9)
df["love"] = df["essay9"].apply(lambda row: 0 if pd.isnull(row) else row.casefold().count("love"))

#unused custom feature: state, based on location
df["state"] = df.apply(lambda row: row["location"].split(", ")[-1], axis=1)
print(df.state.unique())
df.state.value_counts() #we'll find out that most people are from CA
df[df["state"] == "california"]["location"].value_counts() #and a majority is from SF

#exploring income
plt.hist(df.income)
plt.show() #since a lot of people didnt share income, this may not be useful if we drop all those that reported "-1"

#######################
#####NORMALIZATION#####
#######################

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#cleaning up the data by deleting all the NaNs so each feature would have an even index
df = df[["age", "drinks", "smokes", "drugs", "total_text_length", "love", "generation", "languages", "religious_seriousness"]].dropna()
df["drinks"] = scaler.fit_transform(np.reshape(df[["drinks"]], (-1,1)))
df["smokes"] = scaler.fit_transform(np.reshape(df[["smokes"]], (-1,1)))
df["drugs"] = scaler.fit_transform(np.reshape(df[["drugs"]], (-1,1)))
df["total_text_length"] = scaler.fit_transform(np.reshape(df[["total_text_length"]], (-1,1)))
df["love"] = scaler.fit_transform(np.reshape(df[["love"]], (-1,1)))
df["generation"] = scaler.fit_transform(np.reshape(df[["generation"]], (-1,1)))
df["languages"] = scaler.fit_transform(np.reshape(df[["languages"]], (-1,1)))
df["religious_seriousness"] = scaler.fit_transform(np.reshape(df[["religious_seriousness"]], (-1,1)))

####################################
#####MULTIPLE LINEAR REGRESSION#####
####################################

input("\nMultiple Linear Regression\nHit any key to continue...\n")
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

def linear_regression(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x,y,train_size=0.8,test_size=0.2,random_state = 42) #splitting the data

    line_fitter = LinearRegression()
    line_fitter.fit(x_train, y_train)

    predicted_y = line_fitter.predict(x_train)

    score = line_fitter.score(x_test, y_test)
    
    return score

#getitng all combinations of all the features
from itertools import combinations
features = ["generation", "total_text_length", "languages", "drinks", "drugs", "smokes", "religious_seriousness", "love"]
combo = []
for i in range(1,len(features)):
    combo.append(list(combinations(features, i)))

#saving the scores for every combination of features
y = df[["age"]]
scores = []
for i in range(len(combo)):
    for j in range(len(combo[i])):
        x = df[[*combo[i][j]]]
        scores.append([linear_regression(x, y), combo[i][j]])

#printing our scores
for i in scores:
    print(i)

#getting the highest accuracy for our model
MLR_best_score = max(scores, key=lambda score: score[0])

print("\nOur best score for Multiple Linear Regression is", MLR_best_score[0], ".\nThe combination of features used to achieve this is", MLR_best_score[1], ".")
input("Hit any key to continue...\n")

########################
#####KNN REGRESSION#####
########################

input("\nK-Nearest Neighbors Weighted Regression\nHit any key to continue...\n")

from sklearn.neighbors import KNeighborsRegressor

def KNN_regression(x, y, neighbors):
    x_train, x_test, y_train, y_test = train_test_split(
        x,y,train_size=0.8,test_size=0.2,random_state = 42)

    regressor = KNeighborsRegressor(n_neighbors = neighbors, weights = "distance")
    regressor.fit(x_train, y_train)

    score = regressor.score(x_train, y_train)
    
    return score

#saving the scores for every combination of features
y = df[["age"]]
scores = []
for i in range(len(combo)):
    for j in range(len(combo[i])):
        x = df[[*combo[i][j]]]
        scores.append([KNN_regression(x, y, 5), combo[i][j]])

#printing our scores
for i in scores:
    print(i)

#getting the highest accuracy for our model
KNN_best_score = max(scores, key=lambda score: score[0])

print("\nOur best score for KNN Regression is", KNN_best_score[0], ".\nThe combination of features used to achieve this is", KNN_best_score[1], ".")
input("Now we'll find out what could be our ideal number of neighbors. Hit any key to continue...\n")

#finding out ideal neighbors
N_score = []
for neighbors in range(1,15):
        x = df[[*max(scores, key=lambda score: score[0])[1]]]
        y = df[["age"]]
        N_score.append([KNN_regression(x, y, neighbors), neighbors])

print("\nOur best score for KNN Regression is", max(N_score, key=lambda score: score[0]),
      ".\nWe obtained this by making use of these many neighbors:", max(N_score, key=lambda score: score[1]), ".")
input("Hit any key to continue...\n")

########################
#####KNN CLASSIFIER#####
########################

input("\nK-Nearest Neighbors Classifier\nHit any key to continue...\n")

from sklearn.neighbors import KNeighborsClassifier

def KNNClassifier(x, y, neighbors):
    x_train, x_test, y_train, y_test = train_test_split(
        x,y,train_size=0.8,test_size=0.2,random_state = 42)
    
    classifier = KNeighborsClassifier(n_neighbors = neighbors)
    classifier.fit(x_train, y_train)

    score = classifier.score(x_test, y_test)

    return score

#updating the features for our classifier
features = ["total_text_length", "languages", "drinks", "drugs", "smokes", "religious_seriousness", "love"]
combo = []
for i in range(1,len(features)):
    combo.append(list(combinations(features, i)))

#saving the scores for every combination of features
scores = []
y = scaler.inverse_transform(np.reshape(df[["generation"]], (-1,1)))
y = np.ravel(y)
for i in range(len(combo)):
    for j in range(len(combo[i])):
        x = df[[*combo[i][j]]]
        scores.append([KNNClassifier(x, y, neighbors=5), combo[i][j]])

#printing our scores
for i in scores:
    print(i)

#getting the highest accuracy for our model
KNNC_best_score = max(scores, key=lambda score: score[0])

print("\nOur best score for KNN Classifier is", KNNC_best_score[0], ".\nThe combination of features used to achieve this is", KNNC_best_score[1], ".")
input("Now we'll find out what could be our ideal number of neighbors. Hit any key to continue...\n")

#finding out ideal neighbors
N_score = []
x = df[[*max(scores, key=lambda score: score[0])[1]]]
for neighbors in range(1,15):
        N_score.append([KNNClassifier(x, y, neighbors), neighbors])

print("\nOur best score for KNN Classifier is", max(N_score, key=lambda score: score[0]),
      ".\nWe obtained this by making use of these many neighbors:", max(N_score, key=lambda score: score[1]), ".")
input("Hit any key to continue...\n")

########################
#####SVC CLASSIFIER#####
########################

input("\nSupport Vector Machine Classifier\nHit any key to continue...\n")

from sklearn.svm import SVC

def SVCClassifier(x, y, C, gamma):
    x_train, x_test, y_train, y_test = train_test_split(
        x,y,train_size=0.8,test_size=0.2,random_state = 42)

    classifier = SVC(kernel="rbf", C=C, gamma=gamma)
    classifier.fit(x_train, y_train)

    score = classifier.score(x_test, y_test)

    return score

#finding the best C and gamma using the feature combination that worked best for our KNN Classifier
x = df[[*max(scores, key=lambda score: score[0])[1]]] #also equal to df[["religious_seriousness", "love"]]
y = scaler.inverse_transform(np.reshape(df[["generation"]], (-1,1)))
y = np.ravel(y)
c = [0.1, 1.0, 10, 100, 1000]
gamma = [1, 10, 100]

#saving the scores for every combination of features
scores = []
for C_ in c:
    for gamma_ in gamma:
        scores.append([SVCClassifier(x, y, C_, gamma_), C_, gamma_])

#printing our scores
for i in scores:
    print(i)

#getting the highest accuracy for our model
SVC_best_score = max(scores, key=lambda score: score[0])

print("\nOur best score for SVC Classifier is", SVC_best_score[0],
      ".\nWe obtained this by making use of C value of:", SVC_best_score[1], "and gamma value of:",  SVC_best_score[2], ".")
input("Hit any key to continue...\n")

#plotting heat map of diferent C and gamma values and their corresponding accuracies
scores_accuracy = [i[0] for i in scores]

scores_ = []
for i in range(int(len(scores_accuracy)/3)):
    scores_.append(scores_accuracy[i*3:(i*3)+3])

heat_df = pd.DataFrame(scores_, index=c, columns=gamma)
sns.heatmap(heat_df, annot=True, fmt="f", linewidths=0.5, cbar_kws={"label": "ACCURACY"})

plt.xlabel("GAMMA")
plt.ylabel("C VALUE")
plt.show()
