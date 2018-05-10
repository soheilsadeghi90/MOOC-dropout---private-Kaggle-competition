from matplotlib import pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing as pp
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
import random
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

## FILE DESCRIPTION: This version includes voting classifier

data_1 = pd.read_csv("C:\\Users\\ses516\\Desktop\\Data Mining\\Final Project\\activity_log.csv")
data_2 = pd.read_csv("C:\\Users\\ses516\\Desktop\\Data Mining\\Final Project\\enrollment_list.csv")
data_3 = pd.read_csv("C:\\Users\\ses516\\Desktop\\Data Mining\\Final Project\\train_label.csv")
data_4 = pd.read_csv("C:\\Users\\ses516\\Desktop\\Data Mining\\Final Project\\test_label.csv")


data_2['course_id'] = pd.Categorical(data_2['course_id']).codes

count = pd.DataFrame({'count': data_1.groupby("enrollment_id").size()}).reset_index()
rng = data_1['enrollment_id'].unique()

nav_temp = data_1[(data_1['event']=='navigate')]
nav_temp2 = pd.DataFrame({'count': nav_temp.groupby("enrollment_id").size()}).reset_index()
navigate = nav_temp2.set_index('enrollment_id').reindex(range(1,120543)).fillna(0).reset_index()

ac_temp = data_1[(data_1['event']=='access')]
ac_temp2 = pd.DataFrame({'count': ac_temp.groupby("enrollment_id").size()}).reset_index()
access = ac_temp2.set_index('enrollment_id').reindex(range(1,120543)).fillna(0).reset_index()

prob_temp = data_1[(data_1['event']=='problem')]
prob_temp2 = pd.DataFrame({'count': prob_temp.groupby("enrollment_id").size()}).reset_index()
problem = prob_temp2.set_index('enrollment_id').reindex(range(1,120543)).fillna(0).reset_index()

disc_temp = data_1[(data_1['event']=='discussion')]
disc_temp2 = pd.DataFrame({'count': disc_temp.groupby("enrollment_id").size()}).reset_index()
discussion = disc_temp2.set_index('enrollment_id').reindex(range(1,120543)).fillna(0).reset_index()

wiki_temp = data_1[(data_1['event']=='wiki')]
wiki_temp2 = pd.DataFrame({'count': wiki_temp.groupby("enrollment_id").size()}).reset_index()
wiki = wiki_temp2.set_index('enrollment_id').reindex(range(1,120543)).fillna(0).reset_index()

pgc_temp = data_1[(data_1['event']=='page_close')]
pgc_temp2 = pd.DataFrame({'count': pgc_temp.groupby("enrollment_id").size()}).reset_index()
page_close = pgc_temp2.set_index('enrollment_id').reindex(range(1,120543)).fillna(0).reset_index()

vid_temp = data_1[(data_1['event']=='video')]
vid_temp2 = pd.DataFrame({'count': vid_temp.groupby("enrollment_id").size()}).reset_index()
video = vid_temp2.set_index('enrollment_id').reindex(range(1,120543)).fillna(0).reset_index()


feature = count.assign(navigate = pd.Series(navigate['count']) , access = pd.Series(access['count'])
                     , problem = pd.Series(problem['count'])   , discussion = pd.Series(discussion['count']) 
                     , wiki = pd.Series(wiki['count'])         , page_close = pd.Series(page_close['count'])
                     , video = pd.Series(video['count'])       , course_id = pd.Series(data_2['course_id']))
                     
available_data = feature[feature['enrollment_id'] <= 72325]
submission_data = feature[feature['enrollment_id'] > 72325]

scaler = pp.StandardScaler()
X = scaler.fit_transform(feature)[:,1:]
Y = data_3['dropout_prob']
poly = pp.PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest

selection = SelectKBest(k=10)
X_new = selection.fit(X_poly[:72325,:], Y).transform(X_poly)

X = X_new[:72325,:]
X_sub = X_new[72325:,:]

### Classifiers

KNC3 = KNeighborsClassifier(n_neighbors=5)
SVM = svm.SVC(probability=True)
GNB = gnb()
DT = DecisionTreeClassifier(criterion='gini', random_state=1)
GBC = GradientBoostingClassifier(n_estimators=8000, loss = 'deviance')
MPL = MLPClassifier(alpha=1e-5, activation='relu', random_state=1,hidden_layer_sizes=(100,100,100,100))
RFC = RandomForestClassifier(n_estimators=300, criterion='gini', random_state=1,oob_score = True)
SGD = SGDClassifier()

agg = VotingClassifier(estimators=[('SVM', SVM), ('GNB', GNB), ('RFC', RFC)]
, voting='soft',weights = [1,1,1])

boost_classifier = AdaBoostClassifier(SGD, n_estimators=5)

bagg_KNC3 = BaggingClassifier(base_estimator = KNC3,n_estimators=2000,bootstrap='true',max_samples=500, oob_score=False)

aggregate_classifier = VotingClassifier(estimators=[('KNC3', bagg_KNC3), ('MPL', MPL), ('boost_classifier', boost_classifier)]
, voting='soft',weights = [1,1,2])

aggregate_classifier.fit(X,Y)

test = random.sample(range(1, 72325), 1000)

print(accuracy_score(Y[test],aggregate_classifier.predict(X[test,:])))

print(log_loss(Y[test],aggregate_classifier.predict_proba(X[test,:]))) 

result = aggregate_classifier.predict_proba(X_sub)[:,1]
np.savetxt("C:\\Users\\ses516\\Desktop\\Data Mining\\Final Project\\submission.csv", result, delimiter=",")

