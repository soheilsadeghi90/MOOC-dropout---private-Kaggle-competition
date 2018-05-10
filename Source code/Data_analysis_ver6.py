# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing as pp
import numpy as np
from sklearn.ensemble import BaggingClassifier as Bag
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.feature_selection import SelectKBest
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import xgboost as xgb
from xgboost import plot_importance
import statsmodels.api as sm

## FILE DESCRIPTION: This version is the last edition that gives the best score

######################################### Import Data ######################################### 

data_1 = pd.read_csv("C:\\Users\\ses516\\Desktop\\Data Mining\\Final Project\\activity_log.csv")
data_2 = pd.read_csv("C:\\Users\\ses516\\Desktop\\Data Mining\\Final Project\\enrollment_list.csv")
data_3 = pd.read_csv("C:\\Users\\ses516\\Desktop\\Data Mining\\Final Project\\train_label.csv")

######################################### Feature Extraction ######################################### 

data_2['course_id'] = pd.Categorical(data_2['course_id']).codes
data_2['user_id'] = pd.Categorical(data_2['user_id']).codes

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

data_1["time"] = pd.to_datetime(data_1["time"])
data_1["month"] = data_1["time"].dt.month
data_1["hour"] = data_1["time"].dt.hour
data_1["day"] = data_1["time"].dt.day
data_1["year"] = data_1["time"].dt.year

data_1['time_diff'] = data_1.groupby(['enrollment_id','day'])['time'].diff()
data_1['time_diff'] = data_1.time_diff.dt.total_seconds()
entire_time = pd.DataFrame({'entire_time':data_1.groupby('enrollment_id')['time_diff'].sum()}).reset_index()
entire_time = entire_time.fillna(0)

event_count = pd.DataFrame({'event_count':data_1.groupby(['enrollment_id','day']).size()}).reset_index()
day_count = pd.DataFrame({'day_count':data_1.groupby('enrollment_id').day.nunique()}).reset_index()
time_diff = pd.DataFrame({'time_diff':data_1.groupby(['enrollment_id','day'])['hour'].max()-data_1.groupby(['enrollment_id','day'])['hour'].min()}).reset_index()
std = pd.DataFrame({'std':data_1.groupby('enrollment_id')['time_diff'].std()}).reset_index()
std = std.fillna(20)
time_density = pd.DataFrame({'time_density':entire_time['entire_time']/event_count['event_count']}).reset_index()

gb = time_diff.groupby('enrollment_id')['time_diff']

def get_lin_reg_coef(series):
    x=sm.add_constant(range(series.count()))
    result = sm.OLS(series, x).fit().params[1]
    return result/series.mean()

output_df = gb.apply(get_lin_reg_coef)
output_df = pd.DataFrame({'slope': output_df}).reset_index()
output_df = output_df.fillna(0)

month = data_1.groupby('enrollment_id', as_index=False)['month'].mean()
hour = data_1.groupby('enrollment_id', as_index=False)['hour'].median()
year = data_1.groupby('enrollment_id', as_index=False)['year'].mean()

# feature = count.assign(problem = pd.Series(problem['count'])   , page_close = pd.Series(page_close['count'])
#                      , course_id = pd.Series(data_2['course_id'])    , hour = pd.Series(hour['hour']) 
#                      , day_count = pd.Series(day_count['day_count']) , entire_time = pd.Series(entire_time['entire_time'])
#                      , std = pd.Series(std['std'])
#                      , slope = pd.Series(output_df['slope']))

# countperhr = pd.DataFrame({'cph':pd.Series(count['count'])/pd.Series(hour['hour'])}).replace([np.inf, -np.inf], 0).reset_index()

feature = count.assign(slope = pd.Series(output_df['slope'])   , std = pd.Series(std['std'])
                     , navigate = pd.Series(navigate['count']) , access = pd.Series(access['count'])
                     , problem = pd.Series(problem['count'])   , discussion = pd.Series(discussion['count'])
                     , wiki = pd.Series(wiki['count'])         , page_close = pd.Series(page_close['count'])
                     , video = pd.Series(video['count'])       , course_id = pd.Series(data_2['course_id'])
                     , month = pd.Series(month['month'])       , day_count = pd.Series(day_count['day_count']) 
                     , time_density = pd.Series(time_density['time_density']) , hour = pd.Series(hour['hour']))
                            
# available_data = feature[feature['enrollment_id'] <= 72325]
# submission_data = feature[feature['enrollment_id'] > 72325]

######################################### Scaling ######################################### 

scaler = pp.StandardScaler()
Y = np.asarray(data_3['dropout_prob'], dtype=int)
# poly = pp.PolynomialFeatures(degree=2)
feat_final = feature.drop('enrollment_id', axis=1)
# X_poly = poly.fit_transform(feature)
i = feat_final.index
c = feat_final.columns
# selection = PCA(n_components=10)
# selection = SelectKBest(k=15)
# combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
# X_temp = selection.fit(X_poly[:72325,:], Y).transform(X_poly)
# X_new = scaler.fit_transform(X_temp)
X_new = scaler.fit_transform(feat_final)
Y_temp = np.zeros(len(X_new))
# X_new = np.asarray(feature)
X = X_new[:72325,:]
X_sub = X_new[72325:,:]

######################################### Meta-Features ######################################### 

KNC_log = KNeighborsClassifier(n_neighbors=3)
Y_new = np.zeros((len(X_new),1))
KNC_log.fit(np.log(10+X_new),Y_new)

A = list(KNC_log.kneighbors(np.log(10+X_new),9))[0].tolist()
dist = pd.DataFrame(A).reset_index()
dist2 = np.array(dist[[0,1]].sum(axis=1))
dist4 = np.array(dist[[0,1,2,3]].sum(axis=1))
dist6 = np.array(dist[[0,1,2,3,4,5]].sum(axis=1))

#from sklearn.decomposition import PCA
#X_embedded = PCA(n_components=5).fit_transform(X_new)
#KNC_B = KNeighborsClassifier(n_neighbors=3)
#KNC_B.fit(X_embedded,Y_new)
#B = list(KNC_B.kneighbors(X_embedded,7))[0].tolist()
#distB = pd.DataFrame(A).reset_index()
#distB2 = np.array(dist[[0,1]].sum(axis=1))
#distB4 = np.array(dist[[0,1,2,3]].sum(axis=1))
#distB6 = np.array(dist[[0,1,2,3,4,5]].sum(axis=1))

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

kmeans = KMeans(n_clusters=10, random_state=0).fit(X_new)
clusters_kmean = kmeans.labels_                    
DB = DBSCAN(eps=0.3, min_samples=50).fit(X_new)
clusters_DB = DB.labels_


META_feat = np.column_stack((X_new,clusters_kmean,clusters_DB,dist2,dist4,dist6)) 
X_META = META_feat[:72325,:]

######################################### XGBoost ######################################### 

XGB = xgb.XGBClassifier(
 learning_rate =0.1,
 n_estimators=133,
 max_depth=6,
 min_child_weight=9,
 gamma=0.0,
 subsample=1.0,
 colsample_bytree=0.6,
 objective= 'binary:logistic',
 reg_alpha = 10,
 nthread=4,
 scale_pos_weight=1,
 seed=27)

train = X_META[:50400,:]
y_train = Y[:50400]
valid =  X_META[50400:,:]
y_valid = Y[50400:]
eval_set  = [(train,y_train), (valid,y_valid)]
XGB.fit(train, y_train, eval_set=eval_set,
    eval_metric="logloss", early_stopping_rounds=50)

prob_XGB = np.zeros((len(META_feat),1))

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=50)
skf.get_n_splits(X_META, Y)
for train_index, test_index in skf.split(X_META, Y):
    train, valid = X_META[train_index], X_META[test_index]
    y_train, y_valid = Y[train_index], Y[test_index]
    XGB.fit(train, y_train)
    temp = XGB.predict_proba(META_feat)[:,1]
    prob_XGB = np.column_stack((prob_XGB, temp))
    print(log_loss(y_train,temp[train_index]))
    print(log_loss(y_valid,temp[test_index]))


plot_importance(XGB,importance_type='weight')
plt.show()

#XGB.fit(X_META, Y)
#prob_XGB = XGB.predict_proba(META_feat)[:,1]
#result = prob_XGB[72325:]

######################################### KNN [K=3] ######################################### 

KNC3 = KNeighborsClassifier(n_neighbors=3)
bagg_KNC3 = Bag(base_estimator = KNC3,n_estimators=200,bootstrap='true',max_samples=500, oob_score=False)
prob_KNN = np.zeros((len(META_feat),1))
for train_index, test_index in skf.split(X_META, Y):
    train, valid = X_META[train_index], X_META[test_index]
    y_train, y_valid = Y[train_index], Y[test_index]
    bagg_KNC3.fit(train, y_train)
    temp = bagg_KNC3.predict_proba(META_feat)[:,1]
    prob_KNN = np.column_stack((prob_KNN, temp))
    print(log_loss(y_train,temp[train_index]))
    print(log_loss(y_valid,temp[test_index]))

######################################### Classifier Averaging ######################################### 

a = 0.95
prob_agg = prob_XGB*a+prob_KNN*(1-a)
print(log_loss(y_train,prob_agg[:50400]))
print(log_loss(y_valid,prob_agg[50400:72325]))

#result = result_entire[72325:]
#print(log_loss(Y[:50400],prob_ADA_XT[:50400,:]))
#print(log_loss(Y[50400:],prob_ADA_XT[50400:72325,:]))

######################################### Generate Submission Files ######################################### 

prob_final = np.mean(prob_XGB[:,1:],axis = 1)
result = prob_final[72325:]
result = pd.DataFrame({ 'enrollment_id': range(72326,120543),
                            'dropout_prob': result })
result.to_csv("C:\\Users\\ses516\\Desktop\\Data Mining\\Final Project\\subs\\submission26.csv", index=False)

######################################### END ######################################### 
