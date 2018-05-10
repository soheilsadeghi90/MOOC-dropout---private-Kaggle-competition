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

## FILE DESCRIPTION: This version includes the ensemble model that use predictions of individual models as input feature

data_1 = pd.read_csv("C:\\Users\\ses516\\Desktop\\Data Mining\\Final Project\\activity_log.csv")
data_2 = pd.read_csv("C:\\Users\\ses516\\Desktop\\Data Mining\\Final Project\\enrollment_list.csv")
data_3 = pd.read_csv("C:\\Users\\ses516\\Desktop\\Data Mining\\Final Project\\train_label.csv")

data_2['course_id'] = pd.Categorical(data_2['course_id']).codes
# data_2['user_id'] = pd.Categorical(data_2['user_id']).codes

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

event_count = pd.DataFrame({'event_count':data_1.groupby(['enrollment_id','day']).size()}).reset_index()
day_count = pd.DataFrame({'day_count':data_1.groupby('enrollment_id').day.nunique()}).reset_index()
time_diff = pd.DataFrame({'time_diff':data_1.groupby(['enrollment_id','day'])['hour'].max()-data_1.groupby(['enrollment_id','day'])['hour'].min()}).reset_index()
time_diff = time_diff.replace({'time_diff':{0:1}})
entire_time = pd.DataFrame({'entire_time':time_diff.groupby('enrollment_id')['time_diff'].sum()}).reset_index()
time_density = pd.DataFrame({'time_density':entire_time['entire_time']/event_count['event_count']}).reset_index()
std = pd.DataFrame({'std':time_diff.groupby('enrollment_id')['time_diff'].std()}).reset_index()
std = std.fillna(20)

gb = time_diff.groupby('enrollment_id')['time_diff']

def get_lin_reg_coef(series):
    x=sm.add_constant(range(series.count()))
    result = sm.OLS(series, x).fit().params[1]
    return result/series.mean()

output_df = gb.apply(get_lin_reg_coef)
output_df = pd.DataFrame({'slope': output_df}).reset_index()

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
                     , month = pd.Series(month['month']) 
                     , day_count = pd.Series(day_count['day_count']) , entire_time = pd.Series(entire_time['entire_time']))
                                     
# available_data = feature[feature['enrollment_id'] <= 72325]
# submission_data = feature[feature['enrollment_id'] > 72325]

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

X_train = X[:50400,:]
Y_train = Y[:50400]

### Classifiers

KNC2 = KNeighborsClassifier(n_neighbors=2)
bagg_KNC2 = Bag(base_estimator = KNC2,n_estimators=200,bootstrap='true',max_samples=500, oob_score=False)
bagg_KNC2.fit(X_train,Y_train)
prob_KNC2 = bagg_KNC2.predict(X_new)

KNC4 = KNeighborsClassifier(n_neighbors=4)
bagg_KNC4 = Bag(base_estimator = KNC4,n_estimators=200,bootstrap='true',max_samples=500, oob_score=False)
bagg_KNC4.fit(X_train,Y_train)
prob_KNC4 = bagg_KNC4.predict(X_new)

KNC8 = KNeighborsClassifier(n_neighbors=8)
bagg_KNC8 = Bag(base_estimator = KNC8,n_estimators=200,bootstrap='true',max_samples=500, oob_score=False)
bagg_KNC8.fit(X_train,Y_train)
prob_KNC8 = bagg_KNC8.predict(X_new)

KNC16 = KNeighborsClassifier(n_neighbors=16)
bagg_KNC16 = Bag(base_estimator = KNC16,n_estimators=200,bootstrap='true',max_samples=500, oob_score=False)
bagg_KNC16.fit(X_train,Y_train)
prob_KNC16 = bagg_KNC16.predict(X_new)

KNC_log = KNeighborsClassifier(n_neighbors=2)
bagg_KNC = Bag(base_estimator = KNC_log,n_estimators=200,bootstrap='true',max_samples=500, oob_score=False)
bagg_KNC.fit(np.log(10+X_train),Y_train)
prob_log_KNC = bagg_KNC.predict(np.log(10+X_new))


# print(KNC3.score(X[:50400,:],Y[:50400]))
# print(KNC3.score(X[50400:,:],Y[50400:]))
# A = list(KNC3.kneighbors(X,6))[0].tolist()
# dist = pd.DataFrame(A).reset_index()
# dist['sum2'] = dist[[0,1]].sum(axis=1)
# dist['sum4'] = dist[[0,1,2,3]].sum(axis=1)
# dist['sum6'] = dist[[0,1,2,3,4,5]].sum(axis=1)

GBC = GradientBoostingClassifier(n_estimators=100, loss = 'deviance', min_samples_split=500)
GBC.fit(X_train,Y_train)
prob_GBC = GBC.predict(X_new)

MPL = MLPClassifier(alpha=1, activation='relu')
MPL.fit(X_train,Y_train)
prob_MPL = MPL.predict_proba(X_new)
print(log_loss(Y[:50400],prob_MPL[:50400,:]))
print(log_loss(Y[50400:],prob_MPL[50400:72325,:]))

RFC = RandomForestClassifier(n_estimators=250, criterion='gini', max_depth=None, min_samples_split=0.05, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                            max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=5, random_state=None, 
                            verbose=0, warm_start=False, class_weight=None)
FRC_isotonic = CalibratedClassifierCV(RFC, cv=2, method='isotonic')
FRC_isotonic.fit(X_train,Y_train)
prob_FRC = FRC_isotonic.predict(X_new)
# print(log_loss(Y[:50400],prob_FRC[:50400,:]))
# print(log_loss(Y[50400:],prob_FRC[50400:,:]))

LReg = LR()
bagg_LR = Bag(base_estimator = LReg,n_estimators=250,bootstrap='true',max_samples=500, oob_score=False)
bagg_LR.fit(X_train,Y_train)
prob_LR = bagg_LR.predict(X_new)
# print(log_loss(Y[:50400],prob_log_LR[:50400]))
# print(log_loss(Y[50400:],prob_LR[50400:,:]))

LLReg = LR()
bagg_LR = Bag(base_estimator = LLReg,n_estimators=250,bootstrap='true',max_samples=500, oob_score=False)
bagg_LR.fit(np.log(10+X_train),Y_train)
prob_log_LR = bagg_LR.predict(np.log(10+X_new))


ADA = AdaBoostClassifier()
ADA_isotonic = CalibratedClassifierCV(ADA, cv=2, method='isotonic')
ADA_isotonic.fit(X_train,Y_train)
prob_ADA = ADA_isotonic.predict(X_new)

# XGB = xgb.XGBClassifier(max_depth=6, n_estimators=1000, learning_rate=0.09,min_child_weight=0.5,eta=0.03,objective='binary:logistic', seed=99,nthread=5, reg_lambda = 100)
# XGB = xgb.XGBClassifier(max_depth=7, n_estimators=1000, learning_rate=0.05,min_child_weight=0.5,eta=0.03,objective='binary:logistic', seed=99,nthread=5, reg_lambda = 100)
# XGB = xgb.XGBClassifier(max_depth=5, n_estimators=1000, learning_rate=0.08,min_child_weight=0.5,eta=0.03,objective='binary:logistic', seed=99,nthread=5, reg_lambda = 100)

## meta-feature
#feature = count.assign(slope = pd.Series(output_df['slope'])   , std = pd.Series(std['std'])
#                     , navigate = pd.Series(navigate['count']) , access = pd.Series(access['count'])
#                     , problem = pd.Series(problem['count'])   , discussion = pd.Series(discussion['count'])
#                    , wiki = pd.Series(wiki['count'])         , page_close = pd.Series(page_close['count'])
#                     , video = pd.Series(video['count'])       , course_id = pd.Series(data_2['course_id'])
#                     , month = pd.Series(month['month'])       , hour = pd.Series(hour['hour']) 
#                     , day_count = pd.Series(day_count['day_count']) , entire_time = pd.Series(entire_time['entire_time']))                     
                     
META_feat = np.column_stack((X_new,prob_FRC,prob_LR,prob_KNC2,prob_KNC4,prob_KNC8,prob_KNC16,prob_log_LR,prob_log_KNC,prob_GBC,prob_MPL,prob_ADA))
# META_feat = np.column_stack((X_new,nonzeros))
# X_META = X_new[:72325,:]
# X_sub = X_new[72325:,:]
X_META = META_feat[:72325,:]
X_sub = META_feat[72325:,:]

X_train = X_META[:50400,:]
Y_train = Y[:50400]

X_test = X_META[50400:,:]
Y_test = Y[50400:]

XGB = xgb.XGBClassifier(
 learning_rate =0.1,
 n_estimators=400,
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

eval_set  = [(X_train,Y_train), (X_test,Y_test)]
XGB.fit(X_train, Y_train, eval_set=eval_set,
    eval_metric="logloss", early_stopping_rounds=60)
# rounds, 21 for 6, 23 for 5, 18 for 4
prob_XGB = XGB.predict_proba(X_META)
print(log_loss(Y_train,prob_XGB[:50400,1]))
print(log_loss(Y_test,prob_XGB[50400:72325,1]))


from sklearn.ensemble import ExtraTreesClassifier

prob_XGB = np.zeros((len(X_new),1))
prob_XT = np.zeros((len(X_new),1))
EXC = ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=7, min_samples_split=5, 
                        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
                        min_impurity_split=1e-07, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, 
                        verbose=0, warm_start=False, class_weight=None)
ADA_XT = AdaBoostClassifier(base_estimator=EXC, n_estimators=20, learning_rate=1.0, random_state=None)
ADA_XT_isotonic = CalibratedClassifierCV(ADA_XT, cv=2, method='isotonic')

for train_index, test_index in KF.split(X_META):
    train, valid = X_META[train_index],  X_META[test_index]
    trainX, validX = np.log(10+X_META)[train_index],  np.log(10+X_META)[test_index]    
    y_train, y_valid = Y[train_index], Y[test_index]
    eval_set  = [(trainX,y_train), (validX,y_valid)]
    ADA_XT_isotonic.fit(train,y_train)
    temp = ADA_XT_isotonic.predict_proba(X_new)[:,1]
    prob_XT = np.column_stack((prob_XT, temp))
    XGB.fit(trainX, y_train, eval_set=eval_set,
        eval_metric="logloss", early_stopping_rounds=60)
    temp1 = XGB.predict_proba(np.log(10+X_new))[:,1]
    prob_XGB = np.column_stack((prob_XGB, temp1))
                                             
prob_XGB_final = prob_XGB[:,10:]
print(log_loss(y_train,prob_XGB_final[train_index,4]))
print(log_loss(y_valid,prob_XGB_final[test_index,4]))
result_entire = np.mean(prob_XGB_final,axis=1)

prob_XT_final = prob_XT[:,10:]
print(log_loss(y_train,prob_XT_final[train_index,4]))
print(log_loss(y_valid,prob_XT_final[test_index,4]))
result_entire1 = np.mean(prob_XT_final,axis=1)


#result = result_entire[72325:]
#print(log_loss(Y[:50400],prob_ADA_XT[:50400,:]))
#print(log_loss(Y[50400:],prob_ADA_XT[50400:72325,:]))

prob_final = np.multiply(np.power(result_entire1,1),np.power(result_entire,0))
print(log_loss(y_train,prob_final[train_index]))
print(log_loss(y_valid,prob_final[test_index]))

result = result_entire1[72325:]

XGB.fit(np.log(10+X_META),Y)
result = XGB.predict_proba(np.log(10+X_sub))[:,1]
np.savetxt("C:\\Users\\ses516\\Desktop\\Data Mining\\Final Project\\submission.csv", result, delimiter=" ")
plot_importance(XGB,importance_type='weight')
plt.show()

params = {'eta': 0.02, 'max_depth': 5, 'subsample': 0.7, 'colsample_bytree': 0.7, 'objective': 'binary:logistic', 'seed': 99, 'silent': 1, 'eval_metric':'auc', 'nthread':4}


#prob = pd.DataFrame({'prob_KNC3': prob_KNC3[:,1], 'prob_SVM': prob_SVM[:,1], 'prob_GNB': prob_GNB[:,1], 'prob_GBC': prob_GBC[:,1], 
#                     'prob_MPL': prob_MPL[:,1], 'prob_FRC': prob_FRC[:,1], 'prob_LR': prob_LR[:,1], 'prob_ADA': prob_ADA[:,1], 
#                     'prob_GPC': prob_GPC[:,1], 'prob_QDA': prob_QDA[:,1], 'prob_XGB':prob_XGB[:,1]})

prob = pd.DataFrame({'prob_KNC3': prob_KNC3[:,1], 'prob_XGB':prob_XGB[:,1]})
                     
prob['mean'] = prob.mean(axis=1)
prob['prob_guess'] = prob['mean']
# max_vals = prob[['prob_SVM','prob_KNC3','prob_GNB','prob_GBC','prob_MPL','prob_FRC','prob_LR','prob_ADA','prob_GPC','prob_QDA']].max(axis=1)
# min_vals = prob[['prob_SVM','prob_KNC3','prob_GNB','prob_GBC','prob_MPL','prob_FRC','prob_LR','prob_ADA','prob_GPC','prob_QDA']].min(axis=1)
max_vals = prob[['prob_KNC3','prob_XGB']].max(axis=1)
min_vals = prob[['prob_KNC3','prob_XGB']].min(axis=1)
prob['prob_guess'] = np.where(prob['mean'] >= 0.65, max_vals, prob['prob_guess'])
prob['prob_guess'] = np.where(prob['mean'] < 0.35, min_vals, prob['prob_guess'])
                                                                        
prob['label'] = np.where(prob['mean']>=0.5, 1,0)
prob['original'] = Y

print(log_loss(Y[70000:],prob['prob_guess'][70000:]))
print(log_loss(Y[70000:],prob['mean'][70000:]))
print(log_loss(Y[70000:],prob['prob_FRC'][70000:]))
                      
# aggregate_classifier = VotingClassifier(estimators=[('KNC3', bagg_KNC3), ('SVM', bagg_SVM_isotonic), ('GBC', GBC), ('MPL', MPL), ('RFC', FRC_isotonic)
# , ('LR', bagg_LR), ('GNB', bagg_GNB_isotonic), ('GPC',bagg_GPC_isotonic), ('ADA', ADA_isotonic), ('QDA', QDA_isotonic)], voting='soft')

#aggregate_classifier.fit(X,Y)

# aggregate_classifier.fit(X[:70000,:],Y[:70000])

print(accuracy_score(Y[:10000],aggregate_classifier.predict(X[:10000,:])))

print(log_loss(Y[:10000],aggregate_classifier.predict_proba(X[:10000,:])))

print(accuracy_score(Y[70000:],aggregate_classifier.predict(X[70000:,:])))

print(log_loss(Y[70000:],aggregate_classifier.predict_proba(X[70000:,:])))

result = aggregate_classifier.predict_proba(X_sub)[:,1]
np.savetxt("C:\\Users\\ses516\\Desktop\\Data Mining\\Final Project\\submission.csv", result, delimiter=",")

## TO CHECK FOR OVERFITTING

# print(accuracy_score(Y[:70000],MPL.predict(X[:70000,:])))

# print(log_loss(Y[:70000],MPL.predict_proba(X[:70000,:])))

# print(accuracy_score(Y[70000:],MPL.predict(X[70000:,:])))

# print(log_loss(Y[70000:],MPL.predict_proba(X[70000:,:])))