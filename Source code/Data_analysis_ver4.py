# -*- coding: utf-8 -*-
import keras
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

## FILE DESCRIPTION: This version includes Keras Neural Network model and balance subset generation function

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

day_count = pd.DataFrame({'day_count':data_1.groupby('enrollment_id').day.nunique()}).reset_index()
time_diff = pd.DataFrame({'time_diff':data_1.groupby(['enrollment_id','day'])['hour'].max()-data_1.groupby(['enrollment_id','day'])['hour'].min()}).reset_index()
time_diff = time_diff.replace({'time_diff':{0:1}})
entire_time = pd.DataFrame({'entire_time':time_diff.groupby('enrollment_id')['time_diff'].sum()}).reset_index()
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
                     , month = pd.Series(month['month'])       , hour = pd.Series(hour['hour']) 
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

#####################################################################################################################################  

def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    
    return xs,ys
    
#####################################################################################################################################  
               
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Reshape, MaxPooling2D, Flatten, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.optimizers import SGD

# baseline model

X_NN = META_feat.reshape(META_feat.shape[0],1, META_feat.shape[1], 1)
X_NN_train = META_feat[:50400,:].reshape(META_feat[:50400,:].shape[0],1, META_feat.shape[1], 1)
X_NN_test = META_feat[50400:,:].reshape(META_feat[50400:,:].shape[0],1, META_feat.shape[1], 1)

def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=15, kernel_initializer='normal', activation='relu'))
	model.add(Dense(70, kernel_initializer='normal', activation='selu'))
	model.add(Dense(40, kernel_initializer='normal', activation='selu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

conv = Sequential()
conv.add(Conv2D(20, (1,14), input_shape = X_NN.shape[1:4], activation = 'relu'))
conv.add(MaxPooling2D(1,2))
conv.add(Flatten())
conv.add(Dense(60, input_dim=29, kernel_initializer='normal', activation='selu'))
conv.add(Dense(40, kernel_initializer='normal', activation='selu'))
conv.add(Dense(20, kernel_initializer='normal', activation='selu'))
conv.add(Dense(1, activation = 'sigmoid'))
sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
conv.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['binary_accuracy'])
conv.fit(X_NN_train, Y_NN_train, batch_size = 500, epochs = 100, verbose = 0)
predictDATA = conv.predict_proba(X_NN)
print(log_loss(Y[:50400],np.ndarray.tolist(predictDATA[:50400])))
print(log_loss(Y[50400:],np.ndarray.tolist(predictDATA[50400:72325])))

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=64, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(estimator, X_NN[:50400,:,1], Y[:50400], cv=kfold)
results_test = cross_val_score(estimator, X_NN[50400:,:,1], Y[50400:], cv=kfold)
fitDATA = estimator.fit(X_NN[50400:,:,1], Y[50400:])
predictDATA = estimator.predict_proba(X)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("Results: %.2f%% (%.2f%%)" % (results_test.mean()*100, results_test.std()*100))
print(log_loss(Y[:50400],predictDATA[:50400,1]))
print(log_loss(Y[50400:],predictDATA[50400:,1]))

#####################################################################################################################################

   
### Classifiers
KNC2p = np.zeros((len(X_new),1))
KNC4p = np.zeros((len(X_new),1))
KNC8p = np.zeros((len(X_new),1))
KNC16p = np.zeros((len(X_new),1))
KNC_logp = np.zeros((len(X_new),1))
GBCp = np.zeros((len(X_new),1))
MPLp = np.zeros((len(X_new),1))
RFCp = np.zeros((len(X_new),1))
LRegp = np.zeros((len(X_new),1))
LLRegp = np.zeros((len(X_new),1))
ADAp = np.zeros((len(X_new),1))


X_train = X[:50400,:]
Y_train = Y[:50400]
X_test = X[50400:,:]
Y_test = Y[50400:]
    
KNC2 = KNeighborsClassifier(n_neighbors=2)
KNC2.fit(X_train,Y_train)
KNC2p = KNC2.predict(X_new)

KNC4 = KNeighborsClassifier(n_neighbors=4)
KNC4.fit(X_train,Y_train)
KNC4p = KNC4.predict(X_new)

KNC8 = KNeighborsClassifier(n_neighbors=8)
KNC8.fit(X_train,Y_train)
KNC8p = KNC8.predict(X_new)

KNC16 = KNeighborsClassifier(n_neighbors=16)
KNC16.fit(X_train,Y_train)
KNC16p = KNC16.predict(X_new)

KNC_log = KNeighborsClassifier(n_neighbors=3)
KNC_log.fit(np.log(10+X_train),Y_train)
KNC_logp = KNC_log.predict(np.log(10+X_new))

# print(log_loss(Y[:50400],KNC3[:50400,:]))
# print(log_loss(Y[50400:],KNC3[50400:,:]))
# print(KNC3.score(X[:50400,:],Y[:50400]))
# print(KNC3.score(X[50400:,:],Y[50400:]))

GBC = GradientBoostingClassifier(n_estimators=100, loss = 'deviance', min_samples_split=500)
GBC.fit(X_train,Y_train)
GBCp = GBC.predict(X_new)
     
RFC = RandomForestClassifier(n_estimators=250, criterion='gini', max_depth=None, min_samples_split=0.05, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                            max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=5, random_state=None, 
                            verbose=0, warm_start=False, class_weight=None)
RFC.fit(X_train,Y_train)
RFCp = RFC.predict(X_new)
    
LReg = LR()
LReg.fit(X_train,Y_train)
LRegp = LReg.predict(X_new)

LLReg = LR()
LLReg.fit(np.log(10+X_train),Y_train)
LLRegp = LLReg.predict(np.log(10+X_new))

ADA = AdaBoostClassifier()
ADA.fit(X_train,Y_train)
ADAp = ADA.predict(X_new)

KNC2_prob = np.median(KNC2p[:,1:],axis=1)
KNC4_prob = np.median(KNC4p[:,1:],axis=1)
KNC8_prob = np.median(KNC8p[:,1:],axis=1)
KNC16_prob = np.median(KNC16p[:,1:],axis=1)
KNC_log_prob = np.median(KNC_logp[:,1:],axis=1)
GBC_prob = np.median(GBCp[:,1:],axis=1)
MPL_prob = np.median(MPLp[:,1:],axis=1)
RFC_prob = np.median(RFCp[:,1:],axis=1)
LReg_prob = np.median(LRegp[:,1:],axis=1)
LLReg_prob = np.median(LLRegp[:,1:],axis=1)
ADA_prob = np.median(ADAp[:,1:],axis=1)


# XGB = xgb.XGBClassifier(max_depth=6, n_estimators=1000, learning_rate=0.09,min_child_weight=0.5,eta=0.03,objective='binary:logistic', seed=99,nthread=5, reg_lambda = 100)
# XGB = xgb.XGBClassifier(max_depth=7, n_estimators=1000, learning_rate=0.05,min_child_weight=0.5,eta=0.03,objective='binary:logistic', seed=99,nthread=5, reg_lambda = 100)
# XGB = xgb.XGBClassifier(max_depth=5, n_estimators=1000, learning_rate=0.08,min_child_weight=0.5,eta=0.03,objective='binary:logistic', seed=99,nthread=5, reg_lambda = 100)

## meta-feature

KNC_log = KNeighborsClassifier(n_neighbors=3)
Y_new = np.zeros((len(X_new),1))
KNC_log.fit(np.log(10+X_new),Y_new)

A = list(KNC_log.kneighbors(np.log(10+X_new),6))[0].tolist()
dist = pd.DataFrame(A).reset_index()
dist2 = np.array(dist[[0,1]].sum(axis=1))
dist4 = np.array(dist[[0,1,2,3]].sum(axis=1))
dist6 = np.array(dist[[0,1,2,3,4,5]].sum(axis=1))

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(X_new)
clusters = kmeans.labels_

META_feat = np.column_stack((X_new,clusters,dist2,dist4,dist6, KNC2p, KNC4p, 
                        KNC8p, KNC16p, KNC_logp, GBCp, RFCp, LRegp, LLRegp, 
                        ADAp))
                        

META_feat = np.column_stack((X_new,clusters,dist2,dist4,dist6))
                        
X_META = META_feat[:72325,:]

X_train = X_META[:50400,:]
Y_train = Y[:50400]

X_test = X_META[50400:,:]
Y_test = Y[50400:]

XGB = xgb.XGBClassifier(
 learning_rate =0.1,
 n_estimators=121,
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
# rounds, 21 for 6, 23 for 5, 18 for 4


prob_XGB = XGB.predict_proba(META_feat)[:,1]
print(log_loss(y_train,prob_XGB[:50400]))
print(log_loss(y_valid,prob_XGB[50400:72325]))
plot_importance(XGB,importance_type='gain')
plt.show()

XGB.fit(X_META, Y)
prob_XGB = XGB.predict_proba(META_feat)[:,1]
result = prob_XGB[72325:]

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

XGB.fit(X_META,Y)
result = XGB.predict_proba(X_sub)[:,1]

np.savetxt("C:\\Users\\ses516\\Desktop\\Data Mining\\Final Project\\submission.csv", result, delimiter=" ")


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