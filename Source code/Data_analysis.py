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

## FILE DESCRIPTION: This version includes very first tries to find an individual classifer by comparing different classifiers' results

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

### Classifiers

KNC3 = KNeighborsClassifier(n_neighbors=3)
# bagg_KNC3 = Bag(base_estimator = KNC3,n_estimators=200,bootstrap='true',max_samples=500, oob_score=False)
KNC3.fit(X[:50400,:],Y[:50400])
R = KNC3.predict_proba(X)
print(log_loss(Y[:50400],KNC3[:50400,:]))
print(log_loss(Y[50400:],KNC3[50400:,:]))
print(KNC3.score(X[:50400,:],Y[:50400]))
print(KNC3.score(X[50400:,:],Y[50400:]))
A = list(KNC3.kneighbors(X,6))[0].tolist()
dist = pd.DataFrame(A).reset_index()
dist['sum2'] = dist[[0,1]].sum(axis=1)
dist['sum4'] = dist[[0,1,2,3]].sum(axis=1)
dist['sum6'] = dist[[0,1,2,3,4,5]].sum(axis=1)

SVM = svm.SVC(kernel='rbf',probability=True, class_weight='balanced')
bagg_SVM = Bag(base_estimator = SVM,n_estimators=200,bootstrap='true',max_samples=1000)
bagg_SVM_isotonic = CalibratedClassifierCV(bagg_SVM, cv=2, method='isotonic')
bagg_SVM_isotonic.fit(X[:50400,:],Y[:50400])
prob_SVM = bagg_SVM_isotonic.predict_proba(X)
print(log_loss(Y[:50400],prob_SVM[:50400,:]))
print(log_loss(Y[50400:],prob_SVM[50400:,:]))

GNB = gnb()
bagg_GNB = Bag(base_estimator = GNB,n_estimators=50,bootstrap='true',max_samples=500)
bagg_GNB_isotonic = CalibratedClassifierCV(bagg_GNB, cv=2, method='isotonic')
bagg_GNB_isotonic.fit(X[:70000,:],Y[:70000])
prob_GNB = bagg_GNB_isotonic.predict_proba(X)

GBC = GradientBoostingClassifier(n_estimators=100, loss = 'deviance', min_samples_split=500)
GBC.fit(X[:70000,:],Y[:70000])
prob_GBC = GBC.predict_proba(X)

MPL = MLPClassifier(alpha=1, activation='relu')
MPL.fit(X[:70000,:],Y[:70000])
prob_MPL = MPL.predict_proba(X)

RFC = RandomForestClassifier(n_estimators=400, criterion='entropy', max_depth = 5, random_state=1,oob_score = True)
FRC_isotonic = CalibratedClassifierCV(RFC, cv=2, method='isotonic')
FRC_isotonic.fit(X[:50400,:],Y[:50400])
prob_FRC = FRC_isotonic.predict_proba(X)
print(log_loss(Y[:50400],prob_FRC[:50400,:]))
print(log_loss(Y[50400:],prob_FRC[50400:,:]))

LR = LR()
bagg_LR = Bag(base_estimator = LR,n_estimators=200,bootstrap='true',max_samples=500, oob_score=False)
bagg_LR.fit(X[:70000,:],Y[:70000])
prob_LR = bagg_LR.predict_proba(X)

ADA = AdaBoostClassifier()
ADA_isotonic = CalibratedClassifierCV(ADA, cv=2, method='isotonic')
ADA_isotonic.fit(X[:70000,:],Y[:70000])
prob_ADA = ADA_isotonic.predict_proba(X)

GPC = GaussianProcessClassifier(kernel = 1.0 * RBF(1.0))
bagg_GPC = Bag(base_estimator = GPC,n_estimators=50,bootstrap='true',max_samples=100, oob_score=False)
bagg_GPC_isotonic = CalibratedClassifierCV(bagg_GPC, cv=2, method='isotonic')
bagg_GPC_isotonic.fit(X[:70000,:],Y[:70000])
prob_GPC = bagg_GPC_isotonic.predict_proba(X)

QDA = QuadraticDiscriminantAnalysis()
QDA_isotonic = CalibratedClassifierCV(QDA, cv=2, method='isotonic')
QDA_isotonic.fit(X[:70000,:],Y[:70000])
prob_QDA = QDA_isotonic.predict_proba(X)

from sklearn.model_selection import KFold
XGB = xgb.XGBClassifier(max_depth=4, n_estimators=1000, learning_rate=0.06,min_child_weight=0.5,eta=0.03,objective='binary:logistic', seed=99,nthread=5, reg_lambda = 100)
KF = KFold(n_splits=10)
prob_XGB = np.zeros((len(X_new),1))
for train_index, test_index in KF.split(X):
    train, valid = X[train_index], X[test_index]
    y_train, y_valid = Y[train_index], Y[test_index]
    eval_set  = [(train,y_train), (valid,y_valid)]
    XGB.fit(train, y_train, eval_set=eval_set,
        eval_metric="auc", early_stopping_rounds=18)
    temp = XGB.predict_proba(X_new)[:,1]
    prob_XGB = np.column_stack((prob_XGB, temp))


prob_XGB_final = prob_XGB[:,1:]
print(log_loss(y_train,prob_XGB[train_index,10]))
print(log_loss(y_valid,prob_XGB[test_index,10]))

result_entire = np.mean(prob_XGB_final,axis=1)
result = result_entire[72325:]

print(log_loss(y_train,result_entire[train_index]))
print(log_loss(y_valid,result_entire[test_index]))

# XGB = xgb.XGBClassifier(max_depth=6, n_estimators=1000, learning_rate=0.09,min_child_weight=0.5,eta=0.03,objective='binary:logistic', seed=99,nthread=5, reg_lambda = 100)
# XGB = xgb.XGBClassifier(max_depth=7, n_estimators=1000, learning_rate=0.05,min_child_weight=0.5,eta=0.03,objective='binary:logistic', seed=99,nthread=5, reg_lambda = 100)
# XGB = xgb.XGBClassifier(max_depth=5, n_estimators=1000, learning_rate=0.08,min_child_weight=0.5,eta=0.03,objective='binary:logistic', seed=99,nthread=5, reg_lambda = 100)
# XGB.fit(feat_final.head(50400),Y[:50400])
# train = X[:50400,:]
# y_train = Y[:50400]
# valid = X[50400:,:]
# y_valid = Y[50400:]

# rounds, 21 for 6, 23 for 5, 18 for 4


XGB.fit(X,Y)
result = XGB.predict_proba(X_sub)[:,1]
np.savetxt("C:\\Users\\ses516\\Desktop\\Data Mining\\Final Project\\submission.csv", result, delimiter=" ")
plot_importance(XGB)
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