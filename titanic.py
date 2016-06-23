# Importing a few necessary libraries
import numpy as np
import math  as math
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
import re

from IPython.display import display

in_file = 'D:/python/Udacity/Kaggle/Titanic/data/train.csv'

full_data = pd.read_csv(in_file)
sample_test = pd.read_csv('D:/python/Udacity/Kaggle/Titanic/data/test.csv')

#display(sample_test.head())
#display(full_data.head())

full_data = full_data.iloc[np.random.permutation(len(full_data))]

full_data_new = full_data[['PassengerId','Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
test_data_new = sample_test[['PassengerId','Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
target = full_data[['PassengerId', 'Survived']]
full_data_new['type'] = 'train'
test_data_new['type'] = 'test'

full_data_new = full_data_new.append(test_data_new)
full_data_new.index = range(len(full_data_new))

new_order = np.random.permutation(len(full_data_new))
full_data_new = full_data_new.iloc[new_order]

#preprocess data


new_features= pd.DataFrame(columns=[['PassengerId','is_no_name', 'is_no_cabin', 'is_no_age', 'sex_by_name', 'last_name', 'first_name', 'family', 'cabin_num', 'title']])
for index, row in full_data_new.iterrows():
    temp_noname = 0
    temp_nocabin = 0
    temp_noage = 0
    last_name1 = ''
    last_name = ''
    first_name = ''
    temp_id = row['PassengerId']
    temp_family = row['SibSp']+row['Parch']
    temp_cabin = 0
    if pd.isnull(row['Name']) == True:
        temp_noname = 1
    else:
        temp_noname = 0
    if pd.isnull(row['Cabin']) == True:
        temp_nocabin =1
    else:
        temp_nocabin =0
    if pd.isnull(row['Age']) == True:
        temp_noage = 1
    else:
        temp_noage = 0
    if row['Sex'] == 'male':
        sex_by_name = 1
    else:
        sex_by_name = 0
    if pd.isnull(row['Cabin']) == False:
        temp_cabin = len(pd.Series(row['Cabin']).str.split(' ')[0])
    else:
        temp_cabin = 0
    if pd.isnull(row['Name']) == False:
        temp_name1 = pd.Series(row['Name']).str.split(', ')[0]
        temp_title = pd.Series(temp_name1[1]).str.split('.')[0][0]
        if temp_title not in ('Mr','Miss','Mrs','Master'):
            temp_title = 'rare'
        if pd.Series(['Mr.']).isin(pd.Series(row['Name']).str.split(' ')[0])[0] == True:
            sex_by_name = 1
            last_name1 = pd.Series(row['Name']).str.split('Mr.')[0][0]
            first_name = pd.Series(row['Name']).str.split('Mr.')[0][1]
        elif (pd.Series(['Miss.']).isin(pd.Series(row['Name']).str.split(' ')[0])[0] == True) or (pd.Series(['Mrs.']).isin(pd.Series(row['Name']).str.split(' ')[0])[0] == True):
            sex_by_name = 0
            if pd.Series(['Miss.']).isin(pd.Series(row['Name']).str.split(' ')[0])[0] == True:
                last_name1 = pd.Series(row['Name']).str.split('Miss.')[0][0]
                first_name = pd.Series(row['Name']).str.split('Miss.')[0][1]
            else:
                last_name1 = pd.Series(row['Name']).str.split('Mrs.')[0][0]
                first_name = pd.Series(row['Name']).str.split('Mrs.')[0][1]
        else:
            last_name1 = pd.Series(row['Name']).str.split(temp_title)[0][0]
            first_name = pd.Series(row['Name']).str.split('.')[0][1]
        last_name = re.sub('[^A-Za-z0-9]+', '', last_name1)
    new_features = new_features.append(pd.DataFrame([[temp_id, temp_noname, temp_nocabin, temp_noage, sex_by_name, last_name, first_name, temp_family, temp_cabin, temp_title]],
                                                    columns=['PassengerId', 'is_no_name', 'is_no_cabin', 'is_no_age', 'sex_by_name', 'last_name', 'first_name', 'family', 'cabin_num', 'title']))




new_features.index = range(len(new_features))
full_data_new = pd.merge(full_data_new, new_features, on = 'PassengerId')
family_size = pd.DataFrame(pd.Series(full_data_new['last_name']).value_counts(dropna=False))
family_size['family_size'] =  family_size.last_name
family_size['last_name'] =  family_size.index
family_size.index = range(len(family_size))
full_data_new = pd.merge(full_data_new, family_size, on = 'last_name')

test = full_data_new[['last_name']]
full_data_new[['Cabin', 'last_name']].groupby(['last_name']).agg('count')


for column in full_data_new:
    if column in ('Sex', 'Embarked', 'title'):
        new_col = pd.get_dummies(full_data_new[column], prefix=column, dummy_na=True)
        full_data_new = pd.concat([full_data_new, new_col], axis=1)


new_features1= pd.DataFrame(columns=[['PassengerId','is_child', 'is_mom', 'is_dad', 'real_fare']])
for index, row in full_data_new.iterrows():
    temp_id = row['PassengerId']
    if row['Age']<=15:
        is_child = 1
    else:
        is_child = 0
    if row['Age']>=18 and row['Parch']>=1 and row['Sex'] == 'female':
        is_mom = 0
        for index1, row1 in full_data_new[full_data_new.last_name==row['last_name']].iterrows():
            if (row['Age']- row1['Age'])>=16 and row1['Parch']>=1:
                is_mom = 1
    else:
        is_mom = 0
    if row['Age']>=18 and row['Parch']>=1 and row['Sex'] == 'male':
        is_dad = 0
        for index1, row1 in full_data_new[full_data_new.last_name==row['last_name']].iterrows():
            if (row['Age']- row1['Age'])>=18 and row1['Parch']>=1:
                is_dad = 1
    else:
        is_dad = 0
    temp_fare = full_data_new[full_data_new.Pclass == row['Pclass']][['Fare']].mean(skipna=True, axis=0)[0]        
    new_features1 = new_features1.append(pd.DataFrame([[temp_id, is_child, is_mom,is_dad, temp_fare]],
                                                    columns=['PassengerId', 'is_child', 'is_mom', 'is_dad', 'real_fare']))

full_data_new = pd.merge(full_data_new, new_features1, on = 'PassengerId')

full_data_new['SibSp'].fillna(int(float(full_data_new[['SibSp']].mean(skipna=True, axis=0)[0])), inplace=True)
full_data_new['Parch'].fillna(int(float(full_data_new[['Parch']].mean(skipna=True, axis=0)[0])), inplace=True)
#full_data_new['Fare'].fillna(int(float(full_data_new[['Fare']].mean(skipna=True, axis=0))), inplace=True)



full_data_age = full_data_new[full_data_new.Age >= 0][['Pclass', 'SibSp', 'Parch', 'real_fare', 'Sex_female', 'Sex_male',
       'Sex_nan', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Embarked_nan',
       'is_no_name', 'is_no_cabin', 'is_no_age', 'sex_by_name', 'family',
       'cabin_num', 'is_child', 'is_mom', 'is_dad']]
target_age = full_data_new[full_data_new.Age >= 0][['Age']]


full_data_age_pred= full_data_new[pd.isnull(full_data_new.Age)==True][['Pclass', 'SibSp', 'Parch', 'real_fare', 'Sex_female', 'Sex_male',
       'Sex_nan', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Embarked_nan',
       'is_no_name', 'is_no_cabin', 'is_no_age', 'sex_by_name', 'family',
       'cabin_num', 'is_child', 'is_mom', 'is_dad']]

from sklearn import linear_model
from sklearn import grid_search
from sklearn.metrics import make_scorer, mean_squared_error, f1_score, accuracy_score

score_func = make_scorer(mean_squared_error, greater_is_better=False)

Lassoreg = linear_model.Lasso(max_iter = 500,random_state = 42)
parameters ={'alpha': [x*0.01 for x in range(1, 100)]}
clf = grid_search.GridSearchCV(estimator=Lassoreg, param_grid=parameters, cv=10, scoring=score_func)
clf.fit(full_data_age, target_age)
target_age_pred = pd.DataFrame(clf.predict(full_data_age_pred), columns=['Age'])
target_age_pred.index = full_data_age_pred.index
target_age = target_age.append(target_age_pred)
full_data_new.drop(['Age'], axis=1, inplace=True)
full_data_new = pd.concat([full_data_new,target_age],axis=1)

le = preprocessing.LabelEncoder()
le.fit(full_data_new['last_name'].unique().tolist())
name_coded = le.fit_transform(full_data_new['last_name'])
name_uncode = list(le.inverse_transform(name_coded))

full_data_new['name_coded'] = pd.DataFrame(name_coded)

test_data_new = full_data_new[full_data_new.type == 'test'].sort_values(by=['PassengerId'])
full_data_new = full_data_new[full_data_new.type == 'train']
test_id = test_data_new[['PassengerId']]



# RANDOM SHUFFLE DATA

new_order = np.random.permutation(len(full_data_new))
full_data_new = pd.merge(full_data_new, target, on = 'PassengerId')

full_data_new = full_data_new.iloc[new_order]

target = full_data_new.Survived

test_data_new.drop(
    ['PassengerId', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'last_name', 'first_name', 'family_size',
     'is_no_name', 'Sex_nan', 'type', 'Fare', 'title'],
    axis=1, inplace=True)
full_data_new.drop(
    ['PassengerId', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket', 'last_name', 'first_name', 'family_size',
     'is_no_name', 'Sex_nan', 'type', 'Fare', 'title', 'Survived'],
    axis=1, inplace=True)


test_data_new.columns.values
full_data_new.columns.values
full_data_new.dtypes

test_data_new.shape
target.shape
print 'finished clean featuers'



# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.tree import DecisionTreeClassifier


randomforest = RandomForestClassifier(random_state=24, n_estimators=800, min_samples_split = 50,  oob_score=True)

# USE MSE AS SCORE FUNCTION
from sklearn.metrics import make_scorer, mean_squared_error, f1_score, accuracy_score

# tree_score= make_scorer(mean_squared_error ,greater_is_better=False)
tree_score = make_scorer(accuracy_score, greater_is_better=True)

score_record = pd.DataFrame(columns=['feature_num', 'max_depth', 'min_samples_split', 'score'])
parameters = {'max_depth': [x for x in range(3, 27)]}

for feature_number in range(3, 27):
    full_data_processed_new = SelectKBest(f_regression, k=feature_number).fit_transform(full_data_new,target)
    clf = grid_search.GridSearchCV(estimator=randomforest, param_grid=parameters, cv=10, scoring=tree_score)
    clf.fit(full_data_processed_new, target)
    score_record = score_record.append(
        pd.DataFrame([[feature_number, clf.best_estimator_.max_depth, clf.best_estimator_.min_samples_split,
                       clf.best_score_]],
                     columns=['feature_num', 'max_depth', 'min_samples_split', 'score']))

print 'finished parameter selection'

score_record[score_record.score >= score_record.score.max()-0.005]
best_estimation =score_record[score_record.score >= score_record.score.max()-0.005]
best_estimation = best_estimation[best_estimation.feature_num == best_estimation.feature_num.min()]
best_estimation = best_estimation[best_estimation.score == best_estimation.score.max()]


best_estimation
#best_estimation = pd.DataFrame([[20, 8, 70, 0.821549]], columns=['feature_num', 'max_depth', 'min_samples_split', 'score'])

finalRF = RandomForestClassifier(max_depth=best_estimation.max_depth[0],
                                 min_samples_split=best_estimation.min_samples_split[0], n_estimators=1200,
                                 random_state=24)

# finalRF = RandomForestClassifier(max_depth=62,
#                                  min_samples_split=90, n_estimators=1000,
#                                  random_state=24)
full_data_processed_new = SelectKBest(f_regression, k=int(best_estimation.feature_num[0])).fit_transform(full_data_new,
                                                                                                    target)

finalRF.fit(full_data_processed_new, target)

accuracy_score(target, finalRF.predict(full_data_processed_new))


#PROCESS TEST DATA
feature_index = SelectKBest(f_regression, k=int(best_estimation.feature_num[0])).fit(full_data_new, target)
test_data_processed_new = feature_index.transform(test_data_new)

predicted_value = finalRF.predict(test_data_processed_new)

predicted_value_pd = pd.DataFrame(np.array(predicted_value).transpose(), columns=['Survived'])
predicted_value_pd.index = range(len(predicted_value_pd))

test_id = test_id[['PassengerId']].astype(int)
test_id.index = range(len(test_id))
predicted_value_pd = pd.concat([test_id, predicted_value_pd], axis=1)
predicted_value_pd.dtypes

predicted_value_pd.to_csv('D:/python/Udacity/Kaggle/Titanic/data/output_result.csv', index=False)

print('final finished')



# CHANGE TO USE ADABOOST


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import grid_search

parameters = {'learning_rate': [x * 0.05 for x in range(1, 21)], 'n_estimators': [x * 50 for x in range(1,9) ]}
adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3, min_samples_split=50), random_state=42)

# USE MSE AS SCORE FUNCTION
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score

tree_score = make_scorer(accuracy_score, greater_is_better=True)

score_record = pd.DataFrame(columns=['feature_num', 'learning_rate', 'n_estimators', 'min_samples_split', 'score'])
for feature_number in range(3, 27):
    full_data_processed_new = SelectKBest(f_regression, k=feature_number).fit_transform(full_data_new,target)
    adaboostgs = grid_search.GridSearchCV(estimator=adaboost, param_grid=parameters, cv=10, scoring=tree_score)
    adaboostgs.fit(full_data_processed_new, target)
    score_record = score_record.append(
        pd.DataFrame([[feature_number, adaboostgs.best_estimator_.learning_rate, adaboostgs.best_estimator_.n_estimators, 50,
                       adaboostgs.best_score_]],
                     columns=['feature_num', 'learning_rate', 'n_estimators', 'min_samples_split', 'score']))

print 'finished parameter selection'
# tree_score= make_scorer(output_restric ,greater_is_better=False)


score_record[score_record.score >= score_record.score.max()-0.005]
best_estimation =score_record[score_record.score >= score_record.score.max()-0.005]
best_estimation = best_estimation[best_estimation.feature_num == best_estimation.feature_num.min()]
best_estimation = best_estimation[best_estimation.score == best_estimation.score.max()]

print('finished modeling')

# finalAdaB = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
#                              n_estimators=adaboostgs.best_estimator_.n_estimators,
#                              learning_rate=adaboostgs.best_estimator_.learning_rate,
#                              random_state=42)

finalAdaB = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3,min_samples_split=50),
                               n_estimators=int(best_estimation.n_estimators[0]),
                               learning_rate=best_estimation.learning_rate[0],
                               random_state=42)
                               
finalAdaB = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3,min_samples_split=50),
                               n_estimators=100,
                               learning_rate=0.05,
                               random_state=42)

full_data_processed_new = SelectKBest(f_regression, k=int(best_estimation.feature_num[0])).fit_transform(full_data_new,
                                                                                                    target)


full_data_processed_new = SelectKBest(f_regression, k=24).fit_transform(full_data_new,
                                                                                                    target)

finalAdaB.fit(full_data_processed_new, target)

accuracy_score(target, finalAdaB.predict(full_data_processed_new))


#PROCESS TEST DATA
feature_index = SelectKBest(f_regression, k=int(best_estimation.feature_num[0])).fit(full_data_new, target)
feature_index = SelectKBest(f_regression, k=24).fit(full_data_new, target)
test_data_processed_new = feature_index.transform(test_data_new)

predicted_value = finalAdaB.predict(test_data_processed_new)

predicted_value_pd = pd.DataFrame(np.array(predicted_value).transpose(), columns=['Survived'])
predicted_value_pd.index = range(len(predicted_value_pd))

test_id = test_id[['PassengerId']].astype(int)
test_id.index = range(len(test_id))
predicted_value_pd = pd.concat([test_id, predicted_value_pd], axis=1)
predicted_value_pd.dtypes

predicted_value_pd.to_csv('D:/python/Udacity/Kaggle/Titanic/data/output_result.csv', index=False)

print('final finished')
