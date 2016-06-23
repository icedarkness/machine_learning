# Importing a few necessary libraries
import numpy as np
import math  as math
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing

from IPython.display import display

in_file = 'D:/python/Udacity/Kaggle/data/train.csv'

full_data = pd.read_csv(in_file)
sample_outcome = pd.read_csv('D:/python/Udacity/Kaggle/data/sample_submission.csv')
sample_test = pd.read_csv('D:/python/Udacity/Kaggle/data/test.csv')

display(sample_outcome.head())
display(sample_test.head())
display(full_data.head())

# PREPROCESSING CATEGORICAL DATA

# EXTRACT MORE FEATURE FROM CURRENT DATA
new_features = pd.DataFrame(
    columns=['is_mix', 'AnimalID', 'breed_abbrev', 'is_no_name', 'days', 'neutered', 'color1', 'color2', 'year',
             'quarter', 'month', 'dayofweek'])
for index, row in full_data.iterrows():
    tempstr = row['Breed']
    tempid = row['AnimalID']
    tempbreed = pd.Series(row['Breed']).str.replace('Mix', '').str.strip()[0]
    if pd.isnull(row['AgeuponOutcome']):
        tempdays = np.nan
    else:
        tempdate = pd.Series(row['AgeuponOutcome']).str.split(' ')[0]
        if tempdate[1] in ('year', 'years'):
            tempdays = int(float(tempdate[0])) * 365
        elif tempdate[1] in ('week', 'weeks'):
            tempdays = int(float(tempdate[0])) * 7
        elif tempdate[1] in ('month', 'months'):
            tempdays = int(float(tempdate[0])) * 30
        elif tempdate[1] in ('days', 'day'):
            tempdays = int(float(tempdate[0])) * 1
        else:
            tempdays = int(float(tempdate[0]))
    if pd.isnull(row['Name']):
        is_no_name = 1
    else:
        is_no_name = 0
    if pd.isnull(row['SexuponOutcome']):
        tempneutered = 0
    else:
        if pd.Series(row['SexuponOutcome']).str.split(' ')[0][0] == 'Intact':
            tempneutered = 0
        else:
            tempneutered = 1
    # extract more color:
    if pd.isnull(row['Color']):
        tempcolor1 = np.nan
        tempcolor2 = np.nan
    else:
        if pd.Series(row['Color']).str.split('/').count() > 1:
            tempcolor1 = pd.Series(row['Color']).str.split('/')[0][0]
            tempcolor2 = pd.Series(row['Color']).str.split('/')[0][1]
        else:
            tempcolor1 = pd.Series(row['Color']).str.split('/')[0][0]
            tempcolor2 = np.nan
    # proccess datetime to year quarter month weekdays
    if pd.isnull(row['DateTime']):
        tempyear, tempquarter, tempmonth, tempdayofweek = np.nan, np.nan, np.nan, np.nan
    else:
        tempyear = pd.to_datetime(row['DateTime']).year
        tempquarter = pd.to_datetime(row['DateTime']).quarter
        tempmonth = pd.to_datetime(row['DateTime']).month
        tempdayofweek = pd.to_datetime(row['DateTime']).dayofweek
    if pd.Series(tempstr).str.contains('Mix')[0]:
        new_features = new_features.append(
            pd.DataFrame([[1, tempid, tempbreed, is_no_name, tempdays, tempneutered, tempcolor1, tempcolor2, tempyear,
                           tempquarter, tempmonth, tempdayofweek]],
                         columns=['is_mix', 'AnimalID', 'breed_abbrev', 'is_no_name', 'days', 'neutered', 'color1',
                                  'color2', 'year', 'quarter', 'month', 'dayofweek']))
    else:
        new_features = new_features.append(
            pd.DataFrame([[0, tempid, tempbreed, is_no_name, tempdays, tempneutered, tempcolor1, tempcolor2, tempyear,
                           tempquarter, tempmonth, tempdayofweek]],
                         columns=['is_mix', 'AnimalID', 'breed_abbrev', 'is_no_name', 'days', 'neutered', 'color1',
                                  'color2', 'year', 'quarter', 'month', 'dayofweek']))

max_string_value = int(float(new_features[['days']].mean(skipna=True, axis=0)))
new_features['days'].fillna(max_string_value, inplace=True)

new_features.index = range(len(new_features))
new_full_data = pd.merge(full_data, new_features, on='AnimalID')

# DUPLICATES FOR BIAS DATA
pd.Series(new_full_data['OutcomeType']).value_counts(dropna=False)

df_adoption = new_full_data[new_full_data.OutcomeType == 'Adoption']
df_transfer = new_full_data[new_full_data.OutcomeType == 'Transfer']
df_return = new_full_data[new_full_data.OutcomeType == 'Return_to_owner']
df_Euthanasia = new_full_data[new_full_data.OutcomeType == 'Euthanasia']
df_Died = new_full_data[new_full_data.OutcomeType == 'Died']
df_nan = new_full_data[new_full_data.OutcomeType == 'NaN']

new_full_data_final = pd.DataFrame(columns=[list(new_full_data.columns.values)])
new_full_data_final = new_full_data_final.append([df_adoption] * int(float(len(new_full_data) / len(df_adoption))),
                                                 ignore_index=True)
new_full_data_final = new_full_data_final.append([df_transfer] * int(float(len(new_full_data) / len(df_transfer))),
                                                 ignore_index=True)
new_full_data_final = new_full_data_final.append([df_return] * int(float(len(new_full_data) / len(df_return))),
                                                 ignore_index=True)
new_full_data_final = new_full_data_final.append([df_Euthanasia] * int(float(len(new_full_data) / len(df_Euthanasia))),
                                                 ignore_index=True)
new_full_data_final = new_full_data_final.append([df_Died] * int(float(len(new_full_data) / len(df_Died))),
                                                 ignore_index=True)

new_full_data_final = new_full_data

# RANDOM SHUFFLE DATA
new_full_data_final = new_full_data_final.iloc[np.random.permutation(len(new_full_data_final))]

new_full_data_final.index = range(len(new_full_data_final))

# TRANSFORM CATEGORICAL VALUE TO DUMMY
full_data_processed = new_full_data_final[
    ['AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'Color', 'is_mix', 'breed_abbrev', 'is_no_name', 'days',
     'neutered', 'year', 'quarter', 'month', 'dayofweek', 'color1', 'color2']]

for column in new_full_data_final:
    if column in ('AnimalType', 'SexuponOutcome', 'breed_abbrev', 'color1', 'color2', 'AgeuponOutcome'):
        new_col = pd.get_dummies(new_full_data_final[column], prefix=column, dummy_na=True)
        full_data_processed = pd.concat([full_data_processed, new_col], axis=1)

new_out_cate = pd.DataFrame(columns=['out_cate'])
for index, row in new_full_data_final.iterrows():
    tempid = row['AnimalID']
    if row['OutcomeType'] == 'Adoption':
        tempcate = 1
    elif row['OutcomeType'] == 'Transfer':
        tempcate = 2
    elif row['OutcomeType'] == 'Return_to_owner':
        tempcate = 3
    elif row['OutcomeType'] == 'Euthanasia':
        tempcate = 4
    elif row['OutcomeType'] == 'Died':
        tempcate = 5
    else:
        tempcate = 0
    new_out_cate = new_out_cate.append(pd.DataFrame([[tempcate]],
                                                    columns=['out_cate']))

new_out_cate.index = range(len(new_out_cate))

target = pd.get_dummies(new_out_cate['out_cate'], prefix=column, dummy_na=True)
target_combine = new_out_cate['out_cate']

# target=pd.get_dummies(new_full_data_final['OutcomeType'], dummy_na=True)
full_data_processed.drop(
    ['AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'breed_abbrev', 'Color', 'color1', 'color2'],
    axis=1, inplace=True)

# display(full_data_processed.head())
print('finished training data')

# PROCCESS DATA FOR TEST
new_features_test = pd.DataFrame(
    columns=['is_mix', 'ID', 'breed_abbrev', 'is_no_name', 'days', 'neutered', 'color1', 'color2', 'year',
             'quarter', 'month', 'dayofweek'])
for index, row in sample_test.iterrows():
    tempstr = row['Breed']
    tempid = row['ID']
    tempbreed = pd.Series(row['Breed']).str.replace('Mix', '').str.strip()[0]
    if pd.isnull(row['AgeuponOutcome']):
        tempdays = 0
    else:
        tempdate = pd.Series(row['AgeuponOutcome']).str.split(' ')[0]
        if tempdate[1] in ('year', 'years'):
            tempdays = int(float(tempdate[0])) * 365
        elif tempdate[1] in ('week', 'weeks'):
            tempdays = int(float(tempdate[0])) * 7
        elif tempdate[1] in ('month', 'months'):
            tempdays = int(float(tempdate[0])) * 30
        elif tempdate[1] in ('days', 'day'):
            tempdays = int(float(tempdate[0])) * 1
        else:
            tempdays = int(float(tempdate[0]))
    if pd.isnull(row['Name']):
        is_no_name = 1
    else:
        is_no_name = 0
    if pd.isnull(row['SexuponOutcome']):
        tempneutered = 0
    else:
        if pd.Series(row['SexuponOutcome']).str.split(' ')[0][0] == 'Intact':
            tempneutered = 0
        else:
            tempneutered = 1
    # extract more color:
    if pd.isnull(row['Color']):
        tempcolor1 = np.nan
        tempcolor2 = np.nan
    else:
        if pd.Series(row['Color']).str.split('/').count() > 1:
            tempcolor1 = pd.Series(row['Color']).str.split('/')[0][0]
            tempcolor2 = pd.Series(row['Color']).str.split('/')[0][1]
        else:
            tempcolor1 = pd.Series(row['Color']).str.split('/')[0][0]
            tempcolor2 = np.nan
    # proccess datetime to year quarter month weekdays
    if pd.isnull(row['DateTime']):
        tempyear, tempquarter, tempmonth, tempdayofweek = np.nan, np.nan, np.nan, np.nan
    else:
        tempyear = pd.to_datetime(row['DateTime']).year
        tempquarter = pd.to_datetime(row['DateTime']).quarter
        tempmonth = pd.to_datetime(row['DateTime']).month
        tempdayofweek = pd.to_datetime(row['DateTime']).dayofweek
    if pd.Series(tempstr).str.contains('Mix')[0]:
        new_features_test = new_features_test.append(
            pd.DataFrame([[1, tempid, tempbreed, is_no_name, tempdays, tempneutered, tempcolor1, tempcolor2, tempyear,
                           tempquarter, tempmonth, tempdayofweek]],
                         columns=['is_mix', 'ID', 'breed_abbrev', 'is_no_name', 'days', 'neutered', 'color1',
                                  'color2', 'year', 'quarter', 'month', 'dayofweek']))
    else:
        new_features_test = new_features_test.append(
            pd.DataFrame([[0, tempid, tempbreed, is_no_name, tempdays, tempneutered, tempcolor1, tempcolor2, tempyear,
                           tempquarter, tempmonth, tempdayofweek]],
                         columns=['is_mix', 'ID', 'breed_abbrev', 'is_no_name', 'days', 'neutered', 'color1',
                                  'color2', 'year', 'quarter', 'month', 'dayofweek']))

max_string_value = int(float(new_features_test[['days']].mean(skipna=True, axis=0)))
new_features_test['days'].fillna(max_string_value, inplace=True)

new_features_test.index = range(len(new_features_test))
test_data_id = new_features_test[['ID']]
test_data_id['ID'].astype(int)

new_test_data = pd.merge(sample_test, new_features_test, on='ID')

# TRANSFORM CATEGORICAL VALUE TO DUMMY
test_data_processed = new_test_data[
    ['AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'Color', 'is_mix', 'breed_abbrev', 'is_no_name', 'days',
     'neutered', 'year', 'quarter', 'month', 'dayofweek', 'color1', 'color2']]

for column in new_test_data:
    if column in ('AnimalType', 'SexuponOutcome', 'breed_abbrev', 'AgeuponOutcome', 'color1', 'color2'):
        new_col = pd.get_dummies(new_test_data[column], prefix=column, dummy_na=True)
        test_data_processed = pd.concat([test_data_processed, new_col], axis=1)

test_data_processed.drop(
    ['AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'breed_abbrev', 'Color', 'color1', 'color2'],
    axis=1, inplace=True)

# display(test_data_processed.head())
print('finished test data')

# ADD COLUMNS WHICH DO NOT EXIST IN TEST DATA

for column in full_data_processed:
    if column not in test_data_processed:
        test_data_processed[column] = 0

for column in test_data_processed:
    if column not in full_data_processed:
        test_data_processed.drop([column], axis=1, inplace=True)

test_data_processed.shape
full_data_processed.shape

# REORDER COLUMNS
cols = list(full_data_processed.columns.values)
test_data_processed = test_data_processed[cols]

print('finished clean featuers')

# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


randomforest = RandomForestClassifier(random_state=24, n_estimators=100)

# USE MSE AS SCORE FUNCTION
from sklearn.metrics import make_scorer, mean_squared_error, f1_score, accuracy_score

# tree_score= make_scorer(output_restric ,greater_is_better=False)
tree_score = make_scorer(accuracy_score, greater_is_better=True)

score_record = pd.DataFrame(columns=['feature_num', 'max_depth', 'min_samples_split', 'score'])
for feature_number in range(30, 100):
    parameters = {'max_depth': [x for x in range(3, feature_number+1)], 'min_samples_split': [x * 10 for x in range(2, 10)]}
    full_data_processed_new = SelectKBest(f_regression, k=feature_number).fit_transform(full_data_processed,
                                                                                        target_combine)
    clf = grid_search.GridSearchCV(estimator=randomforest, param_grid=parameters, cv=10, scoring=tree_score)
    clf.fit(full_data_processed_new, target_combine)
    score_record = score_record.append(
        pd.DataFrame([[feature_number, clf.best_estimator_.max_depth, clf.best_estimator_.min_samples_split,
                       clf.best_score_]],
                     columns=['feature_num', 'max_depth', 'min_samples_split', 'score']))


score_record[score_record.score == score_record.score.max()]
best_estimation = score_record[score_record.score == score_record.score.max()]


print("best parameter", best_estimation.max_depth[0])
print("best score", best_estimation.score[0])

print('finished modeling')

full_data_processed_new = SelectKBest(f_regression, k=best_estimation.feature_num[0]).fit_transform(full_data_processed,
                                                                                                    target_combine)

randomforest = RandomForestClassifier(max_depth=best_estimation.max_depth[0],random_state=24, n_estimators=100)
parameters = {'min_samples_split': [x * 10 for x in range(2, 10)]}
clf = grid_search.GridSearchCV(estimator=randomforest, param_grid=parameters, cv=10, scoring=tree_score)
clf.fit(full_data_processed_new, target_combine)
print('best score', clf.best_score_)
print('best depth', best_estimation.max_depth[0])
print('best sample split', clf.best_estimator_.min_samples_split)
print('best feature', best_estimation.feature_num[0])

finalRF = RandomForestClassifier(max_depth=best_estimation.max_depth[0],
                                 min_samples_split=clf.best_estimator_.min_samples_split, n_estimators=1000,
                                 random_state=24)

# finalRF = RandomForestClassifier(max_depth=62,
#                                  min_samples_split=90, n_estimators=1000,
#                                  random_state=24)
full_data_processed_new = SelectKBest(f_regression, k=best_estimation.feature_num[0]).fit_transform(full_data_processed,
                                                                                                 target_combine)

finalRF.fit(full_data_processed_new, target_combine)

accuracy_score(target_combine, finalRF.predict(full_data_processed_new))


#PROCESS TEST DATA
feature_index = SelectKBest(f_regression, k=best_estimation.feature_num[0]).fit(full_data_processed, target_combine)
test_data_processed_new = feature_index.transform(test_data_processed)

predicted_value = finalRF.predict_proba(test_data_processed_new)

predicted_value_pd = pd.DataFrame(np.array(predicted_value[:, 0]).transpose(), columns=['Adoption'])
predicted_value_pd['Transfer'] = pd.DataFrame(np.array(predicted_value[:, 1]).transpose())
predicted_value_pd['Return_to_owner'] = pd.DataFrame(np.array(predicted_value[:, 2]).transpose())
predicted_value_pd['Euthanasia'] = pd.DataFrame(np.array(predicted_value[:, 3]).transpose())
predicted_value_pd['Died'] = pd.DataFrame(np.array(predicted_value[:, 4]).transpose())

test = np.array(predicted_value[:, 1])

predicted_value_pd.index = range(len(predicted_value_pd))

predicted_value_pd = predicted_value_pd[['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']]

test_data_id = test_data_id[['ID']].astype(int)
predicted_value_pd = pd.concat([test_data_id, predicted_value_pd], axis=1)
predicted_value_pd.dtypes

predicted_value_pd.to_csv('D:/python/Udacity/Kaggle/data/output_result.csv', index=False)

print('final finished')

# CHANGE TO USE ADABOOST


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import grid_search

parameters = {'n_estimators': [x * 100 for x in range(1, 5)], 'learning_rate': [x * 0.1 for x in range(1, 11)]}
adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), random_state=42)

# USE MSE AS SCORE FUNCTION
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score

# tree_score= make_scorer(output_restric ,greater_is_better=False)
tree_score = make_scorer(accuracy_score, greater_is_better=True)

adaboostgs = grid_search.GridSearchCV(estimator=adaboost, param_grid=parameters, cv=5, scoring=tree_score)
adaboostgs.fit(full_data_processed, target)

mean_squared_error(target, adaboostgs.predict(full_data_processed))
accuracy_score(target, adaboostgs.predict(full_data_processed))

print("best parameter", adaboostgs.best_estimator_)
print("best score", adaboostgs.best_score_)
print("Accuracy: ", accuracy_score(target, adaboostgs.predict(full_data_processed)))

print('finished modeling')

# finalAdaB = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
#                              n_estimators=adaboostgs.best_estimator_.n_estimators,
#                              learning_rate=adaboostgs.best_estimator_.learning_rate,
#                              random_state=42)

finalAdaB = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
                               n_estimators=100,
                               learning_rate=0.2,
                               random_state=42)

finalAdaB.fit(full_data_processed, target)

mean_squared_error(target, finalAdaB.predict(full_data_processed))

predicted_value = pd.get_dummies(finalRF.predict(test_data_processed), dummy_na=True)
predicted_value.columns = ['Adoption', 'Transfer', 'Return_to_owner', 'Euthanasia', 'NAN']
predicted_value['Died'] = 0

predicted_value = predicted_value[['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer', 'NAN']]

predicted_value_pd = pd.concat([test_data_id, predicted_value], axis=1)
predicted_value_pd.drop(['NAN'], axis=1, inplace=True)

print('final finished')

#  OUTPUT TO CSV

import csv
import re

f = open('D:/python/Udacity/Kaggle/data/output_result.csv', 'w', newline='\n')
writer = csv.writer(f)
writer.writerow(list(predicted_value_pd.columns.values))
for index, row in predicted_value_pd.iterrows():
    writer.writerow(row)
f.close()
# print ('finished')

print('output finish')
