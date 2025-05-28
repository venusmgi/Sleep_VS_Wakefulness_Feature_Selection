"""
Feature Selection using Lasso Regularization with Cross-Validation
Author: Venus
Date: 5.27.2025

This script implements feature selection using Lasso regularization with cross-validation. It uses
10 outer folds and  9 inner fold for nested cross-validation in regression.
It processes EEG data to identify significant features that distinguish between sleep and wake states.
The script performs a single iterations with one random seeds to ensure robust feature selection.

Output file:
1. feature_selection_results.csv - Shows the accuracy of each outer-fold of cross validation
and the features that were used in the fold, and the total number of features for each fol
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold, cross_val_score

import scipy.io




def Robust_Scale(data):
    """Scale data using median and IQR"""
    myMedian = data-np.median(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    return (data - myMedian)/iqr

def Z_Score(data):
    """Scale data using mean and standard deviation"""
    return (data - np.mean(data)) / np.std(data)


def Range_Scale(data):
    return (data - np.min(data))/(np.max(data) - np.min(data))


def Run_Cross_validation (x_test, y_test, model_cv, features_to_use):
    features_to_use = [features.replace('scaled_','') for features in features_to_use ]

    coefs = pd.DataFrame({'features' : features_to_use,
                           'coefficient': model_cv.coef_[0]})

    coefs = coefs[coefs['coefficient'] != 0]

    selectedFeatures = coefs.sort_values('coefficient', key=abs, ascending=False)['features'].tolist()

    predictions = model_cv.predict(x_test)
    accuracy = np.mean(predictions == y_test)


    return accuracy, selectedFeatures


def main():

    # Read Data
    rawData = pd.read_csv('allPatinetsMetrics_Cz_train.csv')

    # Extract feature metrics
    excludeColumns = ['patient_ID', 'file_name', 'sleep_or_awake', 'pre_or_post',
                       'case_or_control', 'epoch_start_index', 'epoch_stop_index',
                       'electrod_channel','test_or_train']

    featureMetric = [col for col in rawData.columns if col not in excludeColumns]


    rawData['sleep_or_awake'] = rawData['sleep_or_awake'].replace({
            'Sleep1' : 'sleep', 'Sleep2' : 'sleep',
        'Wake1' : 'wake', 'Wake2' : 'wake'
        })



    # Choose scaling function
    scale = Range_Scale

    # Prepare sleep data
    sleepData = rawData[
        (rawData['sleep_or_awake'] == 'sleep') & (rawData['pre_or_post'] == 'pre')
    ].groupby('patient_ID').sample(n=1)

    #Prepare the wake Data
    wakeData = rawData [
        (rawData['sleep_or_awake'] == 'wake') & (rawData['pre_or_post'] == 'pre')
        ].groupby('patient_ID').sample(n=1)

    # Scale features for both datasets
    features_to_scale = featureMetric

    #     [
    #     'permutation_entropy_delta', 'shannon_entropy_delta',
    #     'permutation_entropy_theta', 'shannon_entropy_theta',
    #     'permutation_entropy_alpha', 'shannon_entropy_alpha',
    #     'permutation_entropy_beta', 'shannon_entropy_beta',
    #     'power_spectral_density_delta', 'power_spectral_density_theta',
    #     'power_spectral_density_alpha', 'power_spectral_density_beta',
    #     'power_spectral_density_broadband', 'amplitude_range',
    #     'spectral_edge_frequency'
    # ]



    for feature in features_to_scale:
        scaled_name = f'scaled_{feature}'
        wakeData[scaled_name] = scale(wakeData[feature])
        sleepData[scaled_name] = scale(sleepData[feature])

    myData = pd.concat([wakeData, sleepData])

    #Define features to use
    featuresToUse = [f'scaled_{feature}' for feature in features_to_scale]


    # Set up cross-validation

    numOuterFolds = 10
    numPatients = len(myData['patient_ID'].unique())
    foldSize = int(np.ceil(numPatients/numOuterFolds))
    resultsList = []


    for i in range(numOuterFolds):
        startIdx = i*foldSize
        stopIdx = min((i+1)*foldSize, numPatients)

        testPatients = myData['patient_ID'].unique()[startIdx:stopIdx]
        trainPatients = np.setdiff1d(myData['patient_ID'].unique(), testPatients)

        #prepare the testing dataset
        testData = myData[myData['patient_ID'].isin(testPatients)]
        y_test = testData['sleep_or_awake'].values
        x_test = testData[featuresToUse].values

        #prepare the training dataset
        trainData = myData[myData['patient_ID'].isin(trainPatients)]
        y_train = trainData['sleep_or_awake'].values
        x_train = trainData[featuresToUse].values

        #assign fold ID
        numInnerFolds = 9

        #Another method for corss validation, but does not work now
        # cv = KFold(n_splits=numInnerFolds, shuffle=False)
        # for i,(trainindx,testindx) in enumerate(cv.split(x_train)):
        #     print(f'foldID {i}')
        #     print(f'training indices {trainindx}')
        #     print(f'testing indices {testindx}')

        # Create custom cross-validation splitter
        foldAssigments = np.repeat(np.arange(1, numInnerFolds+1),
                                   np.ceil(len(trainPatients)/numInnerFolds)
                                   )[:len(trainPatients)]

        patientToFold = dict(zip(trainPatients, foldAssigments))
        foldIds = trainData['patient_ID'].map(patientToFold).values



        cvSlpilts = []

        for fold in range(1 , numInnerFolds+1):
            trainIdx = np.where(foldIds != fold)[0]
            valIdx = np.where(foldIds == fold)[0]
            cvSlpilts.append((trainIdx, valIdx))

        model = LogisticRegressionCV(
            cv=cvSlpilts,
            penalty='l1',
            solver='liblinear',
            Cs=[1.0],  # Single C value as in R code
            scoring='accuracy',
            max_iter=1000,
            random_state=76456687)

        #fit the model
        model.fit(x_train,y_train)

        # Run cross-validation
        accuracy, selected_features = Run_Cross_validation(x_test, y_test, model, featuresToUse)

        resultsList.append({
            'Accuracy' : accuracy,
            'Selected_features': selected_features,
            'CV_scores': np.mean(list(model.scores_.values())[0])  # Get mean CV scores
        })

    # Process results into a DataFrame
    max_features = max(len(result['Selected_features']) for result in resultsList)
    result_df = pd.DataFrame(
        columns=['Accuracy', 'num_features'] + [f'feature_{i + 1}' for i in range(max_features)])

    for i, result in enumerate(resultsList):
        result_df.loc[i, 'Accuracy'] = result['Accuracy']
        result_df.loc[i, 'num_features'] = len(result['Selected_features'])
        for j, feature in enumerate(result['Selected_features']):
            result_df.loc[i, f'feature_{j + 1}'] = feature

    # Save results
    result_df.to_csv(f'feature_selection_results.csv', index=False)
    print("Feature selection completed. Results saved to 'feature_selection_results.csv'")

if __name__ == "__main__":
    main()












