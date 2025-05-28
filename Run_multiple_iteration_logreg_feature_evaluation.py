"""
Feature Selection using Lasso Regularization with Cross-Validation
Author: Venus
Date: 5.27.2025

This script implements feature selection using Lasso regularization with cross-validation.It uses
10 outer folds and  9 inner fold for nested cross-validation in regression.
It processes metrics measured from EEG data to identify significant features that distinguish between sleep and wake states.
The script performs multiple iterations with different random seeds to ensure robust feature selection.

Output files:
1. feature_selection_frequencies.csv - Shows how often each feature was selected
2. feature_selection_detailed_results.csv - Detailed results for each fold in each iteration
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold, cross_val_score
from collections import Counter

import scipy.io



# data = scipy.io.loadmat('allResultsV6.mat')
# print(data.keys())
# npData = data['allResultsAsCell']
# test_numpy = npData[0]
#
# myData = pd.DataFrame(data['allResultsAsCell'])
# test_dataframe = myData[3]
# psdDelta = myData['Power_Spectral_Density_Delta']
# variables = data['allResultsAsCell'][0]

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


def Get_Cross_validation_Results (x_test, y_test, model_cv, featuresToUse):
    """
        Gets the results from running a logistic regression with nested cross validation
         and return accuracy and selected features.

        Args:
            x_test (array-like): Test features
            y_test (array-like): Test labels
            model_cv: Trained model
            featuresToUse (list): List of feature names

        Returns:
            tuple: (accuracy, selected_features)
                - accuracy: Mean accuracy on test set
                - selected_features: List of selected feature names
        """
    # Remove 'scaled_' prefix from feature names
    featuresToUse = [features.replace('scaled_','') for features in featuresToUse ]

    # Get coefficients and create DataFrame
    coefs = pd.DataFrame({'features' : featuresToUse,
                           'coefficient': model_cv.coef_[0]})

    # Filter out non-zero coefficients
    coefs = coefs[coefs['coefficient'] != 0]

    # Sort features by absolute coefficient value
    selectedFeatures = coefs.sort_values('coefficient', key=abs, ascending=False)['features'].tolist()

    # Calculate accuracy
    predictions = model_cv.predict(x_test)
    accuracy = np.mean(predictions == y_test)

    return accuracy, selectedFeatures


def Run_Single_Iteration(seed, rawData, featuresToUse):
    """
        Run a single iteration of the feature selection process. Which uses a logistic regression
         model with lasso regularization

        Args:
            seed (int): Random seed for reproducibility
            rawData (DataFrame): Raw input data
            featuresToUse (list): List of feature names to use

        Returns:
            tuple: (allSelectedFeatures, allAccuracies, allCVScores, foldResults)
                - allSelectedFeatures: List of all selected features
                - allAccuracies: List of accuracies for each fold
                - allCVScores: List of CV scores for each fold
                - foldResults: List of dictionaries containing detailed results for each fold
        """

    np.random.seed(seed)
    # Choose scaling function
    scale = Range_Scale
    # Scale features for both datasets
    features_to_scale = featuresToUse

    # Prepare sleep data
    sleepData = rawData[
        (rawData['sleep_or_awake'] == 'sleep') & (rawData['pre_or_post'] == 'pre')
    ].groupby('patient_ID').sample(n=1)

    #Prepare the wake Data
    wakeData = rawData [
        (rawData['sleep_or_awake'] == 'wake') & (rawData['pre_or_post'] == 'pre')
        ].groupby('patient_ID').sample(n=1)


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

    #initialize variables
    allSelectedFeatures = []
    allAccuracies = []
    allCVScores = []
    foldResults = []

    # Outer cross-validation loop
    for i in range(numOuterFolds):
        # Split data into test and train sets
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

        # Set up inner cross-validation folds
        numInnerFolds = 9

        # # Create custom cross-validation splits
        # cv = KFold(n_splits=numInnerFolds, shuffle=False)
        # for k,(trainindx,testindx) in enumerate(cv.split(x_train)):
        #     print(f'foldID {k}')
        #     print(f'training indices {trainindx}')
        #     print(f'testing indices {testindx}')


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

        # Get results for the current fold
        accuracy, selected_features = Get_Cross_validation_Results(x_test, y_test, model, featuresToUse)
        cvScores = list(model.scores_.values())[0]
        cvMeanScores = np.mean(cvScores)

        foldResults.append({
            'Fold': i + 1,
            'Selected_Features':selected_features,
            'Accuracy': accuracy,
            'Mean_CV_Scores':cvMeanScores,
            'CV_Scores': cvScores,
        })
        allSelectedFeatures.extend(selected_features)
        allAccuracies.append(accuracy)
        allCVScores.append(cvMeanScores)

    return allSelectedFeatures, allAccuracies, allCVScores,foldResults

def run_LR_Lasso_multiple_iterations(numItr):

    """
    Run multiple iterations of the feature selection process.

    Args:
        numItr (int): Number of iterations to run

    Outputs:
        - feature_selection_frequencies.csv: Feature selection frequencies
        - feature_selection_detailed_results.csv: Detailed results for each fold
    """
    # Read and prepare data
    rawData = pd.read_csv('allPatinetsMetrics_Cz_train.csv')

    # Extract feature metrics
    excludeColumns = ['patient_ID', 'file_name', 'sleep_or_awake', 'pre_or_post',
                      'case_or_control', 'epoch_start_index', 'epoch_stop_index',
                      'electrod_channel', 'test_or_train']

    featureMetric = [col for col in rawData.columns if col not in excludeColumns]

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

    # Standardize sleep/wake labels
    rawData['sleep_or_awake'] = rawData['sleep_or_awake'].replace({
        'Sleep1': 'sleep', 'Sleep2': 'sleep',
        'Wake1': 'wake', 'Wake2': 'wake'
    })

    allSelectedFeatures = []
    allFoldResults = []  # List to store all fold results

    # Run multiple iterations
    for i in range(numItr):
        #generate a random number
        seed = np.random.randint(1,1000)
        selectedFeatures, accuracy, cvMeanScores,foldResults = Run_Single_Iteration(seed, rawData, featureMetric)

        # Add iteration information to fold results
        for foldResult in foldResults:
            foldResult['Iteration'] = i + 1
            foldResult['Seed'] = seed
        #Combine results from all iterations
        allFoldResults.extend(foldResults)
        allSelectedFeatures.extend(selectedFeatures)


    # Calculate feature frequencies
    featureFrequencies = Counter(allSelectedFeatures)
    totalSelectionRounds = numItr*10

    # Create frequency DataFrame
    freqDf  = pd.DataFrame({
        'Feature': list(featureFrequencies.keys()),
        'Frequency' : list(featureFrequencies.values()),
        'Percentage' : [freq / totalSelectionRounds * 100 for freq in  featureFrequencies.values()]
    })
    freqDf = freqDf.sort_values('Frequency', ascending=False)
    freqDf.to_csv('feature_selection_frequencies.csv', index=False)

    # Create detailed results DataFrame
    max_features = max(len(result['Selected_Features']) for result in allFoldResults)
    max_scores = max(len(result['CV_Scores']) for result in allFoldResults)

    resultDf = pd.DataFrame(columns=['Seed', 'Iteration', 'Fold', 'Accuracy', 'Mean_CV_Scores',
                                     'num_features'] +
                                    [f'feature_{i + 1}' for i in range(max_features)]+
                                    [f'CV_Score_{k+1}' for k in range(max_scores)])

    # Fill in results DataFrame
    for i, result in enumerate(allFoldResults):
        resultDf.loc[i, 'Seed'] = result['Seed']
        resultDf.loc[i, 'Iteration'] = result['Iteration']
        resultDf.loc[i, 'Fold'] = result['Fold']
        resultDf.loc[i, 'Accuracy'] = result['Accuracy']
        resultDf.loc[i, 'Mean_CV_Scores'] = result['Mean_CV_Scores']
        resultDf.loc[i, 'num_features'] = len(result['Selected_Features'])

        # Store selected features
        for j, feature in enumerate(result['Selected_Features']):
            resultDf.loc[i, f'feature_{j + 1}'] = feature

        # Store all CV scores
        for k, cvScore in enumerate(result['CV_Scores']):
            resultDf.loc[i,f'CV_Score_{k+1}'] = cvScore


    # Save  results
    resultDf.to_csv('feature_selection_detailed_results.csv', index=False)


if __name__ == "__main__":
    run_LR_Lasso_multiple_iterations(5)












