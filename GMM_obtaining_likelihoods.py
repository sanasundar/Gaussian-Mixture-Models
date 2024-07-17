#importing all the necessary packages
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import pickle

def to_predict_labels(h, l, m):
    '''Returns the predicted labels from log-likelihoods of each feature being in the separate GMMs
        
        Parameters
        ----------
        h: ndarray
            The log-likelihoods that the features in the test set belong to landuse category H
        l: ndarray
            The log likelihood that the features in the test set belong to landuse category L
        m: ndarray
            The log-likelihood that the features in the test set belong to the landuse category M
        
        Returns
        --------
        predictions: list
            The list of predicted labels for the features of the test set
        h_scores: list
            The list of likelihoods after removal of backgrounds for a datapoint belonging to landuse category H
        l_scores: list
            The list of likelihoods after removal of backgrounds for a datapoint belonging to landuse category L
        m_score: list
            The list of likelihoods after removal of backgrounds for a datapoint belonging to landuse category M'''
    
    h_scores = []
    l_scores = []
    m_scores = []
    predictions = []
    for i in range(0, len(h)):
        hsc = 2*h[i] - m[i] - l[i]
        lsc = 2*l[i] - h[i] - m[i]
        msc = 2*m[i] - l[i] - h[i]
        h_scores.append(hsc)
        l_scores.append(lsc)
        m_scores.append(msc)
        if hsc is max(hsc, lsc, msc):
            predictions.append('H')
        elif lsc is max(hsc, lsc, msc):
            predictions.append('L')
        else:
            predictions.append('M')
    return predictions, h_scores, l_scores, m_scores

def to_test_model(features_to_split, landuse_categories_to_split, sites_to_split, models_for_testing):
    '''Returns the predicted labels for all folds and confusion matrices for each fold which can be used to calculate the accuracy of the model
        
        Parameters
        ----------
        features_to_split: list
            The list of features of the entire dataset that will be split into training and testing sets
        landuse_categories_to_split: list
            The list of the landuse catergories of the entire dataset that will be split into training and testing sets maintaining proportion of landuse categories in each fold
        sites_to_split: list
            The list of sites of the entire dataset that will be split into training and testing sets ensuring no overlap in sites in the two sets (i.e., the sites in training set will not appear in testing set)
        models_for_testing: dict
            The dictionary that has the trained StandardScaler, pca and GMM models
        
        Returns
        -------
        predictions_for_all_folds: list
            The predicted labels for all folds
        actual_labels_for_all_folds: list
            The actual labels for all folds i.e., ytest
        test_indices_for_all_folds: list
            The test index for all folds
        h_scores_for_all_folds: list
            The likelihoods of a datapoint belonging to habitat H
        l_scores_for_all_folds: list
            The likelihoods of a datapoint belonging to habitat L
        m_scores_for_all_folds: list
            The likelihoods of a datapoint belonging to habitat M'''
    
    #defining the number of splits we want to make
    sgkf = StratifiedGroupKFold(n_splits = 4)

    fold_no = 0
    predictions_for_all_folds = []
    train_indices_for_all_folds = []
    actual_labels_for_all_folds = []
    h_scores_for_all_folds = []
    l_scores_for_all_folds = []
    m_scores_for_all_folds = []
    #generating the splits 
    for train_index, test_index in sgkf.split(features_to_split,landuse_categories_to_split, groups = sites_to_split):

        print(f'Train and test indices have been obtained for fold {fold_no}')

        xtrain, ytrain = np.take(features_to_split, train_index, axis = 0), np.take(landuse_categories_to_split, train_index)
        #xtest, ytest = np.take(features_to_split, test_index, axis = 0), np.take(landuse_categories_to_split, test_index)

        standardscaler = models_for_testing[f'scaled_data {fold_no}']
        xtrain_scaled = standardscaler.transform(xtrain)
        print(f'The test data has been scaled using the trained StandardScaler model for the fold {fold_no}')

        pca = models_for_testing[f'principal_components {fold_no}']
        x_pca = pca.transform(xtrain_scaled)
        print(f'The test data has been projevcted the the pc space that the model was trained on for the fold {fold_no}')
        x_pca_testing = pd.DataFrame(x_pca)

        gmm_H = models_for_testing[f'GMM_H {fold_no}']
        gmm_L = models_for_testing[f'GMM_L {fold_no}']
        gmm_M = models_for_testing[f'GMM_M {fold_no}']

        h_predictions = gmm_H.score_samples(x_pca_testing)
        l_predictions = gmm_L.score_samples(x_pca_testing)
        m_predictions = gmm_M.score_samples(x_pca_testing)
        print(f'Log-likelihoods have been calculated for the test dataset belonging to GMMs of separate landuse categories for the fold {fold_no}')

        predictions_for_separate_folds, h_scores_for_separate_folds, l_scores_for_separate_folds, m_scores_for_separate_folds = to_predict_labels(h_predictions, l_predictions, m_predictions)
        predictions_for_all_folds.append(predictions_for_separate_folds)
        print(f'Lables of landuse have been assinged for fold {fold_no}')

        train_indices_for_all_folds.append(train_index)
        actual_labels_for_all_folds.append(ytrain)
        h_scores_for_all_folds.append(h_scores_for_separate_folds)
        l_scores_for_all_folds.append(l_scores_for_separate_folds)
        m_scores_for_all_folds.append(m_scores_for_separate_folds)

        fold_no += 1
        
    return predictions_for_all_folds, actual_labels_for_all_folds, train_indices_for_all_folds, h_scores_for_all_folds, l_scores_for_all_folds, m_scores_for_all_folds

def inputs_data (data_pickle, models_training_for_time_interval):
    '''Returns the dictionary that contains trained models for the input data
        
        Parameters
        ----------
        data_pickle: DataFrame
            This is the time separated data for each four hour interval
        models_training_for_time_interval: dict
            The dictionary that contains the trained models for all folds of a particular time interval
        
    
        Returns
        --------
        predicted_labels: list
            The list of predicted labels for all folds of a particular time interval
        actual_labels: list
            The list of actual labels for all folds of a particular time interval
        train_indices: list
            The list of train_indices for all folds of a particular time interval'''

    #creating arras with the n_samples and n_features respectively (basically extracting data that we will run our model on)
    y = np.array(data_pickle['landuse'])
    x = np.array(data_pickle['feats'])
    groups = np.array(data_pickle['sites'])
    #creating lists to run standard scaler and pca functions on
    features_for_training = list(x)
    landuse_category_for_training = list(y)
    sites_for_training = list(groups)

    predicted_labels, actual_labels, train_indices, h_likelihoods, l_likelihoods, m_likelihoods = to_test_model(features_for_training, landuse_category_for_training, sites_for_training, models_training_for_time_interval)

    return predicted_labels, actual_labels, train_indices, h_likelihoods, l_likelihoods, m_likelihoods

#reading data using pandas
data = pd.read_pickle(r"C:\Users\Saranya Sundar\OneDrive\Desktop\Data\output_dir\vggish_evergreen_features_960ms.pkl")

#creating list of two hour time intervals
list_of_time_separated_data = []
for i in range(0, 24, 2):
    if i < 8:
        start_time = f'0{i}0000'
        stop_time = f'0{i+2}0000'
    elif i == 8:
        start_time = f'0{i}0000'
        stop_time = f'{i+2}0000'   
    else:
        start_time = f'{i}0000'
        stop_time = f'{i+2}0000'
    
    list_of_time_separated_data.append(data[data['time'].between(start_time, stop_time)])

#dictionary contains information on the predicted labels, actual lables and scroes of different habitats 
dictionary_for_all_time_intervals = {}
i = 0
for z in list_of_time_separated_data:
    models_training = pd.read_pickle(r"C:\Users\Saranya Sundar\OneDrive\Desktop\Data\Evergreen\960\evergreen_time_separated_2_hours.pickle")
    predicted_labels_time_separated, actual_labels_time_separated, train_indices_time_separated, h, l, m = inputs_data(z, models_training[f'dictionary_for_time_interval {i}'])
    dictionary_for_one_time_interval = {f'dictionary_for_time_interval {i}': {f'predicted_labels_for_time_interval {i}': predicted_labels_time_separated, f'actual_labels_for_time_interval {i}': actual_labels_time_separated, 
                                                                              f'train_indices_for_time_interval {i}': train_indices_time_separated, f'h_scores_for_time_interval {i}': h, f'l_scores_for_time_interval {i}': l, 
                                                                              f'm_scores_for_time_interval {i}': m}}
    dictionary_for_all_time_intervals.update(dictionary_for_one_time_interval)
    i+=1

#saving the threshold
with open(r"C:\Users\Saranya Sundar\OneDrive\Desktop\Data\Evergreen\960\evergreen_two-hours_likelihoods.pickle", 'wb') as handle:
    pickle.dump(dictionary_for_all_time_intervals, handle)