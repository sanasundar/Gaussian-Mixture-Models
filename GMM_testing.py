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
            The list of predicted labels for the features of the test set'''
    
    predictions = []
    for i in range(0, len(h)):
        hsc = 2*h[i] - m[i] - l[i]
        lsc = 2*l[i] - h[i] - m[i]
        msc = 2*m[i] - l[i] - h[i]
        if hsc is max(hsc, lsc, msc):
            predictions.append('H')
        elif lsc is max(hsc, lsc, msc):
            predictions.append('L')
        else:
            predictions.append('M')
    return predictions

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
        dictionary: dict
            Contains the fold number, the predicted labels, actual labels and the test indices'''
    
    #defining the number of splits we want to make
    sgkf = StratifiedGroupKFold(n_splits = 4)

    fold_no = 0
    dictionary = {}
    #generating the splits 
    for train_index, test_index in sgkf.split(features_to_split,landuse_categories_to_split, groups = sites_to_split):

        print(f'Train and test indices have been obtained for fold {fold_no}')

        #xtrain, ytrain = np.take(features_to_split, train_index, axis = 0), np.take(landuse_categories_to_split, train_index)
        xtest, ytest = np.take(features_to_split, test_index, axis = 0), np.take(landuse_categories_to_split, test_index)

        standardscaler = models_for_testing[f'scaled_data {fold_no}']
        xtest_scaled = standardscaler.transform(xtest)
        print(f'The test data has been scaled using the trained StandardScaler model for the fold {fold_no}')

        pca = models_for_testing[f'principal_components {fold_no}']
        x_pca = pca.transform(xtest_scaled)
        print(f'The test data has been projevcted the the pc space that the model was trained on for the fold {fold_no}')
        x_pca_testing = pd.DataFrame(x_pca)

        gmm_H = models_for_testing[f'GMM_H {fold_no}']
        gmm_L = models_for_testing[f'GMM_L {fold_no}']
        gmm_M = models_for_testing[f'GMM_M {fold_no}']

        h_predictions = gmm_H.score_samples(x_pca_testing)
        l_predictions = gmm_L.score_samples(x_pca_testing)
        m_predictions = gmm_M.score_samples(x_pca_testing)
        print(f'Log-likelihoods have been calculated for the test dataset belonging to GMMs of separate landuse categories for the fold {fold_no}')

        predictions_for_separate_folds = to_predict_labels(h_predictions, l_predictions, m_predictions)
        print(f'Lables of landuse have been assinged for fold {fold_no}')
        
        mini_dictionary = {f'fold_no {fold_no}': fold_no, f'predictions_for_fold_no {fold_no}': predictions_for_separate_folds, f'actual_labels_for_fold_no {fold_no}': ytest, f'test_indices_for_fold_no {fold_no}': test_index}

        dictionary.update(mini_dictionary)
        fold_no += 1
    return dictionary

#reading data using pandas
data = pd.read_pickle(r"C:\Users\Saranya Sundar\OneDrive\Desktop\Data\output_dir\vggish_deciduous_features_960ms.pkl")

#creating arrays with the n_samples and n_features and the groups respectively (basically extracting data that we will split and run our model on)
y = np.array(data['landuse'])
x = np.array(data['feats'])
groups = np.array(data['sites'])

#creating lists to run standard scaler and pca functions on 
features = list(x)
landuse_category = list(y)
sites = list(groups)

models_testing = pd.read_pickle(r"C:\Users\Saranya Sundar\OneDrive\Desktop\Data\Deciduous\deciduous")
print(models_testing)
print(type(models_testing))

my_dictionary  = to_test_model(features, landuse_category, sites, models_testing)

with open(r'C:\Users\Saranya Sundar\OneDrive\Desktop\Data\Deciduous\deciduous_predictions.pickle', 'wb') as handle:
    pickle.dump(my_dictionary, handle)

print('done')