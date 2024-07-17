#importing all necessary modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
import pickle

def to_perform_standard_scaler(data_to_scale):
    '''Returns the scaled data and the trained model 
        
        Parameters
        ----------
        data_to_scale: list
            the list of features that need to be scaled
        
        Returns
        --------
        data_to_scale: list
            The list of scaled data
        stdscaler: sklearn.preprocessing._data.StandardScaler
            The model that has been trained on the data which on which we will project our test data'''

    stdscaler = StandardScaler()
    data_to_scale = stdscaler. fit_transform(data_to_scale)
    return data_to_scale, stdscaler

def to_perform_pca(scaled_data):
    '''Returns the Principal Components that explain 95% of the variance in the data, the list of all PC components and the trained model
        
        Parameters
        ----------
        scaled_data: ndarray
            The scaled data returned by the StandardScaler function
            
        Returns
        --------
        principal_components: ndarray
            The Principal Components that explain 95% of the variation in the scaled_data
        pca: sklearn.decomposition._pca.PCA
            The model that has been trained on the scaled data on which we will project our scaled test data'''
    
    pca = PCA(n_components = 0.95)
    principal_components = pca.fit_transform(scaled_data)
    return principal_components, pca

def to_separate_data(x_pca, landuse_dataframe):
    '''Returns the three separate dataframes of different landuse categories on which we will train our three landuse GMMs
        
        Parameters
        ----------
        x_pca: ndarray
            The Principal Components returned by the to_perform_pca function
        landuse_dataframe: DataFrame
            The dataframe that stores the training landuse categories
            
        Returns
        --------
        finalDf_H: DataFrame
            The dataframe that contains the scaled and principal components of landuse category H
        finalDf_L: DataFrame
            The dataframe that contains the scaled and principal componenrs of landuse category L
        finalDf_M: DataFrame
            The dataframe that contains the scaled and principal componenrs of landuse category M'''
    
    x_pca_training = pd.DataFrame(x_pca)
    finalDf_training = pd.concat([x_pca_training, landuse_dataframe['landuse']], axis = 1)  
    finalDf_H = finalDf_training[finalDf_training['landuse'] == 'H']
    finalDf_L = finalDf_training[finalDf_training['landuse'] == 'L']
    finalDf_M = finalDf_training[finalDf_training['landuse'] == 'M']
    return finalDf_H, finalDf_L, finalDf_M

def to_train_model(features_to_split, landuse_categories_to_split, sites_to_split):
    '''Returns the dictionary that contains information on fold number, the trained StandardScaler, pca and gmms of the three landuse categories for that fold
    
        Parameters
        ----------
        features_to_split: list
            The list of features of the entire dataset that will be split into training and testing sets
        landuse_categories_to_split: list
            The list of the landuse catergories of the entire dataset that will be split into training and testing sets maintaining proportion of landuse categories in each fold
        sites_to_split: list
            The list of sites of the entire dataset that will be split into training and testing sets ensuring no overlap in sites in the two sets (i.e., the sites in training set will not appear in testing set)
            
        Returns
        --------
        dictionary: dict
            The dictionary that contains information on the fold number, the trained StandardScaler model, the trained pca model and the three separately trained landuse GMMs'''
    
    #defining the number of splits we want to make
    sgkf = StratifiedGroupKFold(n_splits = 4)
    dictionary = {}
    fold_no = 0

    #generating the splits in the dataset
    for train_index, test_index in sgkf.split(features_to_split, landuse_categories_to_split, groups = sites_to_split):

        print(f'Indices have been split into training and testing sets for fold {fold_no}')

        xtrain, ytrain = np.take(features_to_split, train_index, axis = 0), np.take(landuse_categories_to_split, train_index)
        #xtest, ytest = np.take(features_to_split, test_index, axis = 0), np.take(landuse_categories_to_split, test_index)
        landuse_dataframe_training = pd.DataFrame(ytrain, columns= ['landuse'])

        xtrain_scaled, stdscaler_train = to_perform_standard_scaler(xtrain)
        print(f'Standard Scaler has been performed for fold {fold_no}')

        principal_components_for_training, pca_train = to_perform_pca(xtrain_scaled)
        print(f'Principal Component Analysis has been performed for fold {fold_no}')

        finalDf_training_H, finalDf_training_L, finalDf_training_M = to_separate_data(principal_components_for_training, landuse_dataframe_training)
        print(f'Data has been separated into three landuse categories for fold {fold_no}')
        
        #training the GMM on separate landuse categories to estimate density
        gmm_H = GaussianMixture(n_components = 100, random_state= 42)
        gmm_L = GaussianMixture(n_components = 100, random_state= 42)
        gmm_M = GaussianMixture(n_components = 100, random_state= 42)

        gmm_H.fit(finalDf_training_H.loc[:, finalDf_training_H.columns != 'landuse'], finalDf_training_H.iloc[:, -1:])
        gmm_L.fit(finalDf_training_L.loc[:, finalDf_training_L.columns != 'landuse'], finalDf_training_L.iloc[:, -1:])
        gmm_M.fit(finalDf_training_M.loc[:, finalDf_training_M.columns != 'landuse'], finalDf_training_M.iloc[:, -1:])

        print(f'Three separate GMMs have been trained for fold {fold_no}')

        mini_dictionary = {f'fold_no {fold_no}': fold_no, f'scaled_data {fold_no}': stdscaler_train , f'principal_components {fold_no}': pca_train, 
                           f'GMM_H {fold_no}': gmm_H, f'GMM_L {fold_no}': gmm_L, f'GMM_M {fold_no}': gmm_M}

        dictionary.update(mini_dictionary)
        fold_no += 1
    return dictionary

def inputs_data (data_pickle):
    '''Returns the dictionary that contains trained models for the input data
        
        Parameters
        ----------
        data_pickle: DataFrame
            This is the time separated data for each four hour interval
    
        Returns
        --------
        my_dictionary: dict
            The dictionary contains the trained models for the time interval that was inputted'''

    #creating arras with the n_samples and n_features respectively (basically extracting data that we will run our model on)
    y = np.array(data_pickle['landuse'])
    x = np.array(data_pickle['feats'])
    groups = np.array(data_pickle['sites'])
    #creating lists to run standard scaler and pca functions on
    features_for_training = list(x)
    landuse_category_for_training = list(y)
    sites_for_training = list(groups)

    my_dictionary = to_train_model(features_for_training, landuse_category_for_training, sites_for_training)
    print(my_dictionary)

    return my_dictionary

#reading data using pandas
data = pd.read_pickle(r"C:\Users\Saranya Sundar\OneDrive\Desktop\Data\output_dir\vggish_evergreen_features_960ms.pkl")

list_of_time_separated_data = []
for i in range(0, 24, 1):
    if i < 9:
        start_time = f'0{i}0000'
        stop_time = f'0{i}3000' 
        start_time_1 = f'0{i}3000'
        stop_time_1 = f'0{i+1}0000' 
    elif i == 9:
        start_time = f'0{i}0000'
        stop_time = f'0{i}3000'
        start_time_1 = f'0{i}3000'
        stop_time_1 = f'{i+1}0000'
    else:
        start_time = f'{i}0000'
        stop_time = f'{i}3000'
        start_time_1 = f'{i}3000'
        stop_time_1 = f'{i+1}0000'
    print(data[data['time'].between(start_time, stop_time)])
    print(data[data['time'].between(start_time_1, stop_time_1)])
    list_of_time_separated_data.append(data[data['time'].between(start_time, stop_time)])
    list_of_time_separated_data.append(data[data['time'].between(start_time_1, stop_time_1)])

dictionary_for_all_time_intervals = {}
i = 0
for z in list_of_time_separated_data:
    small_dictionary = inputs_data(z)
    dictionary_for_one_time_interval = {f'dictionary_for_time_interval {i}': small_dictionary}
    dictionary_for_all_time_intervals.update(dictionary_for_one_time_interval)
    i+=1

print(dictionary_for_all_time_intervals)

with open(r'C:\Users\Saranya Sundar\OneDrive\Desktop\Data\Evergreen\960\evergreen_time_separated_half_hours.pickle', 'wb') as handle:
    pickle.dump(dictionary_for_all_time_intervals, handle)