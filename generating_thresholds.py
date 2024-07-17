import pandas as pd
import numpy as np
import pickle

def to_separate_true_positives(dictionary):
    '''Returns the dictionary of true positives
    
        Parameters
        ----------
        dictionary: dict
            The dictionary of data that contains the likelihoods of all data points for each landuse category
        
        Returns
        --------
        dictionary_of_true_positives: dict
            The dictionary that contains the likelihoods of the true positives to determine the thresholds'''
    dictionary_of_true_positives = {}
    for z in range(0, len(dictionary)):
        dictionary_for_each_time_interval = dictionary[f'dictionary_for_time_interval {z}']
        predicted_labels_for_all_time_folds = dictionary_for_each_time_interval[f'predicted_labels_for_time_interval {z}']
        actual_labels_for_all_time_folds = dictionary_for_each_time_interval[f'actual_labels_for_time_interval {z}']
        likelihoods_for_all_folds_H = dictionary_for_each_time_interval[f'h_scores_for_time_interval {z}']
        likelihoods_for_all_folds_L = dictionary_for_each_time_interval[f'l_scores_for_time_interval {z}']
        likelihoods_for_all_folds_M = dictionary_for_each_time_interval[f'm_scores_for_time_interval {z}']
        list_for_this_time_interval_h = []
        list_for_this_time_interval_l = []
        list_for_this_time_interval_m = []

        for i in range(0, len(predicted_labels_for_all_time_folds)):
            y_pred = list(predicted_labels_for_all_time_folds[i])
            y_actual = list(actual_labels_for_all_time_folds[i])
            likelihoods_h = list(likelihoods_for_all_folds_H[i])
            likelihoods_l = list(likelihoods_for_all_folds_L[i])
            likelihoods_m = list(likelihoods_for_all_folds_M[i])
            list_for_this_fold_h = []
            list_for_this_fold_l = []
            list_for_this_fold_m = []
            
            for j in range(0, len(y_pred)):
                if y_pred[j] == y_actual[j] == 'H':
                    list_for_this_fold_h.append(likelihoods_h[j])
                if y_pred[j] == y_actual[j] == 'L':
                    list_for_this_fold_l.append(likelihoods_l[j])
                if y_pred[j] == y_actual[j] == 'M':
                    list_for_this_fold_m.append(likelihoods_m[j])
            
            list_for_this_time_interval_h.append(list_for_this_fold_h)
            list_for_this_time_interval_l.append(list_for_this_fold_l)
            list_for_this_time_interval_m.append(list_for_this_fold_m)
        
        mini_dictionary = {f'dictionary_for_time_interval {z}': {f'h_true_positives_for_time_interval {z}': list_for_this_time_interval_h, f'l_true_positives_for_time_interval {z}': list_for_this_time_interval_l, f'm_true_positives_for_time_interval {z}': list_for_this_time_interval_m}}
        dictionary_of_true_positives.update(mini_dictionary)

    return dictionary_of_true_positives

def to_find_thresholds(dictionary_of_tp):
    '''Returns the dictionary that contains the thresholds for each fold for each time interval
        
        Parameters
        -----------
        dictionary_of_tp: dict
            Contains the likelihoods of all the true positives
        
        Returns
        -------
        dictionary_of_thresholds: dict
            The dictionary that contains the thresholds per interval per fold'''
    dictionary_of_thresholds  = {}
    for z in range(0, len(dictionary_of_tp)):
        dictionary_for_each_time_interval = dictionary_of_tp[f'dictionary_for_time_interval {z}']
        h_true_positives_for_each_time_interval = dictionary_for_each_time_interval[f'h_true_positives_for_time_interval {z}']
        l_true_positives_for_each_time_interval = dictionary_for_each_time_interval[f'h_true_positives_for_time_interval {z}']
        m_true_positives_for_each_time_interval = dictionary_for_each_time_interval[f'm_true_positives_for_time_interval {z}']

        list_of_thresholds_for_all_folds_h = []
        list_of_thresholds_for_all_folds_l = []
        list_of_thresholds_for_all_folds_m = []
        for i in range(0, len(h_true_positives_for_each_time_interval)):
            h_true_positives_for_this_fold = h_true_positives_for_each_time_interval[i]
            l_true_positives_for_this_fold = l_true_positives_for_each_time_interval[i]
            m_true_positives_for_this_fold = m_true_positives_for_each_time_interval[i]
             
            h_threshold_for_this_fold = np.percentile(h_true_positives_for_this_fold, 43)
            l_threshold_for_this_fold = np.percentile(l_true_positives_for_this_fold, 43)
            m_threshold_for_this_fold = np.percentile(m_true_positives_for_this_fold, 43)

            list_of_thresholds_for_all_folds_h.append(h_threshold_for_this_fold)
            list_of_thresholds_for_all_folds_l.append(l_threshold_for_this_fold)
            list_of_thresholds_for_all_folds_m.append(m_threshold_for_this_fold)
            
        mini_dictionary = {f'dictionary_for_time_interval {z}': {f'h_thresholds_for_time_interval {z}': list_of_thresholds_for_all_folds_h, f'l_thresholds_for_time_interval {z}': list_of_thresholds_for_all_folds_l, 
                                                                 f'm_thresholds_for_time_interval {z}': list_of_thresholds_for_all_folds_m}}
        dictionary_of_thresholds.update(mini_dictionary)
    return dictionary_of_thresholds

dictionary_for_all_time_intervals = pd.read_pickle(r"C:\Users\Saranya Sundar\OneDrive\Desktop\Data\Evergreen\960\evergreen_two-hours_likelihoods.pickle")

dictionary_of_true_positives_for_all_time_intervals = to_separate_true_positives(dictionary_for_all_time_intervals)
dictionary_of_thresholds_for_all_time_intervals = to_find_thresholds(dictionary_of_true_positives_for_all_time_intervals)

print(dictionary_of_thresholds_for_all_time_intervals)

with open(r"C:\Users\Saranya Sundar\OneDrive\Desktop\Data\Evergreen\960\evergreen_two-hours_thresholds.pickle", 'wb') as handle:
    pickle.dump(dictionary_of_thresholds_for_all_time_intervals, handle)