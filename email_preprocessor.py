'''email_preprocessor.py
Preprocess Enron email dataset into features for use in supervised learning algorithms
YOUR NAME HERE
CS 251/2 Data Analysis Visualization
Spring 2023
'''
import re
import os
import numpy as np


def tokenize_words(text):
    '''Transforms an email into a list of words.

    Parameters:
    -----------
    text: str. Sentence of text.

    Returns:
    -----------
    Python list of str. Words in the sentence `text`.

    This method is pre-filled for you (shouldn't require modification).
    '''
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


def count_words(email_path='data/enron'):
    '''Determine the count of each word in the entire dataset (across all emails)

    Parameters:
    -----------
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_emails: int. Total number of emails in the dataset.

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use the `tokenize_words` function above to chunk it into a list of words.
    - Update the counts of each word in the dictionary.

    Hints:
    - Check out Python functions in the os and os.path modules for walking the directory structure.
    '''
    # pass
    word_freq = {}
    i = 0
    email_count = 0
    for folder in os.listdir(email_path):
        # print(folder)
        for filename in os.listdir(os.path.join(email_path, folder)):
            # print(filename)
            f = os.path.join(os.path.join(email_path, folder), filename)
            if os.path.isfile(f):
                email_count += 1 
                # print(f)
                text_file = open(f, "r")
    
        #read whole file to a string
                data = text_file.read()
                list = tokenize_words(data)
                for datum in list:
                    if datum in word_freq:
                        word_freq[datum] += 1
                    else:
                        word_freq[datum] = 1
                # if i < 2:
                #     print(list)

                
        #close file
                text_file.close()
                # i+=1
    return word_freq, email_count


def find_top_words(word_freq, num_features=200):
    '''Given the dictionary of the words that appear in the dataset and their respective counts,
    compile a list of the top `num_features` words and their respective counts.

    Parameters:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_features: int. Number of top words to select.

    Returns:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    counts: Python list. Counts of the `num_features` words in high-to-low count order.
    '''
    pass
    key_list = []
    value_list = []
    dictionary = sorted(word_freq, key = word_freq.get, reverse=True)
    # print(dictionary)
    i = 0
    for key in dictionary:
        if i < num_features:
            key_list.append(key)
            value_list.append(word_freq[key])
            i += 1
        else:
            break
        
    return key_list, value_list

def modified_find_top_words(word_freq, num_features=200, delete_num = 5):
    '''Given the dictionary of the words that appear in the dataset and their respective counts,
    compile a list of the top `num_features` words and their respective counts.

    Parameters:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_features: int. Number of top words to select.

    Returns:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    counts: Python list. Counts of the `num_features` words in high-to-low count order.
    '''
    # pass
    common_words = ['the','to','and','of','a']
    
    key_list = []
    value_list = []
    dictionary = sorted(word_freq, key = word_freq.get, reverse=True)
    for i in range(delete_num):
        del dictionary[0]

    # print(dictionary)
    i = 0
    for key in dictionary:
        if i < num_features:
            key_list.append(key)
            value_list.append(word_freq[key])
            i += 1
        else:
            break
        
    return key_list, value_list

def make_feature_vectors(top_words, num_emails, email_path='data/enron'):
    '''Count the occurance of the top W (`num_features`) words in each individual email, turn into
    a feature vector of counts.

    Parameters:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    num_emails: int. Total number of emails in the dataset.
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    feats. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use `tokenize_words` to chunk it into a list of words.
    - Count the occurance of each word, ONLY THOSE THAT APPEAR IN `top_words`.

    HINTS:
    - Start with your code in `count_words` and modify as needed.
    '''
    # pass
    feature_array = np.zeros((num_emails, len(top_words)))
    classes_vector = []
    # print(feature_array.shape)
    i = 0
    for folder in os.listdir(email_path):
            # print(folder)
            for filename in os.listdir(os.path.join(email_path, folder)):
                # print(filename)
                f = os.path.join(os.path.join(email_path, folder), filename)
                if os.path.isfile(f):
                    # i += 1
                    # email_count += 1 
                    # print(f)
                    text_file = open(f, "r")
        
            #read whole file to a string
                    data = text_file.read()
                    list = tokenize_words(data)
                    word_freq = {}
                    for datum in list:
                        if datum in word_freq:
                            word_freq[datum] += 1
                        else:
                            word_freq[datum] = 1

                    for j in range (len(top_words)):
                        try:
                            value = word_freq[top_words[j]]
                        except:
                            value = 0 
                        feature_array[i, j] = value
                    i += 1
                    if (folder == 'ham'):
                        classes_vector.append(1)
                    elif (folder == 'spam'):
                        classes_vector.append(0)
                    text_file.close()
    classes_vector = np.array(classes_vector)
    # print(classes_vector)
    return feature_array, classes_vector

def make_train_test_sets(features, y, test_prop=0.2, shuffle=True):
    '''Divide up the dataset `features` into subsets ("splits") for training and testing. The size
    of each split is determined by `test_prop`.

    Parameters:
    -----------
    features. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)
    test_prop: float. Value between 0 and 1. What proportion of the dataset samples should we use
        for the test set? e.g. 0.2 means 20% of samples are used for the test set, the remaining
        80% are used in training.
    shuffle: boolean. Should we shuffle the data before splitting it into train/test sets?

    Returns:
    -----------
    x_train: ndarray. shape=(num_train_samps, num_features).
        Training dataset
    y_train: ndarray. shape=(num_train_samps,).
        Class values for the training set
    inds_train: ndarray. shape=(num_train_samps,).
        The index of each training set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    x_test: ndarray. shape=(num_test_samps, num_features).
        Test dataset
    y_test:ndarray. shape=(num_test_samps,).
        Class values for the test set
    inds_test: ndarray. shape=(num_test_samps,).
        The index of each test set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    '''
    total_samples = features.shape[0]
    number_test_samples = int(total_samples*test_prop)
    number_train_samples = total_samples - number_test_samples
    # print(number_train_samples)
    # print(number_test_samples)
    inds = np.arange(y.size)
    if shuffle:
        features = features.copy()
        y = y.copy()

        inds = np.arange(y.size)
        np.random.shuffle(inds)
        # print(inds)
        features = features[inds]
        # print(features)
        y = y[inds]
        # print(y)
    x_train = features[0:number_train_samples,:]
    y_train = y[0:number_train_samples]
    inds_train = inds[0:number_train_samples]
    x_test = features[number_train_samples:total_samples,:]
    y_test = y[number_train_samples:total_samples]
    inds_test = inds[number_train_samples:total_samples]
    # print(x_train.shape,y_train.shape)
    return x_train, y_train, inds_train, x_test, y_test, inds_test

    # Your code here:


def retrieve_emails(inds, email_path='data/enron'):
    '''Obtain the text of emails at the indices `inds` in the dataset.

    Parameters:
    -----------
    inds: ndarray of nonnegative ints. shape=(num_inds,).
        The number of ints is user-selected and indices are counted from 0 to num_emails-1
        (counting does NOT reset when switching to emails of another class).
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    Python list of str. len = num_inds = len(inds).
        Strings of entire raw emails at the indices in `inds`
    '''
    pass
