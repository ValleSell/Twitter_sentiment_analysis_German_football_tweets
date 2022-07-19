import re
import string
import numpy as np
import pandas as pd
import nltk
import snscrape.modules.twitter as sntwitter #to scrape tweets

from nltk.stem.cistem import Cistem  # for German stemming
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
tt = TweetTokenizer()


german_stop_words = stopwords.words('german')
nltk.download('punkt')


def get_tweets(query, nr):
    # this function takes two parameters, a query and a number, and returns a dataframe containing that number
    # of tweets scraped for the query.
    # The query must be a string in the following format:
    # "KEYWORDS lang:LANG_CODE until:YYYY-DD-MM since:YYYY-DD-MM"
    # Example :query = "Digitalisierung lang:de until:2022-07-03 since:2010-01-01"
    # The number is the desired number of tweets to scrape
    tweets = []

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        tweets.append([tweet.date, tweet.user.username, tweet.content])
        if len(tweets) == nr:
            break
    df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])

    return df


def process_tweet_w_CISTEM (text):
    # this function takes a string and removes RT, hyperlinks, @usernames and punctuation (but not smileys),
    # then the text is converted into a list of tokens and finally stems. returns a list of stems

    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)

    # remove hyperlinks
    text = re.sub(r'https?://[^\s\n\r]+', '', text)

    # remove numbers
    text = re.sub(r'[0-9]', '', text)

    # remove usernames
    text = re.sub('@[^\s]+', '', text)

    # remove hashtags (just the hash # sign from the word, not the word)
    text = re.sub(r'#', '', text)

    # converting a tweet (string) of the list in a list of tokens
    # use the tweet_tokenizer because it recognizes emojis
    tweettokens = tt.tokenize(text)

    # filter stop words and punctuation
    clean_tweet = []
    for token in tweettokens:
        if (token not in german_stop_words and token not in string.punctuation):  ## remove stopwords and punctuation
            clean_tweet.append(token)

    # Use CISTEM for stemming German texts
    stemmer = Cistem()

    stem_tweet = []
    for word in clean_tweet:
        stem_word = stemmer.stem(word)  # stemming word
        stem_tweet.append(stem_word)

    return stem_tweet


def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    yslist = np.squeeze(ys).tolist()

    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet_w_CISTEM(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs


def extract_features(tweet, freqs, process_tweet=process_tweet_w_CISTEM):
    '''
    This output of this function will be used to train the model. The default preprocessing is done with CISTEM for stemming,
    indicate if other
    Input:
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet_w_CISTEM tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)

    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3))

    # bias term is set to 1
    x[0, 0] = 1

    # loop through each word in the list of words
    for word in word_l:
        if (word, 1.0) in freqs:
            # increment the word count for the positive label 1
            x[0, 1] += freqs[(word, 1.0)]

        if (word, 0.0) in freqs:
            # increment the word count for the negative label 0
            x[0, 2] += freqs[(word, 0.0)]
        else:
            continue

    assert (x.shape == (1, 3))
    return x


def sigmoid(z):
    '''
    Input
        can be a scalar or an array
    Output
        the sigmoid of z
    '''
    h = 1 / (1 + np.exp(-z))

    return h


def gradient_descent (x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features, shape (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to do to train your model
    Output:
        J: the final cost
        theta: your final weight vector
    '''
    m = x.shape[0]

    for i in range(0, num_iters):
        # get z, the dot product of x and theta
        z = np.dot(x, theta)

        # get the sigmoid of z
        h = sigmoid(z)

        # calculate the cost function
        J = -1. / m * (np.dot(y.transpose(), np.log(h)) + np.dot((1 - y).transpose(), np.log(1 - h)))

        # update the weights theta
        theta = theta - alpha / m * np.dot(x.transpose(), (h - y))

    J = float(J)
    return J, theta


# the following function predicts the sentiment of a tweet based on the freq dictio and the weights calculated above

def predict_tweet_sigmoid(tweet, freqs, theta):
    '''
    Input:
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output:
        y_pred: the probability of a tweet being positive or negative
    '''

    # extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x, theta))

    return y_pred


# compare the predicted and the annotated sentiments

def test_logistic_regression(test_x, test_y, freqs, theta, predictions=predict_tweet_sigmoid):
    """
    Input:
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output:
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """

    # list for storing predictions
    y_hat = []

    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predictions(tweet, freqs, theta)

        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            # append 0 to the list
            y_hat.append(0.0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    y_hat = np.asarray(y_hat)
    test_y = np.squeeze(test_y)
    correct_predicts = np.float64(0)

    for el in zip(y_hat, test_y):
        if el[0] == el[1]:
            correct_predicts += 1
    m = test_y.shape[0]
    accuracy = correct_predicts / m

    return accuracy