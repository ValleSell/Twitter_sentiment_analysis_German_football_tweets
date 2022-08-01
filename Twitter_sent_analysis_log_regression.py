from helpers_log_regr import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--path', required=True)

args = parser.parse_args()

def main():
    # Annotated Twitter database for sentiment analysis:
    # Open-dataset-for-sentiment-analysis:
    # https://github.com/charlesmalafosse/open-dataset-for-sentiment-analysis/blob/master/betsentiment-DE-tweets-sentiment-teams.zip
    # It is about football and includes less negative tweets, only 4242. I will take the first 4242 positive and all the
    # negative ones.

    #the following code will just work for this particular database
    # create a list of tweet strings with the corresponding sentiment
    twisent = []
    with open(args.path) as twisentALL:
        for line in twisentALL.readlines():
            if line[0] == '"':
                line = line.split(",\"")
                if len(line) > 4:
                    twisent.append([line[2], line[4]])

    # create a balanced data set
    twisent_bal = []

    # take the first 4242 positive and all the negative ones
    n = 0
    #sm is the smaller number, either the number of positive or negative tweets
    sm=4242
    for el in twisent:
        if el[-1] == "POSITIVE\"":
            twisent_bal.append(el)
            n += 1
            if n == sm:
                break
    for el in twisent:
        if el[-1] == "NEGATIVE\"":
            twisent_bal.append(el)

    # now twisent_bal includes an equal number of positive and negative tweets

    # Separate positive and negative
    # create a list with tweets only
    all_positive_tweets = []
    all_negative_tweets = []
    for el in twisent_bal:
        if el[1] == "POSITIVE\"":
            all_positive_tweets.append((el[0])[1:-2])
        elif el[1] == "NEGATIVE\"":
            all_negative_tweets.append((el[0])[1:-2])

    tweets_bal =all_positive_tweets+all_negative_tweets

    # 80% for training and 20% for testing
    train_pos = tweets_bal[:(int(sm*0.8))]
    test_pos = tweets_bal[int(sm*0.8):sm]

    train_neg = tweets_bal[sm:int(sm*1.8)]
    test_neg = tweets_bal[int(sm*1.8):]

    train_x = train_pos + train_neg

    test_x = test_pos + test_neg

    # The corresponding y matrices with the sentiment

    train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)

    test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

    freqs_train = build_freqs(train_x, train_y)

    # Create a matrix to store the info
    X = np.zeros((len(train_x), 3))

    # Iterate over train_x
    for i in range(len(train_x)):
        X[i, :] = extract_features(train_x[i], freqs_train)

    Y = train_y

    n = 3000
    # Apply gradient descent
    J, theta = gradient_descent(X, Y, np.zeros((3, 1)), 1e-9, n)


    # Apply gradient descent again
    J, theta = gradient_descent(X, Y, theta, 1e-9, n * 10)
    print(f"The cost after training is {J:.8f}.")
    print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")

    x = test_logistic_regression(test_x, test_y, freqs_train, theta)

    return x

if __name__ == "__main__":
    main()
