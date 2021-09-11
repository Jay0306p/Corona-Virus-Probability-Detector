import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

def data_split(data,ratio):
    np.random.seed(42) #to fix the shuffled values so that the rest test data is not matched with any of the shuffled data.
    shuffled = np.random.permutation(len(data)) #it shuffles the length of data in random permutation.
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == "__main__":
#Read the data
    df = pd.read_csv('data.csv')
    train,test = data_split(df, 0.2)
    X_train = train[['Fever','BodyPain','Age','RunnyNose','DiffBreath']].to_numpy()
    X_test = test[['Fever','BodyPain','Age','RunnyNose','DiffBreath']].to_numpy()
   
    Y_train = train[['InfectionProb']].to_numpy().reshape(4000,)
    Y_test = test[['InfectionProb']].to_numpy().reshape(1000,)

    clf = LogisticRegression()
    clf.fit(X_train, Y_train) ## fitting the training data

    # Open a file , where you ant to store the data
    file = open('model.pkl','wb')

    #dump information to that file
    pickle.dump(clf, file)
    file.close()


    