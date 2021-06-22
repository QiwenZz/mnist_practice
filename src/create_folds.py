import pandas as pd
from sklearn import model_selection

if __name__== "main":
    train = pd.read_csv('../input/mnist_train.csv')
    test = pd.read_csv('../input/mnist_test.csv')

    df = pd.concat([train,test])
    df = df.reset_index(drop = True)
    df['kfold'] = -1
    kf = model_selection.KFold(n_splits = 5)
    df = df.sample(frac=1).reset_index(drop=True)

    for fold, y in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold


    df.to_csv('../input/mnist_folds.csv')