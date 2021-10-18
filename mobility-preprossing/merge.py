import os
import glob
import pandas as pd


def merge(dataset):
    df = pd.concat(map(pd.read_csv, dataset), axis=1)
    del df['timestamp']
    for index, col in enumerate(df.columns):
        df.columns.values[index] = index

    df.to_csv('users.txt', header=False)


if __name__ == '__main__':
    path = os.getcwd() + '/cleaned_dataset'
    files = sorted(glob.glob(path + "/*.txt"))
    merge(files)
