#!/usr/bin/env python3

from argparse import ArgumentParser
from classifier.dataframe import DataFrame
from classifier.decision_tree import dt_fit, dt_predict

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("train", type=str, help="Training file name")
    parser.add_argument("test", type=str, help="Test file name")
    parser.add_argument("output", type=str, help="Output file name")
    args = parser.parse_args()
    file_train = args.train
    file_test = args.test
    file_output = args.output
    df_train = DataFrame(DataFrame.read_csv(file_train))
    df_test = DataFrame(DataFrame.read_csv(file_test))

    tree = dt_fit(df_train)
    predicted = dt_predict(tree, df_test, df_train.columns[-1])
    predicted.save_csv(file_output)
