#!/usr/bin/env python3

from optparse import OptionParser
from csv import reader
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
from importlib import import_module
from utils.association import association_rule
import time

if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input_file',
                         help='CSV filename',
                         default=None)
    optparser.add_option('-o', '--outputFile',
                         dest='output_file',
                         help='log filename',
                         default='output.txt')
    optparser.add_option('-d', '--delimiter',
                         dest='delimiter',
                         help='delimiter of log',
                         default='\t')
    optparser.add_option('-s', '--minSupport',
                         dest='min_support',
                         help='Minimum support (float)',
                         default=0.5,
                         type='float')
    optparser.add_option('-c', '--minConfidence',
                         dest='min_confidence',
                         help='Minimum confidence (float)',
                         default=0.0,
                         type='float')
    optparser.add_option('-m', '--module',
                         dest='module',
                         help='Miner module',
                         default='apriori')

    (options, args) = optparser.parse_args()

    file = options.input_file
    log = 'result/' + options.output_file
    min_support = options.min_support
    min_confidence = options.min_confidence
    module = options.module
    delim = options.delimiter

    print('---------------------------')
    print('[Frequent Pattern Mining]')
    print(f' - input file     : {file}')
    print(f' - minimum support: {min_support}')
    print(f' - miner moudle   : {module}')
    print('---------------------------')

    dataset = []
    delimiter = ',' if file.split('.')[-1] == 'csv' else '\t'
    with open(file, "r") as f:
        for line in reader(f, delimiter=delimiter):
            dataset.append([x for x in line if x])  # ignore empty

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    miner = import_module('.' + module, 'miner')

    # Frequent patterns
    print()
    print('Mining...  ', end='')
    start = time.time()
    patterns = miner.mine(df, min_support)
    end = time.time()
    print('Finish')
    print(f' - Runtime: {end - start:.3f}s')

    # Association rules
    print()
    print('Logging...  ', end='')
    start = time.time()
    with open(log, "w") as f:
        for l, r, sup, conf in association_rule(patterns, min_support, min_confidence):
            l_str = f"{{{','.join(l)}}}"
            r_str = f"{{{','.join(r)}}}"
            f.write(f"{l_str}{delim}{r_str}{delim}{sup:.2f}{delim}{conf:.2f}\n")
    end = time.time()
    print('Finish')
    print(f' - Runtime: {end - start:.3f}s')

