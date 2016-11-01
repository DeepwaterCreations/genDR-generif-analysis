import re

import pandas as pd
import numpy as np

#Load the GenDR data
gendr = pd.read_csv('data/gendr_manipulations.csv')

#Load the significantly less nicely-formatted geneRIF data
reg = re.compile(r"[\d:,-]+(?![^\t]+$)|[^\t]+$")
datalist = []
with open('data/generifs_basic') as datafile:
    datafile.readline() #The first line is just column headers
    for line in datafile:
        values = reg.findall(line)
        datalist.append(values)
generifs = pd.DataFrame(np.array(datalist), columns=['Tax ID',
                               'Gene ID',
                               'PubMed ID (PMID) list',
                               'last update',
                               'timestamp',
                               'GeneRIF text'])
generifs.drop(['Tax Id', 'PubMed ID (PMID) list', 'last update', 'timestamp'])
generifs['Gene ID'] = generifs['Gene ID'].astype(np.int64)

#Combine the datasets
gendr_rifs = pd.merge(gendr, generifs, left_on="entrez_id", right_on="Gene ID")
corpus = gendr_rifs['GeneRIF text']
