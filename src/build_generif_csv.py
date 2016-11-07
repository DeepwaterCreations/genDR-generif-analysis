import re

import pandas as pd
import numpy as np

def load_gendr():
    #Load the GenDR data
    gendr = pd.read_csv('data/gendr_manipulations.csv')
    return gendr

def load_generifs():
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
    generifs = generifs.drop(['Tax ID', 'PubMed ID (PMID) list', 'last update', 'timestamp'], axis=1)
    generifs['Gene ID'] = generifs['Gene ID'].astype(np.int64)
    return generifs

def match_gendr_to_rifs(gendr, generifs):
    #Combine the datasets
    gendr_rifs = pd.merge(gendr, generifs,
                        how="right",
                        left_on="entrez_id",
                        right_on="Gene ID",
                        indicator="in_genDR")
    gendr_rifs = gendr_rifs.drop(['entrez_id'], axis=1)
    gendr_rifs['in_genDR'] = map(lambda x: 1 if x == "both" else 0, gendr_rifs['in_genDR'])
    return gendr_rifs

if __name__ == "__main__":
    gendr = load_gendr()
    generifs = load_generifs()
    gendr_rifs = match_gendr_to_rifs(gendr, generifs)

    #Export to csv
    gendr_rifs.to_csv('data/gendr_rifs.csv')

