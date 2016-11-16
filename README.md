# genDR-generif-analysis
[GenDR](http://genomics.senescence.info/diet/) is a database of genes thought to be related to the longevity-enhancing effects of dietary restriction. 

[GeneRIFs](https://en.wikipedia.org/wiki/GeneRIF) are short sentences describing the functions of genes.

My goal is to:

1. Build a predictive model to separate DR-relevant geneRIFs from those that are unrelated, which I can then run on a larger set of geneRIFs to find potentially relevant genes that the GenDR database missed.

2. Use unsupervised machine learning techniques on the geneRIFs from the GenDR genes and see if any interesting clusters emerge.

##Usage:
###Python requirements
This project uses the following libraries:
 - Pandas
 - Numpy
 - Sklearn
 - Matplotlib
 - Seaborn
 - Wordcloud
All of these should be available via `pip install`

###Prepare data files
1. Download and extract the files listed in data/data_sources.txt
2. Run src/build_generif_csv.py

###Get predictions
 - Run src/build_model.py to train a Naive Bayes model on the geneRIF text and do a gridsearch to choose parameters. This will probably take a long time to run. When it finishes, it will show you an accuracy score and ROC graph and give you the option to pickle the model into the data/ folder.
 - Run src/run_model.py *data/model_filename.pckl* to run the trained model on the entire dataset and generate a list of predictions. When it finishes, it will give you the option to save the predictions as a .csv, and to save a wordcloud image of the top 100 feature words.
 
###Get categorizations
 - Run src/unsupervised.py to train a Non-negative Matrix Factorization on the GenDR RIFs and output wordclouds showing the top words in each cluster. 
  
src/dataload.py contains helper functions for the other scripts.
