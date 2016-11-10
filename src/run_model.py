import cPickle

import dataload
from build_model import MODEL_FILEPATH

PREDICTIONS_FILEPATH = "data/predictions.csv"

if __name__ == "__main__":
    with open(MODEL_FILEPATH) as model_file:
        model = cPickle.load(model_file)
    df = dataload.get_labeled_df()
    df = df[df['in_genDR'] == 0]
    
    corpus = df["GeneRIF text"]

    predictions = model.predict(corpus)

    df['dr_relevant'] = predictions

    df.to_csv(PREDICTIONS_FILEPATH)
