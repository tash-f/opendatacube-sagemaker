import os
import joblib
import argparse
import pandas as pd
from sklearn import tree
import numpy as np



if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--criterion', type=str, default='gini')
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--splitter', type=str, default='random')
    parser.add_argument('--min_samples_split', type=int, default=100)
    #parser.add.argument('--model_col_indices',nargs='+', type=int)


    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args, _ = parser.parse_known_args()

    # ... load from args.train and args.test, train a model, write model to args.model_dir.
    
    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    train_data = pd.concat(raw_data)
    train_data_arr = np.asarray(train_data)

    
    classifier_params = {'criterion': args.criterion,
                     'max_depth' : args.max_depth,
                     'splitter' : args.splitter,
                     'min_samples_split' : args.min_samples_split}

    model = tree.DecisionTreeClassifier(**classifier_params, random_state=1)
    m = model.fit(train_data_arr[:,1:], train_data_arr[:, 0])
    
    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(m, os.path.join(args.model_dir, "model.joblib"))
    
def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf