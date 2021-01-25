#----------
# Import Libraries
#----------

# General libraries
import argparse
import os
import joblib
import numpy as np
import pandas as pd

# sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import accuracy_score

# Azure ML libraries
from azureml.core.run import Run





#----------
# Code
#----------

def preprocess (ds):

    # Transformation of binary data
    ds["Gender"] = ds.Gender.apply(lambda s: 1 if s == "Female" else 0)
    ds["family_history_with_overweight"] = ds.family_history_with_overweight.apply(lambda s: 1 if s == "yes" else 0)
    ds["FAVC"] = ds.FAVC.apply(lambda s: 1 if s == "yes" else 0)
    ds["SMOKE"] = ds.SMOKE.apply(lambda s: 1 if s == "yes" else 0)
    ds["SCC"] = ds.SCC.apply(lambda s: 1 if s == "yes" else 0)

    # One hot encoding for categorical data
    CAEC_list = pd.get_dummies(ds.CAEC, prefix="CAEC")
    ds.drop("CAEC", inplace=True, axis=1)
    ds = ds.join(CAEC_list)

    CALC_list = pd.get_dummies(ds.CALC, prefix="CALC")
    ds.drop("CALC", inplace=True, axis=1)
    ds = ds.join(CALC_list)

    MTRANS_list = pd.get_dummies(ds.MTRANS, prefix="MTRANS")
    ds.drop("MTRANS", inplace=True, axis=1)
    ds = ds.join(MTRANS_list)

    # Transformation of target feature through a dictionary
    obesity = {"Insufficient_Weight":1, "Normal_Weight":2, "Overweight_Level_I":3, "Overweight_Level_II":4, "Obesity_Type_I":5, "Obesity_Type_II":6, "Obesity_Type_III":7}
    ds["NObeyesdad"] = ds.NObeyesdad.map(obesity)
    
    # Obtain features and target datasets
    x_ds = ds.drop('NObeyesdad',axis=1)
    y_ds = ds['NObeyesdad']
    return (x_ds, y_ds)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_depth', type=int, default=None, help="Maximum depth of the tree")
    parser.add_argument('--min_samples_split', type=int, default=2, help='Minimum number of samples required to split an internal node')

    args = parser.parse_args()

    run.log("Max depth:", np.int(args.max_depth))
    run.log("Minimum samples split:", np.int(args.min_samples_split))

    model = tree.DecisionTreeClassifier(max_depth=args.max_depth, min_samples_split=args.min_samples_split, random_state = 0).fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score (y_test, y_pred)
    run.log("accuracy", np.float(accuracy))

    os.makedirs('./outputs/model_h', exist_ok=True)
    joblib.dump(model, './outputs/model_h/model_h.joblib')



if __name__ == '__main__':
    
    # Define data file
    file ='ObesityDataSet_raw_and_data_sinthetic.csv'

    # Load dataset to pandas dataframe
    df = pd.read_csv(file)

    # Preprocess dataset
    x, y = preprocess (df)
    
    # Split data into train and test sets.
    # Split into 70-30 proportion, since is the general recommended value in the field
    # Set random_state to 0, to ensure that the same random combination is used between runs
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

    # Standard scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Retrieve the current service context for logging metrics and uploading files
    run = Run.get_context()

        
    main()