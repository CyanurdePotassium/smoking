import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# loading the processed data
data_processed = pd.read_csv("../../data/processed/smoking_processed.csv")

# choosing y and X columns
y_col = 'Yes'
non_dependent_variables = ['Yes', 'amt_weekdays', 'amt_weekends', 'Both/Mainly Hand-Rolled', 'Both/Mainly Packets', 'Hand-Rolled', 'Packets', 'nan']

X_cols = filter(lambda x : x not in non_dependent_variables, data_processed.columns)

# splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data_processed[X_cols], data_processed[y_col], test_size = 0.2, stratify = None)

# creating the model
model = RandomForestClassifier(
    n_estimators = 100,
    criterion = 'gini',
    max_depth = None, # exp?
    min_samples_split = 2, # exp?
    min_samples_leaf = 1, # exp?
    min_weight_fraction_leaf = 0.0, # exp?
    max_features = 'sqrt', # exp?
    max_leaf_nodes = None, # exp?
    min_impurity_decrease = 0.0, # exp?
    bootstrap = True,
    oob_score = False,
    n_jobs = None,
    random_state = 1,
    verbose = 1,
    warm_start = False, # exp?
    class_weight = None,
    ccp_alpha = 0.0, # exp?
    max_samples = None # (0.2) exp?
)

# fitting the model
model.fit(X_train, y_train)

# print model accuracy
print('Mean accuracy of the model: ', model.score(X_test, y_test))

# saving the model
dump(model, '../../models/RFC.joblib')