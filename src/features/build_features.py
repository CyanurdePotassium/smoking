import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# loading the raw data
data_raw = pd.read_csv("../../data/raw/smoking.csv")

# creating a new dataframe
data_processed = pd.DataFrame()

# copying ages
data_processed["age"] = data_raw["age"]

# categorizing qualifications
data_processed["highest_qualification"] = (data_raw["highest_qualification"].replace({
    "No Qualification" : 0, "GCSE/O Level" : 1, "GCSE/CSE" : 1, "A Levels" : 2, 
    "Other/Sub Degree" : 3, "Higher/Sub Degree" : 4, "ONC/BTEC" : 5, "Degree" : 6
}))

# categorizing income groups
data_processed["gross_income"] = (data_raw["gross_income"].replace({
    "Above 36,400" : 8, "28,600 to 36,400" : 7, "20,800 to 28,600" : 6, 
    "15,600 to 20,800" : 5, "10,400 to 15,600" : 4, "5,200 to 10,400" : 3,
    "2,600 to 5,200" : 2, "Under 2,600" : 1, "Refused" : 0, "Unknown" : 0
}))

# copying amounts of cigarettes smoked with NA counted as 0
data_processed[["amt_weekdays", "amt_weekends"]] = data_raw[["amt_weekdays", "amt_weekends"]].fillna(0)

# normalizing data
data_processed = round(
    (data_processed - data_processed.min()) / (data_processed.max() - data_processed.min()),
    ndigits = 4)

# preparing columns for one-hot encoding
oh_col_names = ["smoke", "gender", "marital_status", "nationality", "ethnicity", "region", "type"]

# preparing one-hot encoder
oh_encoder = OneHotEncoder(sparse_output = False, drop = 'if_binary', dtype = np.int64, 
                           feature_name_combiner = lambda input_feature, category : str(category))

# one-hot encoding columns, resetting their indexes and renaming them accordingly
oh_columns = pd.DataFrame(oh_encoder.fit_transform(data_raw[oh_col_names]))
oh_columns.index = data_raw.index
oh_columns.columns = oh_encoder.get_feature_names_out()

# concatenating the one-hot encoded columns to the processed data
data_processed = pd.concat([data_processed, oh_columns], axis = 1)

# saving the processed data
data_processed.to_csv("../../data/processed/smoking_processed.csv")