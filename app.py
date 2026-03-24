#set imports needed
import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, BooleanField, CharField, TextField, IntegerField, FloatField, IntegrityError, PostgresqlDatabase, SqliteDatabase
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
import logging

### add function for pipeline
def lower_cat_features(df):
    df_ = df.copy()
    for feat in df_.columns:
        df_[feat] = df_[feat].str.lower()
    return df_

#Set Logger for Railway ##################################################################################################

class CustomRailwayLogFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(log_record)

def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # this should be just "logger.setLevel(logging.INFO)" but markdown is interpreting it wrong here...
    handler = logging.StreamHandler()
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = CustomRailwayLogFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = get_logger()
# End logger stuff #

# Begin database stuff ###################################################################################################

DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff #




# Unpickle the previously-trained model##########################################################
#columns
with open(os.path.join('model_data', 'columns.json')) as fh:
    columns = json.load(fh)

#pipeline
with open(os.path.join('model_data', 'pipeline.pickle'), 'rb') as fh:
    pipeline = joblib.load(fh)

#dtypes
with open(os.path.join('model_data', 'dtypes.pickle'), 'rb') as fh:
    dtypes = pickle.load(fh)

#accepted key types
with open(os.path.join('model_data', 'key_types.pickle'), 'rb') as fh:
    key_types = pickle.load(fh)

# End model un-pickling



#input validation #############################################################################
def check_request(request):
    '''
    Initial request check
    Validates that our request is well formatted (has id and data keys in request dictionary)
        Input:
        Request dictionary with keys "observation_id" and "data"~
        
        Returns:
        - assertion value: True if request is ok, False otherwise
         - error message: empty if request is ok, False otherwise
    '''
     
    #Check if input has Id
    if "observation_id" not in request.keys():
        error = f"No observation_id in request: {request}"
        return False, error
    
    #Check if input has data
    if "data" not in request.keys():
        error = f"No 'data' observation in request: {request}"
        return False, error

    return True, ""



#check columns in request
def check_cols(request):
    """
        Check made after id and data keys are present in request
        Validates that our observation only has valid columns

        Input:
        Request dictionary with keys "observation_id" and "data"

        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    request_cols = set(request["data"].keys())
    expected_cols = set(columns)

    #Get difference between the request keys/columns and the expected column list
    difference = list(request_cols.symmetric_difference(expected_cols))

    if difference:
        error = f"Error in input columns ({difference}) in request {request}"
        return False, error
    
    return True, ""

def check_values(observation):
    '''
        Check make after columns input is okay
        Validates that all fields are in the observation and values are valid

        Input:
        observation dictionary with keys "observation_id" and "data"
        
        Returns:
        - assertion value: True if all provided columns contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    '''

    cat_variables = ["sex","race", "workclass", "education","marital-status"]
    num_variables = ["capital-gain","capital-loss"]
    

    for key in key_types.keys():
        #check in categorical values:
        if key in cat_variables:
            if observation[key] not in key_types[key]:
                error = f"Invalid {key} input: {observation[key]}"
                return False, error
            
        #check in specified numerical values
        elif key in num_variables:
            if observation[key] < 0:
                error = f"{key} input out of bounds: {observation[key]}"
                return False, error
        
        #check the remaining non specified numerical values | "age" and "hours-per-week"
        else:
            if observation[key] < key_types[key][0] or observation[key] > key_types[key][1]:
                error = f"{key} input out of bounds: {observation[key]}"
                return False, error

    return True, ""

########################################

# Begin webserver stuff###################################################################################

app = Flask(__name__)

@app.route("/predict", methods = ["POST"])
def predict():
    obs_dict = request.get_json()

    request_ok, error = check_request(obs_dict)
    if not request_ok:
        response =  {"error": error}
        return jsonify(response)
    
    #get id and data of dictionary as variables
    obs_id = obs_dict["observation_id"]
    obs_data = obs_dict["data"]

    cols_ok, error = check_cols(obs_dict)
    if not cols_ok:
        response = {"error": error}
        return jsonify(response)
    
    vals_okay, error = check_values(obs_data)
    if not vals_okay:
        response = {"error": error}
        return jsonify(response)
    

    #If everythings passes, calculate prediction and proba
    #Get data and turn into dataframe
    request_pd = pd.DataFrame([obs_data], columns = columns).astype(dtypes)

    #prepare response 
    proba = float(pipeline.predict_proba(request_pd)[0, 1])
    prediction = bool(pipeline.predict(request_pd)[0])

    response = {"observation_id": obs_id,
                "prediction" : prediction,
                "probability" : proba}
    
    p = Prediction(observation_id = obs_id,
                   observation = obs_data,
                   proba = prediction)

    try:
        p.save()

    except IntegrityError:
        error_msg = f"ERROR: Observation id '{obs_id}' already exists!"
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
        
    return jsonify(response)

@app.route("/update", methods = ["POST"])
def update():
    obs_dict = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs_dict["observation_id"])
        p.true_class = obs_dict["true_class"]
        p.save()
        return jsonify(model_to_dict(p))
    
    except Prediction.DoesNotExist:
        error_msg = f"Observation id '{obs_dict["observation_id"]}' does not exist!"
        return jsonify({"error": error_msg})


# if __name__ == "__main__":
#     app.run(debug = True)
#     # app.run()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)





    
