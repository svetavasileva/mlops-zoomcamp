from typing import List, Tuple
from pandas import DataFrame, Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from mlops.utils.data_preparation.encoders import vectorize_features
from mlops.utils.data_preparation.feature_selector import select_features
import mlflow 
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pickle
# if 'data_exporter' not in globals():
#     from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def train_model(
    data: DataFrame,
    **kwargs,
) -> Tuple[BaseEstimator, DictVectorizer]:

    target = kwargs.get('target', 'duration')

    X, X_val, dv = vectorize_features(select_features(data))
    y: Series = data[target]

    y = y.values

    print('set_tracking_uri')
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment('linear-regression')  
    
    print('start run')
    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X, y)

    print('get_experiment_by_name')

    client = MlflowClient()
    experiment = client.get_experiment_by_name('linear-regression')
    last_run = client.search_runs(experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1)[0]
    RUN_ID = last_run.info.run_id
    model_uri = "runs:/{}/model".format(RUN_ID)
    mlflow.register_model(model_uri,name="linear-regression_model")
    print('logging_model')
    mlflow.sklearn.log_model(model, artifact_path="model")
    
    # path = client.download_artifacts(run_id=RUN_ID, path='dict_vectorizer.bin')
    
    # with open(path, 'rb') as f_out:
    #     dv = pickle.load(f_out)

    print(model.intercept_)
    return model, dv