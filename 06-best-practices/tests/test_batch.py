import batch 
import pandas as pd
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df = pd.DataFrame(data, columns=columns)

actual_df = batch.prepare_data(df,columns)

expected_df = {'PULocationID': {0: '-1', 1: '1'}, 
               'DOLocationID': {0: '-1', 1: '1'}, 
               'tpep_pickup_datetime': {0: str(int(datetime(2023, 1, 1, 1, 1, 0).timestamp())), 
                                        1: str(int(datetime(2023, 1, 1, 1, 2, 0).timestamp()))} ,
                'tpep_dropoff_datetime': {0: str(int(datetime(2023, 1, 1, 1, 10, 0).timestamp())) , 
                                          1: str(int(datetime(2023, 1, 1, 1, 10, 0).timestamp()))},
                'duration': {0:'9',
                             1:'8'}}
import sys
import traceback
try:
    print(actual_df.head().to_dict())
    print(expected_df)
    assert actual_df.to_dict() == expected_df
except AssertionError:
    _, _, tb = sys.exc_info()
    traceback.print_tb(tb) # Fixed format
    tb_info = traceback.extract_tb(tb)
    filename, line, func, text = tb_info[-1]

    print('An error occurred on line {} in statement {}'.format(line, text))
    exit(1)