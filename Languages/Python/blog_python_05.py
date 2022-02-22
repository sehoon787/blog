import numpy as np
import pandas as pd
import time
from functools import wraps

numOfRows = 50000  # number of rows

def check_time(function):
    @wraps(function)
    def measure(*args, **kwargs):
        stime = time.time()
        result = function(*args, **kwargs)
        print(f"Run Time: {function.__name__} took {time.time() - stime} s")
        return result
    return measure

@check_time
def pandas_append():
    df = pd.DataFrame([[1, 2, 3, 4, 5]], columns=['A', 'B', 'C', 'D', 'E'])
    for i in range(numOfRows - 1):
        df = df.append(dict((a, np.random.randint(100)) for a in ['A', 'B', 'C', 'D', 'E']),
                         ignore_index=True)

@check_time
def pandas_concat():
    df = pd.DataFrame([[1, 2, 3, 4, 5]], columns=['A', 'B', 'C', 'D', 'E'])
    for i in range(numOfRows - 1):
        temp = pd.DataFrame(dict((a, [np.random.randint(100)]) for a in ['A', 'B', 'C', 'D', 'E']))
        df = pd.concat([df, temp], axis=0)

@check_time
def pandas_loc():
    df = pd.DataFrame([[1, 2, 3, 4, 5]], columns=['A', 'B', 'C', 'D', 'E'])
    for i in range(5, numOfRows):
        df.loc[i] = np.random.randint(100, size=(1, 5))[0]

@check_time
def dict_in_list():
    data = [{"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}]
    for i in range(numOfRows-1):
        data.append(dict((a, np.random.randint(100)) for a in ['A', 'B', 'C', 'D', 'E']))
    df = pd.DataFrame.from_dict(data)

@check_time
def dict_list_comprehension():
    data = [{(a, np.random.randint(100)) for a in ['A', 'B', 'C', 'D', 'E']} for i in range(numOfRows)]
    df = pd.DataFrame.from_dict(data)

@check_time
def list_append():
    data = [[1, 2, 3, 4, 5]]
    for i in range(numOfRows-1):
        data.append([np.random.randint(100) for a in range(5)])
    df = pd.DataFrame.from_dict(data)

@check_time
def list_comprehension():
    data = [[np.random.randint(100) for a in range(5)] for i in range(numOfRows)]
    df = pd.DataFrame.from_dict(data)

pandas_append()
pandas_concat()
pandas_loc()
dict_in_list()
dict_list_comprehension()
list_append()
list_comprehension()
