''' nasdaq.py

    - Downloads nasdaq companies via FTP
    - We are only interested in the companies names therefore other
    information is removed during the cleaning process

    Data Source: ftp://ftp.nasdaqtrader.com/symboldirectory/
    Data Documentation: http://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs
    
    Author: Gregory Bryant
'''
import pandas as pd
import re
import os 
from datetime import date
import itertools

nasdaq_listing = 'ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt'
NASDAQ_DATA_DIR = 'datasets/data/nasdaq'

def getNasdaq() -> pd.DataFrame:
    if is_current(NASDAQ_DATA_DIR):
        save(process(),NASDAQ_DATA_DIR)
        return pd.read_parquet(get_current_nasdaq_dataset(NASDAQ_DATA_DIR))
    else:
        return pd.read_parquet(get_current_nasdaq_dataset(NASDAQ_DATA_DIR))
    
def process() -> pd.DataFrame:
    return _cleanse(pd.read_csv(nasdaq_listing, sep="|"))
    
def _cleanse(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # step 1 remove Testing entries 
    df = df[df['Test Issue'] == 'N']
    
    # step 2 _transforms on Security Name and Creates new Column Company Name
    df['Company Name'] = df['Security Name'].apply(lambda x: _transforms(x))
    return df
    
def _transforms(txt):
    # We are using this transform to clean the 'Company Name' Column 
    txt = txt.split(' - ')[0]
    txt = txt.split(' I ')[0]
    # transformation 
    pattern = [
        '[^\w\s]',        # remove punctuation 
        '( - )',          # remove hyphens
        '[^Inc]',
        ' I '
        ]
    txt = re.sub('|'.join(pattern), '', txt)
    return txt.strip()

def save(df:pd.DataFrame, path:str):
    # this data changes frequently therfore we'll add timestape to the file name
    # to denote when the last time file as saved.
    # nasdaq-YYYYMMDD.parquet.gzip
    today = date.today()
    file = f'nasdaq-{str(10000*today.year + 100*today.month + today.day)}.parquet.gzip'
    df.to_parquet(os.path.join(path,file), compression='gzip')

def is_current(path:str) -> bool:
    fnames = [_ for _ in os.listdir(path) if _.endswith('.gzip')]
    today = date.today()
    current_date = str(10000*today.year + 100*today.month + today.day)
    
    if len(fnames) > 0:
       file_dates = list(itertools.chain.from_iterable([re.findall('[0-9]+', f) for f in fnames]))
       file_dates.sort()
       
    if  current_date > file_dates[-1]:
        return True    
    else:
        return False
    
def get_current_nasdaq_dataset(path:str) -> str:
    fnames = [_ for _ in os.listdir(path) if _.endswith('.gzip')]
    fnames.sort()
    return os.path.join(path, fnames[-1])
            
# if __name__ == "__main__":
#     path  = (os.environ.get('NASDAQ_PATH') or 'datasets/data/nasdaq')
#     fname = get_current_nasdaq_dataset(path)
    
    # if not check_if_data_is_current((os.environ.get('NASDAQ_PATH') or 'datasets/data/nasdaq')):
    #     print(f"Updates are needed")
        # save(
        # df = process(), 
        # # TODO: remove this onece docker images is rebuilted 
        # path = os.environ.get('NASDAQ_PATH') or 'datasets/data/nasdaq'
        # )
    