import pandas as pd
import os 

def get_all_domain_names(path: str) -> pd.DataFrame:
    # fnames = [_ for _ in os.listdir(path) if _.endswith('.gz')]
    return pd.read_csv(path)


if __name__ == "__main__":
     path  = (os.environ.get('DOMAIN_NAMES') or 'datasets/data/domains')
     fname = 'allzones_zone_full_20220805.gz'
     gen = pd.read_csv(os.path.join(path,fname), chunksize=100000, header=None, names=['domain_name'])
     df =  pd.concat((x.query("domain_name.str.endswith('.ai')") for x in gen), ignore_index=True)
    #  domains = get_all_domain_names(path)
    #  domains.head()
    
    # schema={
    # "math score": int
    # }
    
    # gen = pd.read_csv(csv_file, dtype=schema, chunksize=10000000)
    # df = pd.concat((x.query("`math score` >= 75") for x in gen), ignore_index=True)
