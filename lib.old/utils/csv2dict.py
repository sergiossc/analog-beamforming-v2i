import pandas as pd

def pathfilesset2dict(csvfilename):
    df = pd.read_csv(csvfilename, index_col=0)
    pathfiles_dict = df.to_dict()
    return pathfiles_dict
