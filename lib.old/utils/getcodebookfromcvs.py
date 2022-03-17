import pandas as pd

def getcodebookfromcsv(csv_pathfile):
    df_rec = pd.read_csv(csv_pathfile) # 'tx_codebook_da0bc8b2-a1c4-46c5-9e9a-457bd9ff64fe.csv')
    codebook = {}
    cb = df_rec.to_dict() 
    for k, v in cb.items():
        #print ('k: ', k)
        #print ('v: ', v.values())
        if str(k) != str('Unnamed: 0'):
            codebook[k] = v.values()
    return codebook


#csv_pathfile = 'tx_codebook_b06c1d1a-8785-461f-ac2e-e0f383791dbd.csv'
#cb = getcodebookfromcsv(csv_pathfile) 
#print (cb)
