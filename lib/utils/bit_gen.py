import numpy as np
def bit_gen(num_streams, length):

    """
    Gera streams de bits aleatórios iid.
    usar yield caso queira gerar stream contínuo e analizar em tempo real
        Input: int num_streams number of streams. int length of stream
        Output: array-like stream num_stream by length sized
    """
    stream = np.random.choice([0, 1], p=[0.5, 0.5], size=(num_streams, length))
    if num_streams == 1:
        stream = stream[0]
    #stream = np.random.randint(0,2,num_streams*length).reshape(num_streams,length)
    return stream
