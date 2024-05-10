import numpy as np

def ERGAS(I,I_Fus,Resize_fact):
    Wn, Hn, Cn = I.shape
    Err = I - I_Fus
    ERGAS = 0
    for iLR in range(Cn):
        ERGAS = ERGAS + np.mean(Err[:,:, iLR]**2) / (np.mean((I[:,:, iLR]))) ** 2

    ERGAS = (100 / Resize_fact) * np.sqrt((1 / Cn) * ERGAS);
    return ERGAS