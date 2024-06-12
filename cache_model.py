from math import floor, ceil
import configparser

def model_level( NL, CL, WL, dataSize, m, n):

    if WL==2:                    
        k = NL * CL / (2.0 * m * dataSize)                        
    else:
        CAr = floor( ( float(WL) - 1.0 ) / (1.0 + float(n) / float(m) ) )
        if CAr==0:
            CAr = 1.0
            CBr = WL - 2
            k = CBr * NL * CL // (n * dataSize)
        else:
            CBr = ceil( ( float(n) / float(m) ) * float(CAr) )
            k = CAr * NL * CL // (m * dataSize);
    return k

def read_cache_config(cfg_file):
    config = configparser.RawConfigParser()
    config.read(cfg_file)

    features = dict(config.items('FEATURES'))
    SL1 = int(features['sl1'])
    WL1 = int(features['wl1'])
    NL1 = int(features['nl1'])
    CL1 = SL1 / (WL1 * NL1)

    SL2 = int(features['sl2'])
    WL2 = int(features['wl2'])
    NL2 = int(features['nl2'])
    CL2 = SL2 / (WL2 * NL2)

    SL3 = int(features['sl3'])
    WL3 = int(features['wl3'])
    NL3 = int(features['nl3'])
    CL3 = SL3 / (WL3 * NL3)
    return NL1, CL1, WL1, NL2, CL2, WL2, NL3, CL3, WL3

def get_optim_mc_nc_kc(dataSize, m, n, k, mr, nr, cfg_file ):
    
    NL1, CL1, WL1, NL2, CL2, WL2, NL3, CL3, WL3 = read_cache_config(cfg_file)

    kc = model_level(NL1, CL1, WL1, dataSize, mr, nr)
    kc = min(k, kc)
    
    mc = model_level(NL2, CL2, WL2, dataSize, nr, kc)
    mc = mc // mr * mr
    mc = min(m, mc)
    
    nc = model_level(NL3, CL3, WL3, dataSize, kc, mc)
    nc = nc // nr * nr
    nc = min(n, nc)
    
    return int(mc), int(nc), int(kc)

    

