
import os




MR_INI=NR_INI=KR_INI=4
MR_END=NR_END=33
KR_END=5

MR_STEP=NR_STEP=KR_STEP=4

mrnr = []
for i in range(MR_INI, MR_END, MR_STEP):
    for j in range(NR_INI, NR_END, NR_STEP):
        for k in range(KR_INI, KR_END, KR_STEP):
            mrnr.append([i, j, k])


MNK=[
        [1, 1605632, 64, 147],
        [2, 401408, 64, 64],  
        [3, 401408, 64, 576], 
        [4, 401408, 256, 64], 
        [5, 401408, 64, 256], 
        [6, 401408, 128, 256], 
        [7, 100352, 128, 1152], 
        [8, 100352, 512, 128], 
        [9, 100352, 512, 256], 
        [10, 100352, 128, 512], 
        [11, 100352, 256, 512], 
        [12, 25088, 256, 2304], 
        [13, 25088, 1024, 256], 
        [14, 25088, 1024, 512], 
        [15, 25088, 256, 1024], 
        [16, 25088, 512, 1024], 
        [17, 6272, 512, 4608],  
        [18, 6272, 2048, 512],  
        [19, 6272, 2048, 1024], 
        [20, 6272, 512, 2048],  
        ]

"""
MNK = [[0, 128,128,128]]
mrnr = [[4,4,4]]
"""
MCNCKC = [[896,3072,512]]
LS = [4,8,16]
for linesize in LS:
    for layer, M, N, K in MNK:
        for MC, NC, KC in MCNCKC:
            for mr, nr, kr in mrnr:
                if nr > linesize and nr % linesize != 0:
                    continue
                """
                if mcnckc_study == 0:
                KC = model_level(NL1, CL1, WL1, Sdata, mr, nr); KC = math.floor(KC);
                MC = model_level(NL2, CL2, WL2, Sdata, KC, nr); MC = math.floor(MC);
                NC = model_level(NL3, CL3, WL3, Sdata, KC, MC); NC = math.floor(NC);
                """
                print("Starting test:", layer, M, N, K, MC, NC, KC, mr, nr, kr, linesize)
                namef="/home/adcastel/test_driver_c/lib/b3a2c0_{}_{}_{}_{}_{}_{}.o".format(M,N,K,mr,nr,linesize)
                print(namef)
                if os.path.exists(namef):
                    print("Ya existe")
                else:
                    os.system("make start M={} N={} K={} MC={} NC={} KC={} MR={} NR={} KR={} LS={}".format( M, N, K, MC, NC, KC, mr, nr, kr, linesize))
           
                print("******************************************************************")
