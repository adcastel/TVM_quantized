#!/bin/bash


MR_INI=NR_INI=KR_INI=4
MR_END=NR_END=33
KR_END=5

MR_STEP=NR_STEP=KR_STEP=4






declare -a id
declare -a mm
declare -a nn
declare -a kk
declare -a mrnr
declare -a linesize
MC=896
NC=3072
KC=512

#resnet50
id=(1	2	3	4	5	6	7	8	9	10	11	12	13	14	15	16	17	18	19	20)
mm=(1605632	401408	401408	401408	401408	401408	100352	100352	100352	100352	100352	25088	25088	25088	25088	25088	6272	6272	6272	6272)
nn=(64	64	64	256	64	128	128	512	512	128	256	256	1024	1024	256	512	512	2048	2048	512)
kk=(147	64	576	64	256	256	1152	128	256	512	512	2304	256	512	1024	1024	4608	512	1024	2048)
linesize=(4 8 16)
mrnr=(4 8 12 16 20 24 28 32)
for (( ls=0; ls<3; ls++ ))
do 
    for (( l=0; l<20; l++ ))
    do 
	for (( mr=0; mr<8; mr++ ))
        do 
	    for (( nr=0; nr<8; nr++ ))
            do 
            echo "Starting layer ${id[l]} with ${mm[l]} ${nn[l]} ${kk[l]} ${MC} ${NC} ${KC} ${mrnr[mr]} ${mrnr[nr]} 8"
	    #make M=${mm[l]} N=${nn[l]} K=${kk[l]} MC=${MC} NC=${NC} KC=${KC} MR=${mrnr[mr]} NR=${mrnr[nr]} KR=4 LS=${linesize[ls]}
	    make run M=${mm[l]} N=${nn[l]} K=${kk[l]} MR=${mrnr[mr]} NR=${mrnr[nr]} LS=8
            done

	done
    done
done


