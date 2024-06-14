#!/bin/bash


declare -a id
declare -a mm
declare -a nn
declare -a kk
declare -a mrnr

#resnet50
id=(1	2	3	4	5	6	7	8	9	10	11	12	13	14	15	16	17	18	19	20)
mm=(1605632	401408	401408	401408	401408	401408	100352	100352	100352	100352	100352	25088	25088	25088	25088	25088	6272	6272	6272	6272)
mm=(12544	3136	3136	3136	3136	3136	784	784	784	784	784	196	196	196	196	196	49	49	49	49)
nn=(64	64	64	256	64	128	128	512	512	128	256	256	1024	1024	256	512	512	2048	2048	512)
kk=(147	64	576	64	256	256	1152	128	256	512	512	2304	256	512	1024	1024	4608	512	1024	2048)
linesize=(4)
mrnr=(4 8 12 16 20 24 28 32)
for (( ls=0; ls<1; ls++ ))
do 
    for (( l=0; l<20; l++ ))
    do 
	for (( mr=0; mr<8; mr++ ))
        do 
	    for (( nr=0; nr<8; nr++ ))
            do 
            echo "Starting layer ${id[l]} with ${mm[l]} ${nn[l]} ${kk[l]} ${mrnr[mr]} ${mrnr[nr]}"
	    make run M=${mm[l]} N=${nn[l]} K=${kk[l]} MR=${mrnr[mr]} NR=${mrnr[nr]}
            done

	done
    done
done


