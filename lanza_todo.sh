#/bin/bash!


MACH="carmel"
TESTS="1"
TRIALS="1000"
BLIS="0"
MODELS="0"
EG=0

for t in ${TESTS}
do
	for r in ${TRIALS}
	do
	        for b in ${BLIS}
		do
			for m in ${MODELS}
			do
				echo "TVM_NUM_THREADS=1 python3.8 -u gemm_auto_sched.py ${EG} ${t} ${r} ${b} ${m} ${MACH}  2>&1 | tee ${MACH}_auto_test_${t}_blis_${b}_trials_${r}_model_${m}.out;"
				TVM_NUM_THREADS=1 python3.8 -u gemm_auto_sched.py ${EG} ${t} ${r} ${b} ${m} ${MACH}  2>&1 | tee ${MACH}_auto_test_${t}_blis_${b}_trials_${r}_model_${m}.out;
			done
		done
	done
done
TESTS="1 2 3 4 5 6 7"
TESTS="1"
EG=1
for t in ${TESTS}
do
	for m in ${MODELS}
	do
		echo "TVM_NUM_THREADS=1 python3.8 -u gemm_auto_sched.py ${EG} ${t} 1 0 ${m} ${MACH}  2>&1 | tee ${MACH}_eg_test_${t}_model_${m}.out;"
		TVM_NUM_THREADS=1 python3.8 -u gemm_auto_sched.py ${EG} ${t} 1 0 ${m} ${MACH}  2>&1 | tee ${MACH}_eg_test_${t}_model_${m}.out;
	done
done

