# TVM_quantized

 TVM_NUM_THREADS=1 python3.8 -u gemm_auto_sched.py EG TEST TRIALS BLIS MODEL MACH CROSS C_DRIVER

EG (0|1)    0-> TVM Auto-Scheduling. 1-> Experience-guided

TEST (0..7) Datatype combination (to be completed)

TRIALS NUM  Number of trials for Auto-Scheduling (No efect with EG=1)

BLIS (0|1) 0-> Auto gemm 1-> BLIS-Auto-scheduling (No efect with EG=1)

MODEL (0|1) 0-> Resnet50v1.5. 1->Googlelenet

MACH -> Configuration file for the destination device 

CROSS (0|1) Enable cross-compilation. Avoids execution when enabled.

C_DRIVER (0|1) Executes the test in c++


