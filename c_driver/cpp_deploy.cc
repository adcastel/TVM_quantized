/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief Example code on load and run TVM module.s
 * \file cpp_deploy.cc
 */
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <sys/time.h>
#include <cstdio>

#define DTYPE float
#define Aref(a1,a2)  AA[ (a2)*(Alda)+(a1) ]
#define Bref(a1,a2)  BB[ (a2)*(Blda)+(a1) ]
#define Cref(a1,a2)  CC[ (a2)*(Clda)+(a1) ]
#define float16_t _Float16
void gemm_base( int m, int n, int k, DTYPE *AA, int Alda, DTYPE *BB, int Blda, DTYPE *CC, int Clda ){
        /*
         *    Baseline micro-kernel, for cases where the dimension does not match MR x NR
         *    */
   int    i, j, p;
   DTYPE tmp;
   for ( j=0; j<n; j++ ){
      for ( i=0; i<m; i++ ){
	      tmp=0.0;
   for ( p=0; p<k; p++ ){
                   tmp+=AA[i*k+p] * BB[p*n+j];
   }
   	CC[i*n+j] = tmp;
      }
   }
}



double Verify(tvm::runtime::Module mod, std::string fname, int M, int N, int K, int mr, int nr, int test) {
  // Get the function from the module.
	std::cout<<"ADRIAN "<<fname<<" Test "<<test<<std::endl;
  tvm::runtime::PackedFunc func = mod.GetFunction(fname);
  ICHECK(func != nullptr);
  // Allocate the DLPack data structures.
  //
  // Note that we use TVM runtime API to allocate the DLTensor in this example.
  // TVM accept DLPack compatible DLTensors, so function can be invoked
  // as long as we pass correct pointer to DLTensor array.
  //
  // For more information please refer to dlpack.
  // One thing to notice is that DLPack contains alignment requirement for
  // the data pointer and TVM takes advantage of that.
  // If you plan to use your customized data container, please
  // make sure the DLTensor you pass in meet the alignment requirement.
  //
  DLTensor* A;
  DLTensor* B;
  DLTensor* C;
  //DLTensor* Caux;
  int ndim = 2;
  // TEST 1
  int dtypeA_code = kDLFloat;
  int dtypeB_code = kDLFloat;
  int dtypeC_code = kDLFloat;
  int dtypeA_bits = 32;
  int dtypeB_bits = 32;
  int dtypeC_bits = 32;
  int dtype_lanes = 1;
  int i32_zero = 0;
  int16_t i16_zero = int16_t(0);
  float fp32_zero = 0.0f;
  float16_t fp16_zero = float16_t(0.0);
  void * zero;
  if (test == 2){
      dtypeA_code = kDLFloat;
      dtypeB_code = kDLFloat;
      dtypeC_code = kDLFloat;
      dtypeA_bits = 16;
      dtypeB_bits = 16;
      dtypeC_bits = 32;
      zero = &fp32_zero;
  }
  else if (test == 3){
      dtypeA_code = kDLInt;
      dtypeB_code = kDLInt;
      dtypeC_code = kDLFloat;
      dtypeA_bits = 8;
      dtypeB_bits = 8;
      dtypeC_bits = 32;
      zero = &fp32_zero;
  }
  else if (test == 4){
      dtypeA_code = kDLInt;
      dtypeB_code = kDLInt;
      dtypeC_code = kDLFloat;
      dtypeA_bits = 8;
      dtypeB_bits = 8;
      dtypeC_bits = 16;
      zero = &fp16_zero;
  }
  else if (test == 5){
      dtypeA_code = kDLInt;
      dtypeB_code = kDLInt;
      dtypeC_code = kDLInt;
      dtypeA_bits = 8;
      dtypeB_bits = 8;
      dtypeC_bits = 32;
      zero = &i32_zero;
  }
  else if (test == 6){
      dtypeA_code = kDLInt;
      dtypeB_code = kDLInt;
      dtypeC_code = kDLInt;
      dtypeA_bits = 8;
      dtypeB_bits = 8;
      dtypeC_bits = 16;
      zero = &i16_zero;
  }
  else if (test == 7){
      dtypeA_code = kDLFloat;
      dtypeB_code = kDLFloat;
      dtypeC_code = kDLFloat;
      dtypeA_bits = 16;
      dtypeB_bits = 16;
      dtypeC_bits = 16;
      zero = &fp16_zero;
  }
  int device_type = kDLCPU;
  int device_id = 0;
  int m=M;
  int n=N;
  int k=K;
  int64_t shapeA[2] = {m,k};
  int64_t shapeB[2] = {k,n};
  int64_t shapeC[2] = {m,n};
  //int64_t shapeCa[2] = {m,1};
  //float16_t * AA = (float16_t *)malloc(m*k*sizeof(float16_t)); 
  //float16_t * BB = (float16_t *)malloc(k*n*sizeof(float16_t)); 
  //float16_t * CC = (float16_t *)malloc(m*n*sizeof(float16_t)); 
  TVMArrayAlloc(shapeA, ndim, dtypeA_code, dtypeA_bits, dtype_lanes, device_type, device_id, &A);
  TVMArrayAlloc(shapeB, ndim, dtypeB_code, dtypeB_bits, dtype_lanes, device_type, device_id, &B);
  TVMArrayAlloc(shapeC, ndim, dtypeC_code, dtypeC_bits, dtype_lanes, device_type, device_id, &C);
  //A->data=AA;
  //B->data=BB;
  //C->data=CC;
  if (test <= 3){   
      for (int i = 0; i < m*n; ++i) {
         static_cast<float*>(C->data)[i] = 0;
      }
  }
  else if (test == 4 or test == 7){
      for (int i = 0; i < m*n; ++i) {
	      static_cast<float16_t*>(C->data)[i] = (float16_t)0;
      }
  
  }
  else if (test == 5){
      for (int i = 0; i < m*n; ++i) {
         static_cast<int*>(C->data)[i] = 0;
      }
  
  }
  else if (test == 6){
      for (int i = 0; i < m*n; ++i) {
         static_cast<int16_t*>(C->data)[i] = 0;
      }
  }
  // A
  if (test == 1){
      for (int i = 0; i < m*k; ++i) {
         static_cast<float*>(A->data)[i] = i;
      }
  }
  else if (test == 2 or test == 7){
      for (int i = 0; i < m*k; ++i) {
	  static_cast<float16_t*>(A->data)[i] = (float16_t)i;
      }
  }
  else if (test > 2 and test < 7){
      for (int i = 0; i < m*k; ++i) {
         static_cast<int8_t*>(A->data)[i] = i;
      }
  }
  // B
  if (test == 1){
      for (int i = 0; i < k*n; ++i) {
          static_cast<float*>(B->data)[i] = i*2;
      }
  }
  else if (test == 2 or test == 7){
      for (int i = 0; i < k*n; ++i) {
         static_cast<float16_t*>(B->data)[i] = (float16_t)(i*2);

      }
  }
  else if (test > 2 or test < 7){
      for (int i = 0; i < k*n; ++i) {
          static_cast<int8_t*>(B->data)[i] = i*2;
      }
  }
  struct timeval t_ini, t_end;
  // The signature of the function is specified in tvm.build
  double wtotal, total;
  int reps=1000;
  if (test <= 3){ //fp32
      gettimeofday (&t_ini, NULL);
      func(A, B, C, fp32_zero);
      gettimeofday (&t_end, NULL);
      wtotal =(t_end.tv_sec - t_ini.tv_sec) * 1e6 + t_end.tv_usec - t_ini.tv_usec;
      total = 0; //  =(t_end.tv_sec - t_ini.tv_sec) * 1000000 + t_end.tv_usec - t_ini.tv_usec;
      gettimeofday (&t_ini, NULL);
      for(int i = 0; i < reps; i++){
      func(A, B, C, fp32_zero);
      }
      gettimeofday (&t_end, NULL);
      total =  (t_end.tv_sec - t_ini.tv_sec) * 1000000 + t_end.tv_usec - t_ini.tv_usec;
  } else if (test == 4 or test == 7){ //fp16
      gettimeofday (&t_ini, NULL);
      //func(A, B, C, fp16_zero);
      gettimeofday (&t_end, NULL);
      wtotal =(t_end.tv_sec - t_ini.tv_sec) * 1e6 + t_end.tv_usec - t_ini.tv_usec;
      total = 0; //  =(t_end.tv_sec - t_ini.tv_sec) * 1000000 + t_end.tv_usec - t_ini.tv_usec;
      gettimeofday (&t_ini, NULL);
      for(int i = 0; i < reps; i++){
      //continue;
	      func(A, B, C, fp16_zero);
      }
      gettimeofday (&t_end, NULL);
      total =  (t_end.tv_sec - t_ini.tv_sec) * 1000000 + t_end.tv_usec - t_ini.tv_usec;
  
  } else if(test == 5){ //int32
      gettimeofday (&t_ini, NULL);
      func(A, B, C, i32_zero);
      gettimeofday (&t_end, NULL);
      wtotal =(t_end.tv_sec - t_ini.tv_sec) * 1e6 + t_end.tv_usec - t_ini.tv_usec;
      total = 0; //  =(t_end.tv_sec - t_ini.tv_sec) * 1000000 + t_end.tv_usec - t_ini.tv_usec;
      gettimeofday (&t_ini, NULL);
      for(int i = 0; i < reps; i++){
      func(A, B, C, i32_zero);
      }
      gettimeofday (&t_end, NULL);
      total =  (t_end.tv_sec - t_ini.tv_sec) * 1000000 + t_end.tv_usec - t_ini.tv_usec;
  } else if (test == 6){ //int16
      gettimeofday (&t_ini, NULL);
      func(A, B, C, i16_zero);
      gettimeofday (&t_end, NULL);
      wtotal =(t_end.tv_sec - t_ini.tv_sec) * 1e6 + t_end.tv_usec - t_ini.tv_usec;
      gettimeofday (&t_ini, NULL);
      for(int i = 0; i < reps; i++){
      func(A, B, C, i16_zero);
      }
      gettimeofday (&t_end, NULL);
      total =  (t_end.tv_sec - t_ini.tv_sec) * 1000000 + t_end.tv_usec - t_ini.tv_usec;
  }
  //total +=(t_end.tv_sec - t_ini.tv_sec) * 1e6 + t_end.tv_usec - t_ini.tv_usec;
  reps *=1;
  total=(total/1e6)/reps;
  wtotal=wtotal/1e6;
  double gflops = (2.0 * m * n * k)/(total *1e9);
  TVMArrayFree(A);
  TVMArrayFree(B);
  TVMArrayFree(C);
  return gflops;

}

double DeploySingleOp(int M, int N, int K, int MR, int NR, int TEST) {
  // Normally we can directly
  // NO HACE FALTA EL SIGUIENTE SWITCH PERO 
  // ES POR SI EN eL FUTURO LO GENERAMOS EN DINÁMICO
	std::string folder="";
    switch(TEST){
	    case(1):
		    folder="float32float32float32";
		    break;
	    case(2):
		    folder="float16float16float32";
		    break;
	    case(3):
		    folder="int8int8float32";
		    break;
	    case(4):
		    folder="int8int8float16";
		    break;
	    case(5):
		    folder="int8int8int32";
		    break;
	    case(6):
		    folder="int8int8int16";
		    break;
	    case(7):
		    folder="float16float16float16";
		    break;
    }

    std::string name="eg_gemm_"+std::to_string(M)+"_"+std::to_string(N)+"_"+std::to_string(K)+"_"+std::to_string(MR)+"_"+std::to_string(NR);
    //std::cout<<name<<std::endl;
  //LOG(INFO) << "Verify load function from system lib";
  tvm::runtime::Module mod_syslib = (*tvm::runtime::Registry::Get("runtime.SystemLib"))();
  return Verify(mod_syslib, name, M, N, K, MR, NR, TEST);
}



int main(int argc, char * argv []) {
  
  int M=atoi(argv[1]);
  int N=atoi(argv[2]);
  int K=atoi(argv[3]);
  int MR_INI=atoi(argv[4]);
  int MR_END=atoi(argv[5]);
  int MR_STEP=atoi(argv[6]);
  int NR_INI=atoi(argv[7]);
  int NR_END=atoi(argv[8]);
  int NR_STEP=atoi(argv[9]);
  
  int TEST=(argc > 9) ? atoi(argv[10]): 1;
  
  double gf;
  double best_gf=-1.0;
  int best_mr = 0;
  int best_nr = 0;
  for (int mr = MR_INI; mr < MR_END; mr = mr + MR_STEP){
      for (int nr = NR_INI; nr < NR_END; nr = nr + NR_STEP){
          gf= DeploySingleOp(M,N,K,mr,nr, TEST);
          printf("Test: %d %d %d %d %d GFLOPS: %f\n",M,N,K,mr,nr,gf);
	  if (gf > best_gf){
	      best_gf = gf;
	      best_mr = mr;
	      best_nr = nr;
	  }
      }
  }
  printf("Best: %d %d %d %d %d GFLOPS: %f\n",M,N,K,best_mr,best_nr,best_gf);
  return 0;
}
