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



double Verify(tvm::runtime::Module mod, std::string fname, int M, int N, int K, int mr, int nr) {
  // Get the function from the module.
  tvm::runtime::PackedFunc f = mod.GetFunction(fname);
  ICHECK(f != nullptr);
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
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  float zero = 0.0;
  int device_type = kDLCPU;
  int device_id = 0;
  int m=M;
  int n=N;
  int k=K;
  int64_t shapeA[2] = {m,k};
  int64_t shapeB[2] = {k,n};
  int64_t shapeC[2] = {m,n};
  //int64_t shapeCa[2] = {m,1};
  //float * AA = (float *)malloc(m*k*sizeof(float)); 
  //float * BB = (float *)malloc(k*n*sizeof(float)); 
  //float * CC = (float *)malloc(m*n*sizeof(float)); 

  TVMArrayAlloc(shapeA, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &A);
  TVMArrayAlloc(shapeB, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &B);
  TVMArrayAlloc(shapeC, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &C);
   
  for (int i = 0; i < m*n; ++i) {
    static_cast<float*>(C->data)[i] = 0;
  }
  for (int i = 0; i < m*k; ++i) {
    static_cast<float*>(A->data)[i] = i;
  }
  for (int i = 0; i < k*n; ++i) {
    static_cast<float*>(B->data)[i] = i*2;
  }
   //A->data=AA;
   //B->data=BB;
   //C->data=CC;
   //gemm_base(m, n, k, AA, m, BB, k, CC, m );

  // Invoke the function
  // PackedFunc is a function that can be invoked via positional argument.
  struct timeval t_ini, t_end;
  // The signature of the function is specified in tvm.build
  int reps=100;
  gettimeofday (&t_ini, NULL);
  f(A, B, C, zero);
  gettimeofday (&t_end, NULL);
  double wtotal =(t_end.tv_sec - t_ini.tv_sec) * 1e6 + t_end.tv_usec - t_ini.tv_usec;
  double total = 0; //  =(t_end.tv_sec - t_ini.tv_sec) * 1000000 + t_end.tv_usec - t_ini.tv_usec;
  gettimeofday (&t_ini, NULL);
  for(int i = 0; i < reps; i++){
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  f(A, B, C, zero);
  }
  gettimeofday (&t_end, NULL);
  total +=(t_end.tv_sec - t_ini.tv_sec) * 1e6 + t_end.tv_usec - t_ini.tv_usec;
  reps *=30;
  total=(total/1e6)/reps;
  wtotal=wtotal/1e6;
  double gflops = (2.0 * m * n * k)/(total *1e9);
  //std::cout<<layer+1<<" "<<M<<" "<<N<<" "<<K<<" "<<MC<<" "<<NC<<" "<<KC<<" "<<mr<<" "<<nr<<" "<<LS<<" "<<gflops<<std::endl;
  // Print out the output
  //for (int i = 0; i < shape[0]; ++i) {
  //  ICHECK_EQ(static_cast<float*>(y->data)[i], i + 1.0f);
  //}
  //for (int i = 0; i < m*n; ++i) {
//	  std::cout<<"TVM "<<static_cast<float*>(C->data)[i]<<" C "<<CC[i]<<std::endl;
  //}
  //LOG(INFO) << "Finish verification...";
  TVMArrayFree(A);
  TVMArrayFree(B);
  TVMArrayFree(C);
  //free(AA);
  //free(BB);
  //free(CC);
  return gflops;

}

double DeploySingleOp(int M, int N, int K, int MR, int NR) {
  // Normally we can directly
    std::string name="eg_gemm_"+std::to_string(M)+"_"+std::to_string(N)+"_"+std::to_string(K)+"_"+std::to_string(MR)+"_"+std::to_string(NR);
    //std::cout<<name<<std::endl;
  //LOG(INFO) << "Verify load function from system lib";
  tvm::runtime::Module mod_syslib = (*tvm::runtime::Registry::Get("runtime.SystemLib"))();
  return Verify(mod_syslib, name, M, N, K, MR, NR);
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

  double gf;
  double best_gf=-1.0;
  int best_mr = 0;
  int best_nr = 0;
  for (int mr = MR_INI; mr < MR_END; mr = mr + MR_STEP){
      for (int nr = NR_INI; nr < NR_END; nr = nr + NR_STEP){
          gf= DeploySingleOp(M,N,K,mr,nr);
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
