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



double Verify(tvm::runtime::Module mod, std::string fname, int M, int N, int K, int mr, int nr, int LS, int MC, int NC, int KC, int layer) {
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
  int device_type = kDLCPU;
  int device_id = 0;
    int m=M;
    int n=N;
    int k=K;
  int64_t shapeA[2] = {m,k};
  int64_t shapeB[2] = {k,n};
  int64_t shapeC[2] = {m,n};
  //int64_t shapeCa[2] = {m,1};
  float * AA = (float *)malloc(m*k*sizeof(float)); 
  float * BB = (float *)malloc(k*n*sizeof(float)); 
  float * CC = (float *)malloc(m*n*sizeof(float)); 

  TVMArrayAlloc(shapeA, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &A);
  TVMArrayAlloc(shapeB, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &B);
  TVMArrayAlloc(shapeC, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &C);
  //for(int i=0; i<n*m;i++)
 //	  static_cast<float*>(C->data)[i] = 0;
 // for(int i=0; i<k*m;i++)
  //	  static_cast<float*>(A->data)[i] = 0;
  //for(int i=0; i<n*k;i++)
//	  static_cast<float*>(B->data)[i] = 0;
  //TVMArrayAlloc(shapeCa, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &Caux);
   
  for (int i = 0; i < m*n; ++i) {
    static_cast<float*>(C->data)[i] = 0;
    CC[i]=0;
  }
  for (int i = 0; i < m*k; ++i) {
    static_cast<float*>(A->data)[i] = i;
    AA[i]=i;
  }
  for (int i = 0; i < k*n; ++i) {
    static_cast<float*>(B->data)[i] = i*2;
    BB[i]=i*2;
  }
   //A->data=AA;
   //B->data=BB;
   //C->data=CC;
   //gemm_base(m, n, k, AA, m, BB, k, CC, m );

  // Invoke the function
  // PackedFunc is a function that can be invoked via positional argument.
  struct timeval t_ini, t_end;
  // The signature of the function is specified in tvm.build
  int reps=2;
  gettimeofday (&t_ini, NULL);
  f(A, B, C);
  gettimeofday (&t_end, NULL);
  double wtotal =(t_end.tv_sec - t_ini.tv_sec) * 1e6 + t_end.tv_usec - t_ini.tv_usec;
  double total = 0; //  =(t_end.tv_sec - t_ini.tv_sec) * 1000000 + t_end.tv_usec - t_ini.tv_usec;
  gettimeofday (&t_ini, NULL);
  for(int i = 0; i < reps; i++){
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
      f(A, B, C);
  }
  gettimeofday (&t_end, NULL);
  total +=(t_end.tv_sec - t_ini.tv_sec) * 1e6 + t_end.tv_usec - t_ini.tv_usec;
  reps *=30;
  //double total =(t_end.tv_sec - t_ini.tv_sec) * 1000000 + t_end.tv_usec - t_ini.tv_usec;
  total=(total/1e6)/reps;
    wtotal=wtotal/1e6;
    //double wgflops = (2.0 * m * n * k)/(wtotal *1e9);
    double gflops = (2.0 * m * n * k)/(total *1e9);
    //std::cout<<M<<","<<N<<" WTime: "<<wtotal<<" WGFLOPS: "<<wgflops<<std::endl;
    std::cout<<layer+1<<" "<<M<<" "<<N<<" "<<K<<" "<<MC<<" "<<NC<<" "<<KC<<" "<<mr<<" "<<nr<<" "<<LS<<" "<<gflops<<std::endl;
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
  free(AA);
  free(BB);
  free(CC);
  return gflops;

}

double DeploySingleOp(int M, int N, int K, int MR, int NR, int LS, int MC, int NC, int KC, int layer) {
  // Normally we can directly
    std::string name="b3a2c0_"+std::to_string(M)+"_"+std::to_string(N)+"_"+std::to_string(K)+"_"+std::to_string(MR)+"_"+std::to_string(NR)+"_"+std::to_string(LS);
    //std::cout<<name<<std::endl;
  //LOG(INFO) << "Verify load function from system lib";
  tvm::runtime::Module mod_syslib = (*tvm::runtime::Registry::Get("runtime.SystemLib"))();
  return Verify(mod_syslib, name, M, N, K, MR, NR, LS, MC, NC, KC, layer);
}



int main(int argc, char * argv []) {
  
  int M=atoi(argv[1]);
  int N=atoi(argv[2]);
  int K=atoi(argv[3]);
  int MR=atoi(argv[4]);
  int NR=atoi(argv[5]);
  int LS=atoi(argv[6]);

  double gf;
  

  int MC=896, NC=3072, KC=512;
if (NR > LS && NR%LS !=0){
    printf("Skiping it!");
    return 0;
  }
  gf= DeploySingleOp(M,N,K,MR,NR,LS, MC, NC, KC, -1);
  printf("Layer %d GFLOPS %f  %dx%d\n",-1,gf, MR,NR);
  return 0;
  //resnet50
  int layers=20;
  double gflops_max = 0.0;
  int mr_max = 4;
  int nr_max = 4;
  int mm[layers]={1605632, 401408,  401408,  401408,  401408,  401408,  100352,  100352 , 100352 , 100352,  100352,  25088 ,  25088,   25088 ,  25088,   25088 ,  6272,    6272,    6272,    6272};
  int nn[layers]={64,  64,      64,      256,     64,      128,     128,     512,     512,     128,     256,     256 ,    1024,    1024,    256,     512,     512,     2048,    2048,    512};
  int kk[layers]={147, 64 ,     576,     64,      256 ,    256  ,   1152,    128 ,    256 ,    512 ,    512,     2304 ,   256 ,    512 ,    1024 ,   1024,    4608,    512,     1024 ,   2048};
 int linesize[3]={4, 8 ,16};
 int mrnr[8]={4, 8, 12, 16, 20, 24, 28, 32};
printf("#Layer  M   N   K   MC   NC  KC  MR  NR   LS   GFLOPS\n");
 for(int ls=0; ls < 3;ls++){
	 for(int l=0;l<20;l++){
		 for(int mr=0;mr<8;mr++){
			 for(int nr=0;nr<8;nr++){
				if ( mrnr[nr] > linesize[ls] && mrnr[nr]%linesize[ls] !=0) continue;
  				gf= DeploySingleOp(mm[l],nn[l],kk[l],mrnr[mr],mrnr[nr],linesize[ls], MC, NC, KC, l);
				if (gf >= gflops_max){
				    gflops_max=gf;
				    mr_max = mrnr[mr];
				    nr_max= mrnr[nr];
				}
			 }
		 }
		 printf("Layer %d GFLOPS %f  %dx%d\n",l+1,gflops_max, mr_max,nr_max);
		 gflops_max=0.0;
		 mr_max=4;
		 nr_max=4;
	 }
 }
  //DeploySingleOp(M,N,K,MR,NR,LS);
  return 0;
}
