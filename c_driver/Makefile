# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Makefile Example to deploy TVM modules.
TVM_ROOT=/home/adcastel/src/tvm_riscv
#TVM_ROOT=/Users/adcastel/opt/tvm
#TVM_ROOT=${TVM_HOME}

DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core

PKG_CFLAGS = -std=c++14 -O3 -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>

CXX=g++
#CXX=clang++-10
PKG_LDFLAGS = -L${TVM_ROOT}/build -ldl -pthread
OBJS=$(wildcard lib/*.o)
.PHONY: clean all

MICRO = lib/gemm_$(M)_$(N)_$(K)_$(MR)_$(NR)_$(LS).o \
	libtvm_runtime_pack.o
#
#all: start test_driver_c run
all: test_driver_c

start:
	mkdir -p lib
	python3 B3A2C0.py --M $(M) --N $(N) --K $(K) --MC $(MC) --NC $(NC) --KC $(KC) --MR $(MR) --NR $(NR) --KR $(KR) --LS $(LS)

run:
	./test_driver_c $(M) $(N) $(K) $(MR) $(NR) $(LS)

# Build rule for all in one TVM package library
libtvm_runtime_pack.o: tvm_runtime_pack.cc
	@mkdir -p $(@D)
	$(CXX) -c $(PKG_CFLAGS) -o $@  $^

# The code library built by TVM
#lib/test_addone_sys.o: prepare_test_libs.py
#	@mkdir -p $(@D)
#	python3 prepare_test_libs.py

# Deploy using the all in one TVM package library
#lib/cpp_deploy_pack: cpp_deploy.cc lib/test_addone_sys.o lib/libtvm_runtime_pack.o
#	@mkdir -p $(@D)
#	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)

# Deploy using pre-built libtvm_runtime.so
#test_driver_c: cpp_deploy.cc $(MICRO) 
test_driver_c: cpp_deploy.cc $(OBJS) libtvm_runtime_pack.o 
	touch cpp_deploy.cc
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)
#$(CXX) $(PKG_CFLAGS) -o $@  $^ -ltvm_runtime $(PKG_LDFLAGS)
clean:
	rm -rf lib/* build/* test_driver_c *.s *.o