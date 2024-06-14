import numpy as np
import tvm
import os
from tvm import te, auto_scheduler
import sys
import math

from cache_model import *

def eg_matmul(M, N, K, mc, nc, kc, mr, nr, lane, laneC, dtypeA, dtypeB, dtypeC, target, target2):
    A = te.placeholder((M, K), name="A", dtype=dtypeA)
    B = te.placeholder((K, N), name="B", dtype=dtypeB)
    C = te.placeholder((M, N), name="C", dtype=dtypeC)
    k = te.reduce_axis((0, K), name="k")
    zero = te.var(name='zero', dtype=dtypeC, span=None)
    unroll_factor=4
    
    if dtypeA != dtypeC:
        if M % mr != 0:
             Ac = te.compute(( math.ceil(K/kc), math.ceil(M/mr), kc, mr),
                     lambda am, an, az, al:
                     te.if_then_else(te.any( an * mr + al >=M, am * kc + az >= K),
                         zero,
                         A[ an * mr + al, am * kc + az  ].astype(dtypeC))
                     , name='Ac')
        else:
             Ac = te.compute(( math.ceil(K/kc), math.ceil(M/mr), kc, mr),
                lambda am, an, az, al: A[ an * mr + al, am * kc + az  ].astype(dtypeC)
                , name='Ac')
    else:
        if M % mr != 0:
             Ac = te.compute(( math.ceil(K/kc), math.ceil(M/mr), kc, mr),
                     lambda am, an, az, al:
                     te.if_then_else(te.any( an * mr + al >=M, am * kc + az >= K),
                         zero,
                         A[ an * mr + al, am * kc + az  ])
                     , name='Ac')
        else:
             Ac = te.compute(( math.ceil(K/kc), math.ceil(M/mr), kc, mr),
                lambda am, an, az, al: A[ an * mr + al, am * kc + az  ]
                , name='Ac')
    
    if dtypeB != dtypeC:
        if N % nr != 0:
            Bc = te.compute(( math.ceil(K/kc), math.ceil(N/nr), kc, nr),
                lambda bm, bn, bz, bl:
                te.if_then_else(te.any(bm*kc+bz>= K, bn*nr+bl>=N),
                    zero,
                    B[ bm * kc + bz, bn * nr + bl ].astype(dtypeC))
                , name='Bc')
        else:
             Bc = te.compute(( math.ceil(K/kc), math.ceil(N/nr), kc, nr),
                lambda bm, bn, bz, bl: B[ bm * kc + bz, bn * nr + bl ].astype(dtypeC)
                , name='Bc')
    else:
        if N % nr != 0:
            Bc = te.compute(( math.ceil(K/kc), math.ceil(N/nr), kc, nr),
                lambda bm, bn, bz, bl:
                te.if_then_else(te.any(bm*kc+bz>= K, bn*nr+bl>=N),
                    zero,
                    B[ bm * kc + bz, bn * nr + bl ])
                , name='Bc')
        else:
             Bc = te.compute(( math.ceil(K/kc), math.ceil(N/nr), kc, nr),
                lambda bm, bn, bz, bl: B[ bm * kc + bz, bn * nr + bl ]
                , name='Bc')

        
    ac = te.compute(( math.ceil(K/kc), math.ceil(M/mr), kc, mr), lambda am, an, az, al: Ac[ am, an, az, al], name='ac')
    bc = te.compute(( math.ceil(K/kc), math.ceil(N/nr), kc, nr), lambda bm, bn, bz, bl: Bc[ bm, bn, bz, bl ], name='bc')

    matmul = te.compute(
            (M, N),
            lambda i, j:
            te.sum(
                ac[k//kc, i//mr,tvm.tir.indexmod(k,kc), tvm.tir.indexmod(i,mr) ]
                *
                bc[k//kc, j//nr,tvm.tir.indexmod(k,kc), tvm.tir.indexmod(j,nr) ]
                ,
                axis=k),
            name="matmul",
            )

    s = te.create_schedule(matmul.op)
     
    ic, jc, icin, jcin= s[matmul].tile(matmul.op.axis[0], matmul.op.axis[1], mc, nc)
    
    ir, it = s[matmul].split( icin, factor = mr )
    jr, jt = s[matmul].split( jcin, factor = nr )
    
    p, = s[matmul].op.reduce_axis
    pc, pr = s[matmul].split(p, factor=kc)
     
    s[matmul].reorder(jc, pc, ic, jr, ir, pr, it, jt)
     
    s[matmul].unroll(it)
    jto, jti = s[matmul].split(jt, factor=laneC)
    s[matmul].vectorize(jti)
    s[matmul].unroll(jto)
     
    s[Ac].compute_at(s[matmul],ic)
    s[Bc].compute_at(s[matmul],pc)
    s[ac].compute_at(s[matmul],pr)
    s[bc].compute_at(s[matmul],pr)

    b0, b1, b2, b3 = Bc.op.axis
    b30, b31 = s[Bc].split(b3, factor=lane)
    s[Bc].vectorize(b31)
    b300, b301 = s[Bc].split(b30, factor=4)
    s[Bc].unroll(b301)
     
    a0, a1, a2, a3 = Ac.op.axis
    a30, a31 = s[Ac].split(a3, factor=lane)
    s[Ac].vectorize(a31)
    a300, a301 = s[Ac].split(a30, factor=4)
    s[Ac].unroll(a301)
     
    a0, a1, a2, a3 = ac.op.axis
    a30, a31 = s[ac].split(a3, factor=laneC)
    s[ac].unroll(a30)
    s[ac].vectorize(a31)
     
    b0, b1, b2, b3 = bc.op.axis
    b30, b31 = s[bc].split(b3, factor=laneC)
    s[bc].unroll(b30)
    s[bc].vectorize(b31)

    s[matmul].pragma(jc, "auto_unroll_max_step", 64)
    s[matmul].pragma(jc, "unroll_explicit", True)
    
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        f = tvm.lower(s, [A, B, matmul,zero], name="gemm", simple_mode=False)
        print(f)
        name="eg_matmul_{}_{}_{}_{}_{}".format(M,N,K, mr, nr)
        func = tvm.build(f, target=target)
        func.save("asm/{}.s".format(name), 's')
        curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
        base_path=os.path.join(curr_path, "lib")
        func_save = tvm.build(s, [A, B, C], target=target2, name=name)
        syslib_path = os.path.join(base_path, name+".o")
        func_save.save(syslib_path)
        return func

@auto_scheduler.register_workload
def matmul_blis(M, N, K,  mc, nc, kc, mr, nr, dtypeA, dtypeB, dtypeC, test):
    A = te.placeholder((M, K), name="A", dtype=dtypeA)
    B = te.placeholder((K, N), name="B", dtype=dtypeB)
    k = te.reduce_axis((0, K), name="k")
    zero = te.var(name='zero', dtype=dtypeC, span=None)
    if dtypeA != dtypeC:
        if M % mr != 0:
             Ac = te.compute(( math.ceil(K/kc), math.ceil(M/mr), kc, mr),
                     lambda am, an, az, al:
                     te.if_then_else(te.any( an * mr + al >=M, am * kc + az >= K),
                         zero,
                         A[ an * mr + al, am * kc + az  ].astype(dtypeC))
                     , name='Ac')
        else:
             Ac = te.compute(( math.ceil(K/kc), math.ceil(M/mr), kc, mr),
                lambda am, an, az, al: A[ an * mr + al, am * kc + az  ].astype(dtypeC)
                , name='Ac')
    else:
        if M % mr != 0:
             Ac = te.compute(( math.ceil(K/kc), math.ceil(M/mr), kc, mr),
                     lambda am, an, az, al:
                     te.if_then_else(te.any( an * mr + al >=M, am * kc + az >= K),
                         zero,
                         A[ an * mr + al, am * kc + az  ])
                     , name='Ac')
        else:
             Ac = te.compute(( math.ceil(K/kc), math.ceil(M/mr), kc, mr),
                lambda am, an, az, al: A[ an * mr + al, am * kc + az  ]
                , name='Ac')
    if dtypeB != dtypeC:
        if N % nr != 0:
            Bc = te.compute(( math.ceil(K/kc), math.ceil(N/nr), kc, nr),
                lambda bm, bn, bz, bl:
                te.if_then_else(te.any(bm*kc+bz>= K, bn*nr+bl>=N),
                    zero,
                    B[ bm * kc + bz, bn * nr + bl ].astype(dtypeC))
                , name='Bc')
        else:
             Bc = te.compute(( math.ceil(K/kc), math.ceil(N/nr), kc, nr),
                lambda bm, bn, bz, bl: B[ bm * kc + bz, bn * nr + bl ].astype(dtypeC)
                , name='Bc')
    else:
        if N % nr != 0:
            Bc = te.compute(( math.ceil(K/kc), math.ceil(N/nr), kc, nr),
                lambda bm, bn, bz, bl:
                te.if_then_else(te.any(bm*kc+bz>= K, bn*nr+bl>=N),
                    zero,
                    B[ bm * kc + bz, bn * nr + bl ])
                , name='Bc')
        else:
             Bc = te.compute(( math.ceil(K/kc), math.ceil(N/nr), kc, nr),
                lambda bm, bn, bz, bl: B[ bm * kc + bz, bn * nr + bl ]
                , name='Bc')

    ac = te.compute(( math.ceil(K/kc), math.ceil(M/mr), kc, mr), lambda am, an, az, al: Ac[ am, an, az, al], name='ac')
    
    bc = te.compute(( math.ceil(K/kc), math.ceil(N/nr), kc, nr), lambda bm, bn, bz, bl: Bc[ bm, bn, bz, bl ], name='bc')
    
    matmul = te.compute(
            (M, N),
            lambda i, j:
            te.sum(
                ac[k//kc, i//mr,tvm.tir.indexmod(k,kc), tvm.tir.indexmod(i,mr) ]
                *
                bc[k//kc, j//nr,tvm.tir.indexmod(k,kc), tvm.tir.indexmod(j,nr) ]
                ,
                axis=k),
            name="matmul",
            )
        
    return [A, B, matmul,zero]

@auto_scheduler.register_workload
def matmul_add(M, N, K, dtypeA, dtypeB, dtypeC, test):
    A = te.placeholder((M, K), name="A", dtype=dtypeA)
    B = te.placeholder((K, N), name="B", dtype=dtypeB)
    C = te.placeholder((M, N), name="C", dtype=dtypeC)
    zero = te.var(name='zero', dtype=dtypeC, span=None)
    
    k = te.reduce_axis((0, K), name="k")
    if dtypeA == dtypeB == dtypeC:
        matmul = te.compute(
            (M, N),
            lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
            name="matmul",
            attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
            )
    else:
        matmul = te.compute(
            (M, N),
            lambda i, j: te.sum((A[i, k].astype(dtypeC) * B[k, j].astype(dtypeC)), axis=k),
            #lambda i, j: te.sum(((A[i, k] * B[k, j]).astype(dtypeC)), axis=k),
            name="matmul",
            attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
            )
    #out = te.compute((M, N), lambda i, j: matmul[i, j] + C[i, j], name="out")
            
    #return [A, B, C, out]
    return [A, B, matmul]


def main(M,N,K,test,trials, blis=0, eg=0, cfg="carmel"):
    if cfg == "carmel":
        target = tvm.target.Target("llvm  -device=arm_cpu -mattr=+v8.2a,+fp-armv8,+neon,+fp16fml,+fullfp16")
        target2 = tvm.target.Target("llvm  --system-lib -device=arm_cpu -mattr=+v8.2a,+fp-armv8,+neon,+fp16fml,+fullfp16")
    else:
        target = tvm.target.Target("llvm")
        target = tvm.target.Target("llvm --system-lib")

    if test == 1:
        typeA="float32"
        typeB="float32"
        typeC="float32"
    elif test == 2:
        typeA="float16"
        typeB="float16"
        typeC="float32"
    elif test == 3:
        typeA="int8"
        typeB="int8"
        typeC="float32"
    elif test == 4:
        typeA="int8"
        typeB="int8"
        typeC="float16"
    elif test == 5:
        typeA="int8"
        typeB="int8"
        typeC="int32"
    elif test == 6:
        typeA="int8"
        typeB="int8"
        typeC="int16"
    else:
        typeA="float16"
        typeB="float16"
        typeC="float16"
    
    if typeA == "float32":
        a_np = np.random.uniform(size=(M, K)).astype(np.float32)
    elif typeA == "float16":
        a_np = np.random.uniform(size=(M, K)).astype(np.float16)
    elif typeA == "int8":
        a_np = np.random.uniform(size=(M, K)).astype(np.int8)
    else:
        a_np = np.random.uniform(size=(M, K)).astype(np.int32)

    if typeB == "float32":
        b_np = np.random.uniform(size=(K, N)).astype(np.float32)
    elif typeB == "float16":
        b_np = np.random.uniform(size=(K, N)).astype(np.float16)
    elif typeB == "int8":
        b_np = np.random.uniform(size=(K, N)).astype(np.int8)
    else:
        b_np = np.random.uniform(size=(K, N)).astype(np.int32)
    
    if typeC == "float32":
        out_np = np.zeros((M, N)).astype(np.float32)
        zero=np.float32(0) if typeC == 'float32' else np.int32(0)
    
    elif typeC == "float16":
        out_np = np.zeros((M, N)).astype(np.float16)
        zero=np.float16(0) if typeC == 'float16' else np.int16(0)
    
    elif typeC == "int8":
        out_np = np.zeros((M, N)).astype(np.int8)
        zero=np.float8(0) if typeC == 'float8' else np.int8(0)
    
    elif typeC == "int16":
        out_np = np.zeros((M, N)).astype(np.int16)
        zero=np.float16(0) if typeC == 'float16' else np.int16(0)
    
    else:
        zero=np.float32(0) if typeC == 'float32' else np.int32(0)
        out_np = np.zeros((M, N)).astype(np.int32)
    
    print("{} = {} * {}".format(out_np.dtype, a_np.dtype, b_np.dtype))
    dev = tvm.cpu()
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, dtype=typeC, device=dev)
    
    mr = nr = mc = nc = kc = -1
    if eg == 1:
        if typeA in ['float16','int16']:
            lane = 8
        elif typeA in ['float8','int8']:
            lane = 16
        else:
            lane = 4
        
        if typeC in ['float16','int16']:
            laneC = 8
            datasize = 2
            maxm = min(65,M+1)
            maxn = min(65,N+1)
            stride = ini = 8
        elif typeC in ['float8','int8']:
            laneC = 16
            datasize = 1
            maxm = min(129,M+1)
            maxn = min(129,N+1)
            stride = ini = 16
        else:
            maxm = min(33,M+1)
            maxn = min(33,N+1)
            laneC = 4
            datasize = 4
            stride = ini =4
        best=-1
        bestmr=-1
        bestnr=-1
        cfg_file=cfg+".cfg"
        for mr in range(ini,maxm,stride):
            for nr in range(ini,maxn,stride):
                mc, nc, kc = get_optim_mc_nc_kc(datasize,M,N,K,mr,nr,cfg_file)
                matmul = eg_matmul(M, N, K, mc, nc, kc, mr, nr, lane, laneC, typeA, typeB, typeC, target)
                matmul(a_tvm, b_tvm, out_tvm,zero)

                evaluator = matmul.time_evaluator(matmul.entry_name, dev, min_repeat_ms=1500)
                tt = np.median(evaluator(a_tvm, b_tvm, out_tvm,zero).results)
                gflops = ((2.0 * M * N * K)/(1e9*1.0))/tt
                if gflops > best:
                    best = gflops
                    bestmr=mr
                    bestnr=nr
                print("test: {} {} {} {} {} {} {} {}: {} gflops".format(M,N,K,mr,nr,mc, nc, kc, gflops))
        print("Best: {} {} {} {} {}: {}".format(M,N,K,bestmr,bestnr,best))
    else:    
        if blis == 1:
            mr=8
            nr=32
            mc, nc, kc = get_optim_mc_nc_kc(datasize,M,N,K,mr,nr,cfg_file)
            task = tvm.auto_scheduler.SearchTask(func=matmul_blis, args=(M, N, K, mc, nc, kc, mr, nr, typeA, typeB, typeC, test), target=target)
        else:
            task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(M, N, K, typeA, typeB, typeC, test), target=target)
        folder=typeA+typeB+typeC
        # Inspect the computational graph
        print("Computational DAG:")
        print(task.compute_dag)
        trials=trials
        log_file = "auto_{}_{}{}/matmul_{}_{}_{}.json".format(trials,folder,"_blis" if blis == 1 else "",M,N,K)
        if os.path.isfile(log_file) == False:
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=trials,
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                verbose=0,
                )

            # Run auto-tuning (search)
            task.tune(tune_option)
        # Apply the best schedule
        sch, args = task.apply_best(log_file)

        print("Lowered TIR:")
        print(tvm.lower(sch, args, simple_mode=True))
    
        func = tvm.build(sch, args, target)
    
        
        func.save("auto_{}_{}{}/{}_{}_{}.s".format(trials,folder,"_blis" if blis == 1 else "",M,N,K), 's')
    #try:
    #    func.save("{}/{}_{}_{}.c".format(folder,M,N,K), 'c')
    #except:
    #    pass
    # Check results
    # Check results
    #np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

        # Evaluate execution time.
        evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=1500)
        #print("AUTO", M,N,K)
        time = np.median(evaluator(a_tvm, b_tvm, out_tvm).results) #* 1000
        gflops = ((2.0 * M * N * K)/(1e9*1.0))/time
        print(
            "{} {} {}: {} s {} gflops".format(M,N,K,time, gflops) #(np.median(evaluator(a_tvm, b_tvm, out_tvm).results) * 1000)            #% (np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results) * 1000)
    )
    print("Equivalent python schedule:")
    print(task.print_best(log_file))

if __name__ == "__main__":
    bs = 1 
    eg = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    test = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    trials = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    blis = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    google = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    cfg = sys.argv[6] if len(sys.argv) > 6 else "carmel"
    MNK=[
            [12544*bs, 64, 147], 
            [3136*bs, 64, 64],
            [3136*bs, 64, 576], 
            [3136*bs, 256, 64],
            [3136*bs, 64, 256],
            [3136*bs, 128, 256],
            [784*bs, 128, 1152],
            [784*bs, 512, 128],
            [784*bs, 512, 256],
            [784*bs, 128, 512],
            [784*bs, 256, 512],
            [196*bs, 256, 2304],
            [196*bs, 1024, 256],
            [196*bs, 1024, 512], 
            [196*bs, 256, 1024],
            [196*bs, 512, 1024], 
            [49*bs, 512, 4608],  
            [49*bs, 2048, 512], 
            [49*bs, 2048, 1024],
            [49*bs, 512, 2048], 
            #[1000, 1000, 1000], 
            #[2000, 2000, 2000], 
            #[3000, 3000, 3000], 
            #[4000, 4000, 4000], 
            #[5000, 5000, 5000], 
            ]
    #MNK = [[49*bs, 2048, 512]]
    #MNK = [[2048, 49, 512]]
    MNK =  [[12544, 64, 147]] 
    MNK = [[1024,1024,1024]]
    if google != 0:
        import googlenet as gl
        MNK = gl.googlenet(bs)
    for M, N, K in MNK:
       main(M,N,K, test,trials, blis, eg,cfg)
