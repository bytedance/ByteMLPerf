import torch
import numpy as np
import time

from ixgemmblaslt import gemm88

#shape_0 = 1
# shape_m = 3072
# shape_n = 4096
# shape_k = 30176

shape_0 = 1
shape_m = 4
shape_n = 8
shape_k = 4

np.random.seed(int(time.time()))

for kk in range(0, 2):
    alist = []
    blist = []
    clist = []
    begini = -5
    endi = 5
    for ii in range(0,3):
        #arr1 = np.random.randint(begini, endi, (shape_0, shape_m, shape_k))
        arr1 = np.random.randint(begini, endi, (shape_m, shape_k))
        t1 = torch.from_numpy(arr1).to(torch.int8).to("cuda")
        alist.append(t1)

        #arr2 = np.random.randint(begini, endi, (shape_0, shape_k, shape_n))
        arr2 = np.random.randint(begini, endi, (shape_k, shape_n))
        t2 = torch.from_numpy(arr2).to(torch.int8).to("cuda")
        blist.append(t2)

    blasLtIns = gemm88.gemm_init()

    begin_t = int(time.time() * 1000)
    clist11 = gemm88.gemm_run(blasLtIns, alist, blist)
    end_t = int(time.time() * 1000)

    #print("clist11:", clist11)

    gemm88.gemm_release(blasLtIns)

    alist2 = []
    blist2 = []
    clist2 = []
    for a, b in zip(alist, blist):
        a2 = a.clone().to(torch.float32)
        alist2.append(a2)
        b2 = b.clone().to(torch.float32)
        blist2.append(b2)
        c_shape = (a2.shape[0], b2.shape[1])
        zeros_tensor = torch.zeros(c_shape).to("cuda").to(torch.float32)
        clist2.append(zeros_tensor)

    begin_t2 = int(time.time() * 1000)
    #clist2 = [a @ b for a, b in zip(alist2, blist2)]
    clist2 = [torch.matmul(a,b) for a, b in zip(alist2, blist2)]
    end_t2 = int(time.time() * 1000)
    #print("clist2:", clist2)

    print("cost time:", end_t - begin_t, "; ", end_t2 - begin_t2)


    ball = True
    
    for c, c2 in zip(clist11, clist2):
        c1 = c.to(torch.float32)
        ball = torch.allclose(c1, c2, rtol=1e-4, atol=1e-6)
        if not ball:
            break

    print("\n")
    if ball:
        print("***all is ok***")
    else:
        print("??? not all is ok !!!")

