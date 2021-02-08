import math

res = [0, 0, 0, 0]
b = []
for j in range(7):
    # a.append(0)
    b.append(0)

a = [
    [1, 1, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 1]
]

res_to_ress = dict()
res_to_num_1 = dict()

def to_num(aa):
    mull = 1
    ress = 0
    for jj in range(len(aa) - 1, -1, -1):
        ress += aa[jj] * mull
        mull *= 2
    return ress

def mult(aa, bb):
    alll = 0
    for j in range(len(aa)):
        alll += aa[j] * bb[j]
    return alll % 2

for j in range(128):
    tmp = j

    num_1 = 0
    for _ in range(7):
        b[6 - _] = tmp % 2
        num_1 += tmp % 2
        tmp = tmp // 2
    print(j, b)

    for _ in range(4):
        res[_] = mult(b, a[_])
    print(res)
    ress = to_num(res)
    if ress not in res_to_ress:
        res_to_num_1[ress] = num_1
        res_to_ress[ress] = b.copy()
    elif res_to_num_1[ress] < num_1:
        res_to_num_1[ress] = num_1
        res_to_ress[ress] = b.copy()

for ress in res_to_ress:
    print(ress, res_to_ress[ress], res_to_num_1[ress])

