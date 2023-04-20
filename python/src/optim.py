import numpy as np


def secant(f, a, b, maxiter=100, eps=1e-6, debug=True):
    fa = f(a)
    fb = f(b)

    if np.abs(fa) > np.abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    if debug:
        print(f"Iter: 0\ta={a:.4f}\tfa={fa:.4f}")
        print(f"Iter: 1\tb={b:.4f}\tfb={fb:.4f}")

    conv = False
    for i in range(2, maxiter + 1):
        if np.abs(fa) > np.abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        d = (b - a) / (fb - fa)
        b = a
        fb = fa
        d = d * fa
        if np.abs(d) < eps:
            print("CONVERGENCE")
            conv = True
            break
        a = a - d
        fa = f(a)
        print(f"Iter: {i}\ta={a:.6f}\tfa={fa:.6f}")

    return a, fa, conv


def powell(f, x0, h=0.1, maxiter=100, eps=1e-6, debug=True):
    raise "Not implemented yet"


def nelderMead(
    f, x0: np.array, step=0.1, maxiter=0, noimprviter=10, eps=1e-6, debug=True
):
    dim = len(x0)
    prev_best = f(x0)
    res = [[x0, prev_best]]
    alpha = 1
    gamma = 2
    rho = 0.5
    sigma = 0.5
    no_improv = 0

    for i in range(dim):
        x = x0.copy()
        x[i] += step
        score = f(x)
        res.append([x, score])

    if debug:
        print("Initial Simplexes")
        for r in res:
            print(f"Simplex: {r[0]}\tScore: {r[1]:.6f}")

    iters = 0
    fevals = 2
    while True:
        res.sort(key=lambda x: x[1])
        best_score = res[0][1]

        if maxiter and iters >= maxiter:
            if debug:
                print(f"{fevals} function evalutions")
                print(f"{iters} iterations")
                print(f"Minimum was found at {res[0][0]}, value {res[0][1]:.6f}")
            return res[0]
        iters += 1

        if best_score < prev_best - eps:
            no_improv = 0
            prev_best = best_score
        else:
            no_improv += 1

        if no_improv > noimprviter:
            if debug:
                print(f"{fevals} function evalutions")
                print(f"{iters} iterations")
                print(f"Minimum was found at {res[0][0]}, value {res[0][1]:.6f}")
            return res[0]

        xc = [0.0] * dim
        for t in res[:-1]:
            for i, c in enumerate(t[0]):
                xc[i] += c / (len(res) - 1)

        xr = xc + alpha * (xc - res[-1][0])
        rscore = f(xr)
        fevals += 1
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        if rscore < res[0][1]:
            xe = xc + gamma * (xc - res[-1][0])
            escore = f(xe)
            fevals += 1
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        xco = xc + rho * (xc - res[-1][0])
        cscore = f(xco)
        fevals += 1
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xco, cscore])

        xre = res[0][0]
        nres = []
        for t in res:
            redx = xre + sigma * (t[0] - xre)
            score = f(redx)
            fevals += 1
            nres.append([redx, score])
        res = nres
