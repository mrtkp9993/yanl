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


def nelderMead(f, x0, s=0.1, maxiter=100, eps=1e-6, debug=True):
    raise "Not implemented yet"
