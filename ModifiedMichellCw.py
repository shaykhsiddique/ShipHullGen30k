# -*- coding: utf-8 -*-
"""
Updated on Wed Mar 18 11:47:31 2026
@updated by: shaykhsiddique
 -> Updated the exception handling with extreme values 


Created on Mon Nov 14 20:17:37 2022

@author: nbagz

This code is based off of sample code provided in the PhD Defense of Douglas Read
from the University of Maine (2009):

    'A Drag Estimate for Concept Stage Ship Design Optimization'

The goal of this code is to use Michell Thin Ship Theory to predict the wave drag
from a ship progressing at a steady speed U.

Inputs:
    Y   -> hull offsets at scale, indexed as [X_idx, Z_idx]
    U   -> speed of the hull
    X   -> vector of x positions on the ship, must be odd in length and uniform
    Z   -> vector of z positions from 0 to -Draft, uniformly distributed
    RHO -> density of water / fluid
    N   -> number of angles for numerical integration

This version fixes numerical overflow in the Z-integral by combining the Filon
weights with the exponential term before evaluation.
"""

import numpy as np

g = 9.81  # gravity constant


def ModMichell(Y, U, X, Z, RHO, N):
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)

    Nx = len(Y)       # number of stations
    Nz = len(Y[0])    # number of waterlines

    if Nx % 2 == 0:
        print('Nx must be odd.')  # required for x Filon algorithm
        return -1

    if len(X) != Nx:
        raise ValueError("Length of X must match number of rows in Y.")

    if len(Z) != Nz:
        raise ValueError("Length of Z must match number of columns in Y.")

    # Ensure Z runs from 0 to -Draft, as expected by the method
    if Z[0] < Z[-1]:
        Z = Z[::-1]
        Y = Y[:, ::-1]

    # ------ integration variables ---------
    dz = Z[1] - Z[0]
    L = X[-1]
    dx = L / (Nx - 1)

    # theta spacing, slightly clipped away from pi/2 to avoid sec(theta) blowup
    theta = michspace(N)
    theta = np.minimum(theta, np.pi / 2 - 1e-6)

    k0 = g / U**2.0
    c = (4.0 * RHO * U**2.0) / np.pi
    a = 1.0 / np.cos(theta)     # sec(theta)
    k = k0 * a**2.0

    # ---------- Z INTEGRAL ----------------
    # Stable formulation: do not compute exp(Kz) separately from exp(k0*Z*a^2)

    f = np.zeros((Nz,), dtype=np.float64)
    F = np.zeros((Nx, N), dtype=np.float64)

    for j in range(N):
        aj2 = a[j]**2.0
        K = k0 * dz * aj2
        K2 = K * K

        # fallback in the unlikely case K is extremely small
        if abs(K) < 1e-14:
            for m in range(Nx):
                for n in range(Nz):
                    s = k0 * Z[n] * aj2
                    es = np.exp(np.clip(s, -700.0, 700.0))
                    f[n] = Y[m, n] * es * dz
                F[m, j] = np.sum(f)
            continue

        for m in range(Nx):
            for n in range(Nz):
                s = k0 * Z[n] * aj2

                # Clip exponent arguments into a safe floating-point range
                es = np.exp(np.clip(s, -700.0, 700.0))
                esp = np.exp(np.clip(s + K, -700.0, 700.0))
                esm = np.exp(np.clip(s - K, -700.0, 700.0))

                if n == 0:
                    term = (esp - es - K * es) / K2
                elif n == Nz - 1:
                    term = (esm - es + K * es) / K2
                else:
                    term = (esp + esm - 2.0 * es) / K2

                f[n] = Y[m, n] * dz * term

            F[m, j] = np.sum(f)

    # ---------- X INTEGRAL -----------------
    Kx = k0 * dx * a

    # Avoid divide-by-zero issues if Kx is tiny
    Kx_safe = np.where(np.abs(Kx) < 1e-14, 1e-14, Kx)

    alp = (Kx_safe**2 + 0.5 * Kx_safe * np.sin(2.0 * Kx_safe) + np.cos(2.0 * Kx_safe) - 1) / Kx_safe**3.0
    bet = (3.0 * Kx_safe + Kx_safe * np.cos(2.0 * Kx_safe) - 2.0 * np.sin(2.0 * Kx_safe)) / Kx_safe**3.0
    gam = 4.0 * (np.sin(Kx_safe) - Kx_safe * np.cos(Kx_safe)) / Kx_safe**3.0

    Nev = int((Nx + 1) / 2)   # even Filon index count
    Nod = int((Nx - 1) / 2)   # odd Filon index count

    pev = np.zeros((Nev,), dtype=np.float64)
    qev = np.zeros((Nev,), dtype=np.float64)
    pod = np.zeros((Nod,), dtype=np.float64)
    qod = np.zeros((Nod,), dtype=np.float64)

    Pt = np.zeros((N,), dtype=np.float64)
    Qt = np.zeros((N,), dtype=np.float64)
    Pev = np.zeros((N,), dtype=np.float64)
    Qev = np.zeros((N,), dtype=np.float64)
    Pod = np.zeros((N,), dtype=np.float64)
    Qod = np.zeros((N,), dtype=np.float64)
    P = np.zeros((N,), dtype=np.float64)
    Q = np.zeros((N,), dtype=np.float64)

    for j in range(N):
        for m in range(Nev):
            pev[m] = F[2 * m, j] * np.cos(k0 * X[2 * m] * a[j])
            qev[m] = F[2 * m, j] * np.sin(k0 * X[2 * m] * a[j])

        for m in range(Nod):
            pod[m] = F[2 * m + 1, j] * np.cos(k0 * X[2 * m + 1] * a[j])
            qod[m] = F[2 * m + 1, j] * np.sin(k0 * X[2 * m + 1] * a[j])

        Pt[j] = F[-1, j] * np.cos(k0 * L * a[j])
        Qt[j] = F[-1, j] * np.sin(k0 * L * a[j])

        Pev[j] = np.sum(pev) - 0.5 * Pt[j]
        Pod[j] = np.sum(pod)
        Qev[j] = np.sum(qev) - 0.5 * Qt[j]
        Qod[j] = np.sum(qod)

        P[j] = dx * (alp[j] * Qt[j] + bet[j] * Pev[j] + gam[j] * Pod[j])
        Q[j] = dx * (-alp[j] * Pt[j] + bet[j] * Qev[j] + gam[j] * Qod[j])

    R = c * k**2.0 / (a**3.0) * (
        k**2.0 * (P**2.0 + Q**2.0)
        + 2.0 * k * a * (Q * Pt - P * Qt)
        + a**2.0 * (Pt**2.0 + Qt**2.0)
    )

    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------- THETA INTEGRAL -------------
    rw = np.zeros((N - 1,), dtype=np.float64)

    for jj in range(N - 1):
        rw[jj] = 0.5 * (R[jj] + R[jj + 1]) * (theta[jj + 1] - theta[jj])

    return np.sum(rw)


def michspace(N):
    """
    MICHSPACE log spacing for Michell integral

    MICHSPACE(N) produces log base 10 spacing over N propagation
    angles between 0 and pi/2. Points are more closely spaced near pi/2.
    """
    xm = (np.logspace(0, 1, N, base=10.0) - 1.0) * np.pi / 18 - np.pi / 2
    xm = -xm[::-1]
    return xm


def CalcDrag(U, LOA, WL, CW, T, SA, rho=1025.0):
    """
    Parameters
    ----------
    U : Ship Speed in m/s
    LOA : Length Overall in m
    WL : Waterline length in m at draft mark T
    CW : Vector of wave drag calculations [size 32]
    T : Fraction of Dd that is draft of hull
    SA : Wetted surface area of hull in m^2 at draft mark T
    rho : density of water in kg/m^3

    Returns
    -------
    Total drag = wave drag + skin friction drag
    and the Froude number
    """
    Cf = Calc_Cf(U=U, WL=WL)
    Rf = 0.5 * Cf * rho * SA * U**2.0

    Fn = U / np.sqrt(g * WL)
    Cw = interp_CW(Fn=Fn, T=T, CW=CW)
    Rw = 0.5 * Cw * rho * (LOA**2.0) * (U**2.0)

    return (Rw + Rf), Fn


def Calc_Cf(U, WL, v=1.19 * 10.0**-6.0):
    """
    Calculates the skin friction coefficient based on the 1957 ITTC line.
    """
    Re = U * WL / v
    Cf = 0.075 / (np.log10(Re) - 2.0)**2.0
    return Cf


def interp_CW(
    Fn,
    T,
    CW,
    Fn_list=np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]),
    T_list=np.array([0.25, 0.33, 0.5, 0.67])
):
    """
    Interpolates wave drag coefficient based on Froude number and draft fraction.
    """
    Cw_tab = np.reshape(CW, (len(T_list), len(Fn_list)))

    Fn_idx = 0
    Fn_frac = 0.0
    T_idx = 0
    T_frac = 0.0

    if Fn >= Fn_list[-1]:
        Fn_idx = len(Fn_list) - 2
        Fn_frac = 1.0
    elif Fn <= Fn_list[0]:
        Fn_idx = 0
        Fn_frac = 0.0
    else:
        arr = np.where(Fn_list < Fn)[0]
        Fn_idx = arr[-1]
        Fn_frac = (Fn - Fn_list[Fn_idx]) / (Fn_list[Fn_idx + 1] - Fn_list[Fn_idx])

    if T >= T_list[-1]:
        T_idx = len(T_list) - 2
        T_frac = 1.0
    elif T <= T_list[0]:
        T_idx = 0
        T_frac = 0.0
    else:
        arr = np.where(T_list < T)[0]
        T_idx = arr[-1]
        T_frac = (T - T_list[T_idx]) / (T_list[T_idx + 1] - T_list[T_idx])

    CwL = Cw_tab[T_idx, Fn_idx] + Fn_frac * (Cw_tab[T_idx, Fn_idx + 1] - Cw_tab[T_idx, Fn_idx])
    CwU = Cw_tab[T_idx + 1, Fn_idx] + Fn_frac * (Cw_tab[T_idx + 1, Fn_idx + 1] - Cw_tab[T_idx + 1, Fn_idx])

    cw = CwL + T_frac * (CwU - CwL)

    return cw
