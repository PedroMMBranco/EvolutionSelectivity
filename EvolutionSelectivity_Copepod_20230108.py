import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from numba import jit
import gzip
import os


#Simulation time
tmax = int(1e4) #int(2.5e4) #int(1e5)
tstep = 1 #0.01 #0.1 #1
# nstep = int(tmax/tstep)
t = np.linspace(0, tmax, tmax//tstep + 1) #np.linspace(0, tmax, nstep + 1)
startsample = int(8e3) #int(2e4) #int(9e4)
endsample = tmax

#Parameters nutrients
delta = 0
#T1 = 100
T1min1 = 1
T1min2 = 10
T1max = 200
T1step = 10
T1range = np.hstack([T1min1, np.arange(T1min2, T1max + T1step, T1step)])
# T1n = int(round((T1max - T1min)/T1step) + 1)
T1n = len(T1range)

#T2 = 2
T2min1 = 0.05
T2min2 = 0.125 #0.2 #0.1
T2max = 2.5 #4 #1.5
T2step = 0.125 #0.2 #0.1
T2range = np.hstack([T2min1, np.arange(T2min2, T2max + T2step, T2step)])
# T2n = int(round((T2max - T2min)/T2step) + 1)
T2n = len(T2range)

#Parameters phytoplankton
mumaxP = 1.9
FMAX1 = 5.1*10**-8
FMAX2 = 8.7*10**-8
KN1 = 4.3
KN2 = 9.32
Qmin1 = 3.1*10**-8
Qmin2 = 1.23*10**-9
Qmax1 = 1.26*10**-7
Qmax2 = 7.7*10**-9
d = 0.1
pL = 0
pU = 1
xP = 1
VP = 0.1
hP = 10**-5

#Parameters zooplankton
#ch = 3.6*10**-7
h = 3.6*10**-7
a = 0.001
q1 = 0.024
q2 = 7*10**-4
#m = 0.1
m0 = 0.05 #Background mortality (excluding predation): Hirst & Kiørboe (MEPS 2002); Kiørboe et al. (L&O 2018)
mf = 0.2 #0.15 #0.05
BZmax = 1
xZ = 1
VZ = 1 #0.1
hZ = 10**-5

#List of results to save
resultsT1T2 = []
minT1T2 = []
maxT1T2 = []
meanT1T2 = []
pmeanT1T2 = []
ratioPmeanT1T2 = []
AmeanT1T2 = []
alphameanT1T2 = []


#Parameter loop
# for T1 in np.arange(T1min, T1max + T1step, T1step):
for T1 in T1range:
    
    resultsT2 = []
    minT2 = []
    maxT2 = []
    meanT2 = []
    pmeanT2 = []
    ratioPmeanT2 = []
    AmeanT2 = []
    alphameanT2 = []
    
    # for T2 in np.arange(T2min, T2max + T2step, T2step):
    for T2 in T2range:
        
        print("T1 = ", T1, ", T2 = ", T2)

        #Functions
        @jit
        def fmax1(t, p, FMAX1):
            return p*FMAX1
        
        @jit
        def fmax2(t, p, FMAX2):
            return (1-p)*FMAX2

        @jit
        def f1(N1, Q1, t, p, FMAX1, KN1, Qmin1, Qmax1):
            return fmax1(t, p, FMAX1)*N1/(N1 + KN1)*(Qmax1 - Q1)/(Qmax1 - Qmin1)
        
        @jit
        def f2(N2, Q2, t, p, FMAX2, KN2, Qmin2, Qmax2):
            return fmax2(t, p, FMAX2)*N2/(N2 + KN2)*(Qmax2 - Q2)/(Qmax2 - Qmin2)

        @jit
        def muP(mumaxP, Q1, Q2, Qmin1, Qmin2, t):
            return mumaxP*min(1 - Qmin1/Q1,1 - Qmin2/Q2)

        @jit
        def alpha(A, q1, q2, t, Q1, Q2):
#            return np.exp(-A*((q1/q2 - Q1/Q2)/(q1/q2))**2)
#            return np.exp(-A*((q2/q1 - Q2/Q1)/(q2/q1))**2)
            return np.exp(-A*((np.log(q1/q2) - np.log(Q1/Q2)))**2)

#        @jit
#        def h(t, ch, A):
#            return ch/A

        @jit
        def g(A, q1, q2, t, Q1, Q2, P, h):
            return a*alpha(A, q1, q2, t, Q1, Q2)*P/(1 + a*alpha(A, q1, q2, t, Q1, Q2)*h*P)

        @jit
        def e(t, Q1, Q2, q1, q2):
            return min(Q1/q1,Q2/q2)
        
        @jit
        def m(t, Q1, Q2, q1, q2, m0, A, mf):
            return m0 + alpha(A, q1, q2, t, Q1, Q2)*mf       

        @jit
        def fmax1m(t, p, FMAX1, hP):
            return (1 - hP)*p*FMAX1

        @jit
        def fmax1M(t, p, FMAX1, hP):
            return (1 + hP)*p*FMAX1

        @jit
        def fmax2m(t, p, FMAX2, hP):
            return (1-(1 - hP)*p)*FMAX2
        
        @jit
        def fmax2M(t, p, FMAX2, hP):
            return (1-(1 + hP)*p)*FMAX2

        @jit
        def f1m(t, p, FMAX1, hP, N1, KN1, Qmin1, Qmax1, Q1m):
            return fmax1m(t, p, FMAX1, hP)*N1/(N1 + KN1)*(Qmax1 - Q1m)/(Qmax1 - Qmin1)

        @jit
        def f1M(t, p, FMAX1, hP, N1, KN1, Qmin1, Qmax1, Q1M):
            return fmax1M(t, p, FMAX1, hP)*N1/(N1 + KN1)*(Qmax1 - Q1M)/(Qmax1 - Qmin1)

        @jit
        def f2m(t, p, FMAX2, hP, N2, KN2, Qmin2, Qmax2, Q2m):
            return fmax2m(t, p, FMAX2, hP)*N2/(N2 + KN2)*(Qmax2 - Q2m)/(Qmax2 - Qmin2)

        @jit
        def f2M(t, p, FMAX2, hP, N2, KN2, Qmin2, Qmax2, Q2M):
            return fmax2M(t, p, FMAX2, hP)*N2/(N2 + KN2)*(Qmax2 - Q2M)/(Qmax2 - Qmin2)

        @jit
        def muPm(mumaxP, Q1m, Q2m, Qmin1, Qmin2, t):
            return mumaxP*min(1 - Qmin1/Q1m,1 - Qmin2/Q2m)

        @jit
        def muPM(mumaxP, Q1M, Q2M, Qmin1, Qmin2, t):
            return mumaxP*min(1 - Qmin1/Q1M,1 - Qmin2/Q2M)

        @jit
        def alphamP(A, q1, q2, t, Q1m, Q2m):
#            return np.exp(-A*((q1/q2 - Q1m/Q2m)/(q1/q2))**2)
#            return np.exp(-A*((q2/q1 - Q2m/Q1m)/(q2/q1))**2)
            return np.exp(-A*((np.log(q1/q2) - np.log(Q1m/Q2m)))**2)
       
        @jit
        def alphaMP(A, q1, q2, t, Q1M, Q2M):
#            return np.exp(-A*((q1/q2 - Q1M/Q2M)/(q1/q2))**2)
#            return np.exp(-A*((q2/q1 - Q2M/Q1M)/(q2/q1))**2)
           return np.exp(-A*((np.log(q1/q2) - np.log(Q1M/Q2M)))**2)

        @jit
        def gmP(A, q1, q2, t, Q1m, Q2m, P, h):
            return a*alphamP(A, q1, q2, t, Q1m, Q2m)*P/(1 + a*alphamP(A, q1, q2, t, Q1m, Q2m)*h*P)

        @jit
        def gMP(A, q1, q2, t, Q1M, Q2M, P, h):
            return a*alphaMP(A, q1, q2, t, Q1M, Q2M)*P/(1 + a*alphaMP(A, q1, q2, t, Q1M, Q2M)*h*P)

        @jit
        def Am(t, A, hZ):
            return (1 - hZ)*A

        @jit
        def AM(t, A, hZ):
            return (1 + hZ)*A

        @jit
        def alphamZ(A, hZ, q1, q2, t, Q1, Q2):
#            return np.exp(-Am(t, A, hZ)*((q1/q2 - Q1/Q2)/(q1/q2))**2)
#            return np.exp(-Am(t, A, hZ)*((q2/q1 - Q2/Q1)/(q2/q1))**2)
            return np.exp(-Am(t, A, hZ)*((np.log(q1/q2) - np.log(Q1/Q2)))**2)
        
        @jit
        def alphaMZ(A, hZ, q1, q2, t, Q1, Q2):
#            return np.exp(-AM(t, A, hZ)*((q1/q2 - Q1/Q2)/(q1/q2))**2)
#            return np.exp(-AM(t, A, hZ)*((q2/q1 - Q2/Q1)/(q2/q1))**2)
            return np.exp(-AM(t, A, hZ)*((np.log(q1/q2) - np.log(Q1/Q2)))**2)

#        @jit
#        def hmZ(t, ch, A, hZ):
#            return ch/Am(t, A, hZ)

#        @jit
#        def hMZ(t, ch, A, hZ):
#            return ch/AM(t, A, hZ)

        @jit
        def gmZ(a, A, hZ, q1, q2, t, Q1, Q2, P, h):
            return a*alphamZ(A, hZ, q1, q2, t, Q1, Q2)*P/(1 + a*alphamZ(A, hZ, q1, q2, t, Q1, Q2)*h*P)

        @jit
        def gMZ(a, A, hZ, q1, q2, t, Q1, Q2, P, h):
            return a*alphaMZ(A, hZ, q1, q2, t, Q1, Q2)*P/(1 + a*alphaMZ(A, hZ, q1, q2, t, Q1, Q2)*h*P)

        @jit
        def mm(t, Q1, Q2, q1, q2, m0, A, mf, hZ):
            return m0 + alphamZ(A, hZ, q1, q2, t, Q1, Q2)*mf
        
        @jit
        def mM(t, Q1, Q2, q1, q2, m0, A, mf, hZ):
            return m0 + alphaMZ(A, hZ, q1, q2, t, Q1, Q2)*mf        

        @jit
        def BP(t, p, pL, pU):
            return (p - pL)*(pU - p)

        @jit
#        def BZ(t):
#            return 1
        def BZ(t, A, BZmax):
#             if(A < BZmax):
#                 return A
#             else:
#                 return BZmax
            return min(A, BZmax)

        @jit
        def wPm(t, mumaxP, Q1m, Q2m, Qmin1, Qmin2, d, delta, A, q1, q2, P, h, Z):
            return muPm(mumaxP, Q1m, Q2m, Qmin1, Qmin2, t) - d - delta - gmP(A, q1, q2, t, Q1m, Q2m, P, h)*Z/P

        @jit
        def wPM(t, mumaxP, Q1M, Q2M, Qmin1, Qmin2, d, delta, A, q1, q2, P, h, Z):
            return muPM(mumaxP, Q1M, Q2M, Qmin1, Qmin2, t) - d - delta - gMP(A, q1, q2, t, Q1M, Q2M, P, h)*Z/P

        @jit
        def wZm(t, Q1, Q2, q1, q2, a, A, hZ, P, h, m, delta, m0, mf):
            return e(t, Q1, Q2, q1, q2)*gmZ(a, A, hZ, q1, q2, t, Q1, Q2, P, h) - mm(t, Q1, Q2, q1, q2, m0, A, mf, hZ) - delta

        @jit
        def wZM(t, Q1, Q2, q1, q2, a, A, hZ, P, h, m, delta, m0, mf):
            return e(t, Q1, Q2, q1, q2)*gMZ(a, A, hZ, q1, q2, t, Q1, Q2, P, h) - mM(t, Q1, Q2, q1, q2, m0, A, mf, hZ) - delta

        @jit
        def gradP(t, mumaxP, Q1m, Q2m, Q1M, Q2M, Qmin1, Qmin2, d, delta, A, q1, q2, P, h, Z, hP, p):
            return (wPM(t, mumaxP, Q1M, Q2M, Qmin1, Qmin2, d, delta, A, q1, q2, P, h, Z) - wPm(t, mumaxP, Q1m, Q2m, Qmin1, Qmin2, d, delta, A, q1, q2, P, h, Z))/(2*hP*p)

        @jit
        def gradZ(t, Q1, Q2, q1, q2, a, A, hZ, P, h, m, delta, m0, mf):
            return (wZM(t, Q1, Q2, q1, q2, a, A, hZ, P, h, m, delta, m0, mf) - wZm(t, Q1, Q2, q1, q2, a, A, hZ, P, h, m, delta, m0, mf))/(2*hZ*A)

        #Differential equations
        @jit
        # def eqs(eq, t):
        def eqs(t, eq):            
            # global P, Q1, Q2, Z, N1, N2, Q1m, Q1M, Q2m, Q2M, p, A
            P = eq[0]
            Q1 = eq[1]
            Q2 = eq[2]
            Z = eq[3]
            N1 = eq[4]
            N2 = eq[5]
            Q1m = eq[6]
            Q1M = eq[7]
            Q2m = eq[8]
            Q2M = eq[9]
            p = eq[10]
            A = eq[11]
            
            dPdt = (muP(mumaxP, Q1, Q2, Qmin1, Qmin2, t) - d - delta)*P- g(A, q1, q2, t, Q1, Q2, P, h)*Z
            dQ1dt = f1(N1, Q1, t, p, FMAX1, KN1, Qmin1, Qmax1) - muP(mumaxP, Q1, Q2, Qmin1, Qmin2, t)*Q1
            dQ2dt = f2(N2, Q2, t, p, FMAX2, KN2, Qmin2, Qmax2) - muP(mumaxP, Q1, Q2, Qmin1, Qmin2, t)*Q2
            dZdt = (e(t, Q1, Q2, q1, q2)*g(A, q1, q2, t, Q1, Q2, P, h) - m(t, Q1, Q2, q1, q2, m0, A, mf) - delta)*Z
            dN1dt = delta*(T1 - N1) - f1(N1, Q1, t, p, FMAX1, KN1, Qmin1, Qmax1)*P + d*P*Q1 + m(t, Q1, Q2, q1, q2, m0, A, mf)*Z*q1 + g(A, q1, q2, t, Q1, Q2, P, h)*Z*(Q1 - e(t, Q1, Q2, q1, q2)*q1)
            dN2dt = delta*(T2 - N2) - f2(N2, Q2, t, p, FMAX2, KN2, Qmin2, Qmax2)*P + d*P*Q2 + m(t, Q1, Q2, q1, q2, m0, A, mf)*Z*q2 + g(A, q1, q2, t, Q1, Q2, P, h)*Z*(Q2 - e(t, Q1, Q2, q1, q2)*q2)            
            dQ1mdt = f1m(t, p, FMAX1, hP, N1, KN1, Qmin1, Qmax1, Q1m) - muPm(mumaxP, Q1m, Q2m, Qmin1, Qmin2, t)*Q1m
            dQ1Mdt = f1M(t, p, FMAX1, hP, N1, KN1, Qmin1, Qmax1, Q1M) - muPM(mumaxP, Q1M, Q2M, Qmin1, Qmin2, t)*Q1M   
            dQ2mdt = f2m(t, p, FMAX2, hP, N2, KN2, Qmin2, Qmax2, Q2m) - muPm(mumaxP, Q1m, Q2m, Qmin1, Qmin2, t)*Q2m
            dQ2Mdt = f2M(t, p, FMAX2, hP, N2, KN2, Qmin2, Qmax2, Q2M) - muPM(mumaxP, Q1M, Q2M, Qmin1, Qmin2, t)*Q2M   
            
            dpdt = 1/xP*BP(t, p, pL, pU)*VP*gradP(t, mumaxP, Q1m, Q2m, Q1M, Q2M, Qmin1, Qmin2, d, delta, A, q1, q2, P, h, Z, hP, p)
            dAdt = 1/xZ*BZ(t, A, BZmax)*VZ*gradZ(t, Q1, Q2, q1, q2, a, A, hZ, P, h, m, delta, m0, mf)
            
            return [dPdt, dQ1dt, dQ2dt, dZdt, dN1dt, dN2dt, dQ1mdt, dQ1Mdt, dQ2mdt, dQ2Mdt, dpdt, dAdt]

        #Initial conditions
        p0 = 0.5
        A0 = 0.1
        P0 = 10**7
        Q10 = Qmin1
        Q20 = Qmin2
        Z0 = 1
        N10 = T1 - P0*Q10 - Z0*q1
        N20 = T2 - P0*Q20 - Z0*q2        
        Q10m = Qmin1
        Q10M = Qmin1
        Q20m = Qmin2
        Q20M = Qmin2

        #Numerical integration
        init = [P0, Q10, Q20, Z0, N10, N20, Q10m, Q10M, Q20m, Q20M, p0, A0]
        # sol = odeint(eqs, init, t, ixpr = True)
        # sol = solve_ivp(fun = eqs, t_span = (0, tmax), y0 = init, dense_output = False, method = "RK45", first_step = 0.1, atol = 1e-6, rtol = 1e-3)
        # sol = solve_ivp(fun = eqs, t_span = (0, tmax), y0 = init, dense_output = True, method = "LSODA", atol = 1e-9, rtol = 1e-4)
        sol = solve_ivp(fun = eqs, t_span = (0, tmax), y0 = init, dense_output = True, method = "LSODA", atol = 1e-9, rtol = 1e-4, max_step = 0.05)
        
        # Ps = sol[0:tmax,0]
        # Q1s = sol[0:tmax,1]
        # Q2s = sol[0:tmax,2]
        # Zs = sol[0:tmax,3]
        # N1s = sol[0:tmax,4]
        # N2s = sol[0:tmax,5]

        # Q1ms = sol[0:tmax,6]
        # Q1Ms = sol[0:tmax,7]
        # Q2ms = sol[0:tmax,8]
        # Q2Ms = sol[0:tmax,9]

        # ps = sol[0:tmax,10]
        # As = sol[0:tmax,11]
        
        # ratioP = Q1s/Q2s
        
        # Pmin = min(sol[startsample:endsample, 0])
        # Pmax = max(sol[startsample:endsample, 0])
        # Zmax = max(sol[startsample:endsample, 3])
        # Zmin = min(sol[startsample:endsample, 3])
        # N1min = min(sol[startsample:endsample, 4])
        # N1max = max(sol[startsample:endsample, 4])
        # N2min = min(sol[startsample:endsample, 5])
        # N2max = max(sol[startsample:endsample, 5])
        
        # ratioPmin = min(sol[startsample:endsample, 1]/sol[startsample:endsample, 2])
        # ratioPmax = max(sol[startsample:endsample, 1]/sol[startsample:endsample, 2])
        # ratioPmean = np.mean(sol[startsample:endsample, 1]/sol[startsample:endsample, 2])

        # pmin = min(sol[startsample:endsample, 10])
        # pmax = max(sol[startsample:endsample, 10])
        # pmean = np.mean(sol[startsample:endsample, 10])
        # Amin = min(sol[startsample:endsample, 11])
        # Amax = max(sol[startsample:endsample, 11])
        # Amean = np.mean(sol[startsample:endsample, 11])
        
        Ps = sol.sol(t)[0][0:tmax]
        Q1s = sol.sol(t)[1][0:tmax]
        Q2s = sol.sol(t)[2][0:tmax]
        Zs = sol.sol(t)[3][0:tmax]
        N1s = sol.sol(t)[4][0:tmax]
        N2s = sol.sol(t)[5][0:tmax]

        Q1ms = sol.sol(t)[6][0:tmax]
        Q1Ms = sol.sol(t)[7][0:tmax]
        Q2ms = sol.sol(t)[8][0:tmax]
        Q2Ms = sol.sol(t)[9][0:tmax]

        ps = sol.sol(t)[10][0:tmax]
        As = sol.sol(t)[11][0:tmax]
        
        ratioP = Q1s/Q2s
        
        Pmin = min(sol.sol(t)[0][startsample:endsample])
        Pmax = max(sol.sol(t)[0][startsample:endsample])
        Pmean = np.mean(sol.sol(t)[0][startsample:endsample])
        Zmax = max(sol.sol(t)[3][startsample:endsample])
        Zmin = min(sol.sol(t)[3][startsample:endsample])
        Zmean = np.mean(sol.sol(t)[3][startsample:endsample])
        N1min = min(sol.sol(t)[4][startsample:endsample])
        N1max = max(sol.sol(t)[4][startsample:endsample])
        N1mean = np.mean(sol.sol(t)[4][startsample:endsample])
        N2min = min(sol.sol(t)[5][startsample:endsample])
        N2max = max(sol.sol(t)[5][startsample:endsample])
        N2mean = np.mean(sol.sol(t)[5][startsample:endsample])
        
        ratioPmin = min(sol.sol(t)[1][startsample:endsample]/sol.sol(t)[2][startsample:endsample])
        ratioPmax = max(sol.sol(t)[1][startsample:endsample]/sol.sol(t)[2][startsample:endsample])
        ratioPmean = np.mean(sol.sol(t)[1][startsample:endsample]/sol.sol(t)[2][startsample:endsample])
        
#        alphamin = min(np.exp(-sol.sol(t)[11][startsample:endsample]*((q1/q2 - sol.sol(t)[1][startsample:endsample]/sol.sol(t)[2][startsample:endsample])/(q1/q2))**2))
#        alphamax = max(np.exp(-sol.sol(t)[11][startsample:endsample]*((q1/q2 - sol.sol(t)[1][startsample:endsample]/sol.sol(t)[2][startsample:endsample])/(q1/q2))**2))
#        alphamean = np.mean(np.exp(-sol.sol(t)[11][startsample:endsample]*((q1/q2 - sol.sol(t)[1][startsample:endsample]/sol.sol(t)[2][startsample:endsample])/(q1/q2))**2))

#        alphamin = min(np.exp(-sol.sol(t)[11][startsample:endsample]*((q2/q1 - sol.sol(t)[2][startsample:endsample]/sol.sol(t)[1][startsample:endsample])/(q2/q1))**2))
#        alphamax = max(np.exp(-sol.sol(t)[11][startsample:endsample]*((q2/q1 - sol.sol(t)[2][startsample:endsample]/sol.sol(t)[1][startsample:endsample])/(q2/q1))**2))
#        alphamean = np.mean(np.exp(-sol.sol(t)[11][startsample:endsample]*((q2/q1 - sol.sol(t)[2][startsample:endsample]/sol.sol(t)[1][startsample:endsample])/(q2/q1))**2))

        alphamin = min(np.exp(-sol.sol(t)[11][startsample:endsample]*((np.log(q1/q2) - np.log(sol.sol(t)[1][startsample:endsample]/sol.sol(t)[2][startsample:endsample])))**2))
        alphamax = max(np.exp(-sol.sol(t)[11][startsample:endsample]*((np.log(q1/q2) - np.log(sol.sol(t)[1][startsample:endsample]/sol.sol(t)[2][startsample:endsample])))**2))
        alphamean = np.mean(np.exp(-sol.sol(t)[11][startsample:endsample]*((np.log(q1/q2) - np.log(sol.sol(t)[1][startsample:endsample]/sol.sol(t)[2][startsample:endsample])))**2))

        pmin = min(sol.sol(t)[10][startsample:endsample])
        pmax = max(sol.sol(t)[10][startsample:endsample])
        pmean = np.mean(sol.sol(t)[10][startsample:endsample])
        
        Amin = min(sol.sol(t)[11][startsample:endsample])
        Amax = max(sol.sol(t)[11][startsample:endsample])
        Amean = np.mean(sol.sol(t)[11][startsample:endsample])
        
        
        fmax1s = ps*FMAX1
        fmax2s = (1-ps)*FMAX2
        f1s = fmax1s*N1s/(N1s + KN1)*(Qmax1 - Q1s)/(Qmax1 - Qmin1)
        f2s = fmax2s*N2s/(N2s + KN2)*(Qmax2 - Q2s)/(Qmax2 - Qmin2)
        es = list(map(min, zip(Q1s/q1,Q2s/q2)))
#        alphas = np.exp(-As*((q1/q2 - Q1s/Q2s)/(q1/q2))**2)
#        alphas = np.exp(-As*((q2/q1 - Q2s/Q1s)/(q2/q1))**2)
        alphas = np.exp(-As*((np.log(q1/q2) - np.log(Q1s/Q2s)))**2)
        gs = a*alphas*Ps/(1 + a*alphas*h*Ps)
        ms = m0 + alphas*mf
        
        fmax1ms = (1 - hP)*ps*FMAX1
        fmax1Ms = (1 + hP)*ps*FMAX1
        fmax2ms = (1-(1 - hP)*ps)*FMAX2
        fmax2Ms = (1-(1 + hP)*ps)*FMAX2
        f1ms = fmax1ms*N1s/(N1s + KN1)*(Qmax1 - Q1ms)/(Qmax1 - Qmin1)
        f1Ms = fmax1Ms*N1s/(N1s + KN1)*(Qmax1 - Q1Ms)/(Qmax1 - Qmin1)
        f2ms = fmax2ms*N2s/(N2s + KN2)*(Qmax2 - Q2ms)/(Qmax2 - Qmin2)
        f2Ms = fmax2Ms*N2s/(N2s + KN2)*(Qmax2 - Q2Ms)/(Qmax2 - Qmin2)
        muPms = mumaxP*np.asarray(list(map(min, zip(1 - Qmin1/Q1ms,1 - Qmin2/Q2ms))))
        muPMs = mumaxP*np.asarray(list(map(min, zip(1 - Qmin1/Q1Ms,1 - Qmin2/Q2Ms))))
#        hs = ch/As      
#        alphamPs = np.exp(-As*((q1/q2 - Q1ms/Q2ms)/(q1/q2))**2)
#        alphaMPs = np.exp(-As*((q1/q2 - Q1Ms/Q2Ms)/(q1/q2))**2)
#        alphamPs = np.exp(-As*((q2/q1 - Q2ms/Q1ms)/(q2/q1))**2)
#        alphaMPs = np.exp(-As*((q2/q1 - Q2Ms/Q1Ms)/(q2/q1))**2)
        alphamPs = np.exp(-As*((np.log(q1/q2) - np.log(Q1ms/Q2ms)))**2)
        alphaMPs = np.exp(-As*((np.log(q1/q2) - np.log(Q1Ms/Q2Ms)))**2)

        gmPs = a*alphamPs*Ps/(1 + a*alphamPs*h*Ps)
        gMPs = a*alphaMPs*Ps/(1 + a*alphaMPs*h*Ps)

        Ams = (1 - hZ)*As
        AMs = (1 + hZ)*As
#        alphamZs = np.exp(-Ams*((q1/q2 - Q1s/Q2s)/(q1/q2))**2)
#        alphaMZs = np.exp(-AMs*((q1/q2 - Q1s/Q2s)/(q1/q2))**2)        
#        alphamZs = np.exp(-Ams*((q2/q1 - Q2s/Q1s)/(q2/q1))**2)
#        alphaMZs = np.exp(-AMs*((q2/q1 - Q2s/Q1s)/(q2/q1))**2)
        alphamZs = np.exp(-Ams*((np.log(q1/q2) - np.log(Q1s/Q2s)))**2)
        alphaMZs = np.exp(-AMs*((np.log(q1/q2) - np.log(Q1s/Q2s)))**2)
#        hmZs = ch/Ams
#        hMZs = ch/AMs
        gmZs = a*alphamZs*Ps/(1 + a*alphamZs*h*Ps)
        gMZs = a*alphaMZs*Ps/(1 + a*alphaMZs*h*Ps)
        mms = m0 + alphamZs*mf
        mMs = m0 + alphaMZs*mf
        
        wPms = muPms - d - delta - gmPs*Zs/Ps
        wPMs = muPMs - d - delta - gMPs*Zs/Ps
        gradPs = (wPMs - wPms)/(2*hP*ps)

        wZms = es*gmZs - mms - delta
        wZMs = es*gMZs - mMs - delta
        gradZs = (wZMs - wZms)/(2*hZ*As)

        T1s = N1s + Ps*Q1s + Zs*q1
        T2s = N2s + Ps*Q2s + Zs*q2
        

        resultsT2.append([N1s, N2s, Ps, Q1s, Q2s, ratioP, Zs, ps, As, alphas])
        minT2.append([N1min, N2min, Pmin, ratioPmin, Zmin, pmin, Amin, alphamin])
        maxT2.append([N1max, N2max, Pmax, ratioPmax, Zmax, pmax, Amax, alphamax])
        meanT2.append([N1mean, N2mean, Pmean, ratioPmean, Zmean, pmean, Amean, alphamean])
        pmeanT2.append(pmean)
        ratioPmeanT2.append(ratioPmean)
        
        if Zmean < 1.0e-3: 
            AmeanT2.append('')
            alphameanT2.append('')
        else:
            AmeanT2.append(Amean)
            alphameanT2.append(alphamean)
            
#        alphameanT2.append(alphamean)
    
    resultsT1T2.append(resultsT2)
    minT1T2.append(minT2)
    maxT1T2.append(maxT2)
    meanT1T2.append(meanT2)
    pmeanT1T2.append(pmeanT2)
    ratioPmeanT1T2.append(ratioPmeanT2)
    AmeanT1T2.append(AmeanT2)
    alphameanT1T2.append(alphameanT2)
    

#Save output
os.makedirs('output')
##Eco-evolutionary dynamics
for i in range(0, T1n):
    for j in range(0, T2n):
        np.savetxt('output/abundances_traits_T1_' + str(T1range[i]) + '_T2_' + str(T2range[j]) + '.txt', np.transpose((range(tmax), resultsT1T2[i][j][0], resultsT1T2[i][j][1], resultsT1T2[i][j][2], resultsT1T2[i][j][3], resultsT1T2[i][j][4], resultsT1T2[i][j][5], resultsT1T2[i][j][6], resultsT1T2[i][j][7], resultsT1T2[i][j][8], resultsT1T2[i][j][9])), delimiter = "\t", fmt = '%s')
        
        # Compress output files
        in_file = 'output/abundances_traits_T1_' + str(T1range[i]) + '_T2_' + str(T2range[j]) + '.txt'
        in_data = open(in_file, 'rb').read()
        out_gz = 'output/abundances_traits_T1_' + str(T1range[i]) + '_T2_' + str(T2range[j]) + '.txt.gz'
        gzf = gzip.open(out_gz, "wb")
        gzf.write(in_data)
        gzf.close()
        os.unlink(in_file)          

##Contour plots
np.savetxt('output/pmean.txt', pmeanT1T2, delimiter = "\t", fmt = '%s')
np.savetxt('output/ratioPmean.txt', ratioPmeanT1T2, delimiter = "\t", fmt = '%s')
np.savetxt('output/Amean.txt', AmeanT1T2, delimiter = "\t", fmt = '%s')
np.savetxt('output/alphamean.txt', alphameanT1T2, delimiter = "\t", fmt = '%s')

##Bifurcation plots: T1-abundances
for j in range (0, T2n):
    out = []
    out_final = []
    for i in range(0, T1n):
        out = [minT1T2[i][j][0], maxT1T2[i][j][0], meanT1T2[i][j][0], minT1T2[i][j][1], maxT1T2[i][j][1], meanT1T2[i][j][1], minT1T2[i][j][2], maxT1T2[i][j][2], meanT1T2[i][j][2], minT1T2[i][j][3], maxT1T2[i][j][3], meanT1T2[i][j][3], minT1T2[i][j][4], maxT1T2[i][j][4], meanT1T2[i][j][4], minT1T2[i][j][5], maxT1T2[i][j][5], meanT1T2[i][j][5], minT1T2[i][j][6], maxT1T2[i][j][6], meanT1T2[i][j][6], minT1T2[i][j][7], maxT1T2[i][j][7], meanT1T2[i][j][7]]
        out_final.append(out)
    np.savetxt('output/T1-abundances_T2_' + str(T2range[j]) + '.txt', out_final, delimiter = "\t", fmt = '%s')

##Bifurcation plots: T2-abundances
for i in range(0, T1n):
    out = []
    out_final = []
    for j in range (0, T2n):
        out = [minT1T2[i][j][0], maxT1T2[i][j][0], meanT1T2[i][j][0], minT1T2[i][j][1], maxT1T2[i][j][1], meanT1T2[i][j][1], minT1T2[i][j][2], maxT1T2[i][j][2], meanT1T2[i][j][2], minT1T2[i][j][3], maxT1T2[i][j][3], meanT1T2[i][j][3], minT1T2[i][j][4], maxT1T2[i][j][4], meanT1T2[i][j][4], minT1T2[i][j][5], maxT1T2[i][j][5], meanT1T2[i][j][5], minT1T2[i][j][6], maxT1T2[i][j][6], meanT1T2[i][j][6], minT1T2[i][j][7], maxT1T2[i][j][7], meanT1T2[i][j][7]]
        out_final.append(out)
    np.savetxt('output/T2-abundances_T1_' + str(T1range[i]) + '.txt', out_final, delimiter = "\t", fmt = '%s')
    
    
