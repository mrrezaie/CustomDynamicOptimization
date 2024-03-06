import numpy as np  
import casadi

# # select the formulation
# formulation = 'mathematical'
formulation = 'interpolant'

# data
AID = np.load('./dataDynamic.npz')

# parameters
iN  = 73 # time frames
iM  = 80 # columns
iDt = 0.01 # time interval

opti = casadi.Opti()

# activation
_a = opti.variable(iN,iM)
opti.set_initial(_a, AID['iSOA'])
opti.subject_to( opti.bounded(0,_a,1) )

# normalized fiber length
_NML = opti.variable(iN,iM)
opti.set_initial(_NML, AID['iNML'])
opti.subject_to( opti.bounded(0.1,_NML,1.9) )

# normalized fiber velocity
_NMV = opti.variable(iN,iM)
opti.set_initial(_NMV, AID['iNMV'])
opti.subject_to( opti.bounded(-1,_NMV,1) )

# fiber length projected
_MLAT = opti.variable(iN,iM)
opti.set_initial(_MLAT, AID['iMLAT'])
opti.subject_to( 1e-4 < casadi.vec(_MLAT)) # 0.1 mm


########## interpolant functions
x1 = AID['x1'].tolist()
x2 = AID['x2'].tolist()
x3 = AID['x3'].tolist()
x4 = AID['x4'].tolist()
y1 = AID['y1'].tolist()
y2 = AID['y2'].tolist()
y3 = AID['y3'].tolist()
y4 = AID['y4'].tolist()

# examples:
# https://github.com/MarcoPenne/AMR_MPC-CBF/blob/33862ab3c3163967543dcf26eb0e8d825755456f/continuos_time/CarModel.py#L27
# https://github.com/YznMur/mown-project/blob/448dce8b1a66346cffb55d375c290b5f7bb4b23b/planning/move_controller/src/move_controller/move_nmpc.py#L176
# https://github.com/MIT-REALM/density_planner/blob/2c3b6d9b7bb6e62546a866ada151d2ec61021a18/motion_planning/MotionPlannerNLP.py#L402

AFLC = casadi.interpolant('AFLC', 'linear', [x1], y1) #, {"inline": True}
AFVC = casadi.interpolant('AFVC', 'linear', [x2], y2)
PFLC = casadi.interpolant('PFLC', 'linear', [x3], y3)
TFLC = casadi.interpolant('TFLC', 'linear', [x4], y4)

# # the outputs of the interpolants are correct numerically
# import matplotlib.pyplot as plt 
# plt.plot(x1, AFLC(x1), label='AFLC')
# plt.plot(x2, AFVC(x2), label='AFVC')
# plt.plot(x3, PFLC(x3), label='PFLC')
# plt.plot(x4, TFLC(x4), label='TFLC')
# plt.legend()
# plt.show(block=False)

for i in range(iN):

    # equality constraint on fiber geometry
    opti.subject_to( (_NML[i,:]*AID['MLopt'][np.newaxis])**2 == \
                      _MLAT[i,:]**2 + AID['height'][np.newaxis]**2 )

    # equality constraint on fiber length and velocity
    if i == 0:
        _NMLDot = 0
    else: # backward euler method
        _NMLDot = (_NML[i,:]-_NML[i-1,:])/iDt

    opti.subject_to( _NMLDot / AID['CVmax'][np.newaxis] == _NMV[i,:] )


    # equality constraint on force equilibrium
    ML  = _NML[i,:] * AID['MLopt'][np.newaxis]
    TL  = AID['iMTL'][i,:][np.newaxis] - _MLAT[i,:]
    NTL = TL / AID['TLslk'][np.newaxis]
    PAcos = _MLAT[i,:] / ML

    if formulation == 'mathematical':

        ################################################################################
        # Option 1 (mathematical functions)
        # being solved in 12s; 25 iterations; objective: 93.186948
        ################################################################################

        # Elastic force-length multiplier
        kt=35; c1=0.200; c2=0.995; c3=0.250; shift=0
        TFLM = c1*casadi.exp(kt*(NTL-c2))-c3+shift
        NTF  =  TFLM

        # Active force-length multiplier
        b11=+0.815; b21=+1.055; b31=+0.162; b41=+0.063
        b12=+0.433; b22=+0.717; b32=-0.030; b42=+0.200
        b13=+0.100; b23=+1.000; b33=+0.354; b43=+0.000 
        NFF1 = b11*casadi.exp(-0.5*(_NML[i,:]-b21)**2 / (b31+b41*_NML[i,:])**2)
        NFF2 = b12*casadi.exp(-0.5*(_NML[i,:]-b22)**2 / (b32+b42*_NML[i,:])**2)
        NFF3 = b13*casadi.exp(-0.5*(_NML[i,:]-b23)**2 / (b33+b43*_NML[i,:])**2)
        AMFLM = NFF1+NFF2+NFF3

        # Pasive force-length multiplier
        kp=4.0; e0=0.6
        PMFLM = (casadi.exp(kp*(_NML[i,:]-1)/e0)-1) / (casadi.exp(kp)-1)

        # Active force-velocity multiplier
        d1=-0.318; d2=-8.149; d3=-0.374; d4=+0.886
        AMFVM = d1*casadi.log(d2*_NMV[i,:]+d3 + casadi.sqrt((d2*_NMV[i,:]+d3)**2+1))+d4

        NMFAT = (_a[i,:]*AMFLM*AMFVM + PMFLM) * PAcos

        opti.subject_to(NTF == NMFAT)

        # equality constraint on moment equilibrium
        for ci,cName in enumerate(AID['include']):
            _tau       = AID['tau'][i,ci] # float
            _momentArm = AID[cName][i,:][np.newaxis] # array
            _moment    = casadi.sum2( _momentArm * NMFAT*AID['IFmax'][np.newaxis]) # float
            opti.subject_to( _moment == _tau )



    elif formulation == 'interpolant':

        ################################################################################
        # Option 2 (interpolant functions)
        # something is wrong here; the objective is turning around 310
        # pagefile and memory are being increased too much (4~5 GB)
        ################################################################################

        # force = list()
        # for mi in range(iM):
        #     NTF   = TFLC(NTL[mi])
        #     NMFAT = ( _a[i,mi]*AFLC(_NML[i,mi])*AFVC(_NMV[i,mi]) + PFLC(_NML[i,mi]) ) * PAcos[mi]
        #     opti.subject_to(NTF == NMFAT)
        #     force.append(NMFAT)
        # # convert force list to casadi MX
        # force2 = casadi.horzcat(*force)

        NTF   = TFLC(NTL)
        NMFAT = ( _a[i,:]*AFLC(_NML[i,:])*AFVC(_NMV[i,:]) + PFLC(_NML[i,:]) ) * PAcos
        opti.subject_to(NTF == NMFAT)

        for ci,cName in enumerate(AID['include']):
            _tau       = AID['tau'][i,ci] # float
            _momentArm = AID[cName][i,:] # array
            # _moment    = sum( _momentArm[mi] * force[mi] * AID['IFmax'][mi] for mi in range(iM)) # float
            # _moment    = casadi.sum2( _momentArm[np.newaxis] * force2 * AID['IFmax'][np.newaxis]) # float
            _moment    = casadi.sum2( _momentArm[np.newaxis] * NMFAT * AID['IFmax'][np.newaxis]) # float
            opti.subject_to( _moment == _tau )

    else:
        raise ValueError('formulation is not defined correctly')


# objective function
activitation = casadi.sumsqr(_a)
fiberVelNorm = 0.1*casadi.sumsqr(_NMV)
# opti.minimize( activitation + fiberVelNorm)
opti.minimize( activitation)

# plugin_options
p_opts = {'expand':False, 
          'detect_simple_bounds':True
          }

# solver_options
s_opts = {'max_iter':10000, 
          'linear_solver':'mumps', 
          'tol':1e-6, 
          # 'hessian_approximation': 'limited-memory',
          'nlp_scaling_method':'gradient-based',
          }
          
opti.solver('ipopt', p_opts, s_opts)

solver = opti.solve()

# %%

# examples
import casadi 

opti = casadi.Opti()

x = opti.variable()
y = opti.variable()

opti.minimize((1-x)**2+(y-x**2)**2)

opti.solver('ipopt')
sol = opti.solve()

print(sol.value(x),sol.value(y))