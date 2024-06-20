import numpy as np  
import casadi

########## functions

def firstOrderSmoothedActivationDynamics(e,a, t_a=0.015, t_d=0.05, b=0.1):

    '''
    smoothed approximation of first order activation dynamics (Winters1995, Zajac1989)
    e       muscle excitation,
    a       muscle activation,
    t_a     activation time constant 
                (range: 10 ms for fast to 20 ms for slow muscle)
                (0.015 s in Thelen2003 and DeGroote2016),
    t_d     deactivation time constant
                (range: 30 ms for fast to 50 ms for slow muscle)
                ()
                (0.05 s in Thelen2003, 0.06 s in DeGroote2016),
    b       parameter determining transition smoothness (0.1).

    https://opensimconfluence.atlassian.net/wiki/x/HhkqAw
    https://github.com/opensim-org/opensim-core/blob/main/OpenSim/Actuators/DeGrooteFregly2016Muscle.cpp#L218-L220
    https://github.com/opensim-org/opensim-core/blob/main/OpenSim/Simulation/Model/Bhargava2004SmoothedMuscleMetabolics.h#L90-L98
    '''
    # DeGroote2016
    f = 0.5 * casadi.tanh(b*(e-a))
    z = 0.5 + 1.5*a
    const = (0.5+f)/(t_a*z) + (0.5-f)*z/t_d
    return const * (e-a) # == da/dt

    # f = 0.5 * casadi.tanh(b*(e-a))
    # z = 0.5 + 1.5*a
    # const = t_a*z*(0.5+f) + t_d/z*(0.5-f)
    # return (e-a) / const 



def calcDeGrooteTendonCurveShift(kt):
    # Tendon force-length multipliers
    c1=0.200; c2=0.995; c3=0.250; shift=0 #1.17507567e-02
    TFLM35 = c1*np.exp(35*(1.-c2))-c3+shift
    TFLMkt = c1*np.exp(kt*(1.-c2))-c3+shift
    shift = TFLM35-TFLMkt
    return shift

def calcDeGrooteTendonForceLengthMultiplier(NTL, kt, shift):
    # Tendon force-length multiplier (S1)
    c1=0.200; c2=0.995; c3=0.250
    NTF = c1*casadi.exp(kt*(NTL-c2))-c3+shift
    # ignore negative tendon forces
    NTF = casadi.if_else(NTF<=0.0, 0.0, NTF)
    return NTF

def calcDeGrooteActiveFiberForceLengthMultiplier(NML):
    # Active muscle force-length multiplier (S2)

    # default De Groote 
    b11=+0.815; b21=+1.055; b31=+0.162; b41=+0.063
    b12=+0.433; b22=+0.717; b32=-0.030; b42=+0.200
    b13=+0.100; b23=+1.000; b33=+0.354; b43=+0.000
    NFF1 = b11*casadi.exp(-0.5*(NML-b21)**2 / (b31+b41*NML)**2)
    NFF2 = b12*casadi.exp(-0.5*(NML-b22)**2 / (b32+b42*NML)**2)
    NFF3 = b13*casadi.exp(-0.5*(NML-b23)**2 / (b33+b43*NML)**2)
    AMFLM = NFF1+NFF2+NFF3
    # ignore normalized active forces more than 1
    AMFLM = casadi.if_else(AMFLM>=1.0, 1.0, AMFLM)
    return AMFLM

def calcDeGrooteFiberForceVelocityMultiplier(NMV):
    # Active muscle force-velocity multiplier (S4)
    d1=-0.318; d2=-8.149; d3=-0.374; d4=+0.886
    AMFVM = d1*casadi.log(d2*NMV+d3 + casadi.sqrt((d2*NMV+d3)**2+1))+d4
    return AMFVM

def calcDeGrootePassiveFiberForceLengthMultiplier(NML):
    # Pasive muscle force multiplier (S3)
    kp=4.0; e0=0.6
    PMFLM = (casadi.exp(kp*(NML-1)/e0) - 1) / (casadi.exp(kp) - 1)
    # ignore negative passive forces
    PMFLM = casadi.if_else(PMFLM<=0.0, 0.0, PMFLM)
    return PMFLM




# data
AID = dict( np.load('./data.npz') )

# adjust tendon stiffness and curve shift (DeGroote muscle)
stiffness,shift = list(),list()
for mName in AID['nameMuscles']:
    if mName.split('_')[0] in ['soleus', 'gasmed', 'gaslat']: # triceps surae
        value = 20
        stiffness.append(value)
        shift.append( calcDeGrooteTendonCurveShift(value))
    elif mName.split('_')[0] in ['recfem', 'vasint', 'vaslat', 'vasmed']: # quadriceps
        value = 25
        stiffness.append(value)
        shift.append( calcDeGrooteTendonCurveShift(value))
    else:
        stiffness.append(35)
        shift.append( 0)

AID['tendonStiffness']  = np.array(stiffness)
AID['tendonCurveShift'] = np.array(shift)

# parameters
iN, iM = AID['MTL'].shape
iDt = 0.01 # time interval (mesh frequency == 100 Hz)

opti = casadi.Opti()

########## variables, initial guesses, and bounds

# muscle activation
_act = opti.variable(iN,iM)
opti.set_initial(_act, 0.1*np.ones(_act.shape) )

# muscle excitation
_exc = opti.variable(iN,iM)
opti.set_initial(_exc, 0.1*np.ones(_exc.shape) )

# normalized fiber length
_NML = opti.variable(iN,iM)
opti.set_initial(_NML, np.ones(_NML.shape))

# normalized fiber velocity
_NMV = opti.variable(iN,iM)
opti.set_initial(_NMV, np.zeros(_NMV.shape) )

# normalized tendon length
_NTL = opti.variable(iN,iM)
opti.set_initial(_NTL, np.ones(_NTL.shape))


########## bounds

opti.subject_to( opti.bounded(0,_act,1) )
opti.subject_to( opti.bounded(0,_exc,1) )
opti.subject_to( opti.bounded(0.2,_NML,2) )
opti.subject_to( opti.bounded(-1,_NMV,1) )
opti.subject_to( 1.0 <= casadi.vec(_NTL))

_TL = _NTL*np.tile(AID['TLslk'],[iN,1])
_ML = _NML * np.tile(AID['MLopt'],[iN,1])
_MLAT = AID['MTL'] - _TL
_PAcos = _MLAT / _ML

# bounds on on fiber pennation angle, min == 84.26deg or arccos(0.1)
opti.subject_to( opti.bounded(0.1,_PAcos,1))

########## constraints

# equality constraint on fiber geometry
opti.subject_to( _ML == casadi.sqrt( _MLAT**2 + np.tile(AID['muscleHeight'],[iN,1])**2) )

# backward Euler method
_NML_dot = (_NML[1:,:]-_NML[:-1,:]) / iDt
_act_dot = (_act[1:,:]-_act[:-1,:]) / iDt
# my workaround, zeros for the first time frame 
_NML_dot = casadi.vertcat( np.zeros([1,iM]), _NML_dot)
_act_dot = casadi.vertcat( np.zeros([1,iM]), _act_dot)

# equality constraint on fiber length derivative and fiber velocity
opti.subject_to( _NML_dot / np.tile(AID['CVmax'],[iN,1]) == _NMV )

# equality constraint on muscle activation derivative and activation dynamics
opti.subject_to( _act_dot == firstOrderSmoothedActivationDynamics(_exc, _act) )

# normalized tendon force
_NTF = calcDeGrooteTendonForceLengthMultiplier(_NTL, 
    np.tile(AID['tendonStiffness'],  [iN,1]),
    np.tile(AID['tendonCurveShift'], [iN,1]) )

# normalized fiber force
_AMFLM = calcDeGrooteActiveFiberForceLengthMultiplier(_NML)
_PMFLM = calcDeGrootePassiveFiberForceLengthMultiplier(_NML)
_AMFVM = calcDeGrooteFiberForceVelocityMultiplier(_NMV)
_NMFAT = (_act*_AMFLM*_AMFVM + _PMFLM) * _PAcos

# equality constraint on fiber and tendon force equilibrium
opti.subject_to( _NTF == _NMFAT )

# equality constraint on moment equilibrium

# 'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',                                                                                                    
# 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',                                                                                                    
# 'knee_angle_r', 'knee_angle_l', 'ankle_angle_r', 'ankle_angle_l'

for ci,cName in enumerate(AID['include']): # loop over the coordinates
    _tau       = AID['tau'][:,ci] # joint moment from ID
    _momentArm = AID[cName]
    _moment    = casadi.sum2( _momentArm * _NTF*np.tile(AID['IFmax'],[iN,1]) ) # muscle moment
    opti.subject_to( _moment == _tau )

muscleExcitation = casadi.sumsqr(_exc) # 9.7998063e+01 (23) 8.835s

opti.minimize( muscleExcitation )

# plugin_options
p_opts = {
        'expand': True, 
        'detect_simple_bounds': True,
        # "print_time": False,
        }

# solver_options
s_opts = {
        'linear_solver': 'mumps',
        # 'print_level': 5,
        # 'print_timing_statistics': 'yes',
        'output_file': './IPOPT_output.txt',
        'max_iter':        1e+3,
        'tol':             1e-5,
        'constr_viol_tol': 1e-4,
        'compl_inf_tol':   1e-4,
        'print_user_options': 'yes',
        # 'hessian_approximation': 'limited-memory',
        # 'mu_strategy': 'adaptive',
        'nlp_scaling_method':'gradient-based',#'none'
        }
          
opti.solver('ipopt', p_opts, s_opts)

solver = opti.solve()
