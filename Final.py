import numpy as np  
import casadi

########## functions

def calcMuscleTendonForceMultipliers(AID, activity, NML, NMV, NTL, MTL, PAcos):
    '''
    muscle force-length-velocity and tendon force-length curves
    De Groote et al. (2016)
    '''
    # Tendon force-length multiplier (S1)
    c1=0.200; c2=0.995; c3=0.250
    kt = np.tile(AID['tendonStiffness'], [NTL.shape[0],1])
    shift = np.tile(AID['tendonCurveShift'], [NTL.shape[0],1])
    NTF = c1*np.exp(kt*(NTL-c2))-c3+shift

    # Active muscle force-length multiplier (S2)
    b11=+0.815; b21=+1.055; b31=+0.162; b41=+0.063
    b12=+0.433; b22=+0.717; b32=-0.030; b42=+0.200
    b13=+0.100; b23=+1.000; b33=+0.354; b43=+0.000 
    NFF1 = b11*np.exp(-0.5*(NML-b21)**2 / (b31+b41*NML)**2)
    NFF2 = b12*np.exp(-0.5*(NML-b22)**2 / (b32+b42*NML)**2)
    NFF3 = b13*np.exp(-0.5*(NML-b23)**2 / (b33+b43*NML)**2)
    AMFLM = NFF1+NFF2+NFF3

    # Pasive muscle force multiplier (S3)
    kp=4.0; e0=0.6
    PMFLM = (np.exp(kp*(NML-1)/e0)-1) / (np.exp(kp)-1)

    # Active muscle force-velocity multiplier (S4)
    d1=-0.318; d2=-8.149; d3=-0.374; d4=+0.886
    AMFVM = d1*np.log(d2*NMV+d3 + np.sqrt((d2*NMV+d3)**2+1))+d4   

    # Total muscle force-length-velocity force multiplier
    NMFAT = (activity*AMFLM*AMFVM + PMFLM) * PAcos

    return NTF, NMFAT



def ActivationDynamics(e,a, tact=0.015, tdeact=0.06, b=0.1):
    '''
    by Winters 1995
    De Groote et al. 2016 Eq(1) and (2)
    e       muscle excitation, 
    a       muscle activation,
    tact    activation time constant (0.015 s),
    tdeact  deactivation time constant (0.060 s),
    b       parameter determining transition smoothness (0.1).
    '''
    
    f = 0.5 * np.tanh(b*(e-a))
    constant = ((f+0.5)/(tact*(0.5+1.5*a)) + (0.5+1.5*a)/tdeact*(-f+0.5))
    return constant * (e-a) # == da/dt



def calcShift(kt):
    # force-length multipliers
    c1=0.200; c2=0.995; c3=0.250; shift=0
    TFLM35 = c1*np.exp(35*(1.-c2))-c3+0.
    TFLMkt = c1*np.exp(kt*(1.-c2))-c3+0.
    return TFLM35-TFLMkt



# data
AID = dict( np.load('./dataDynamicCASADI.npz') )

# adjust tendon stiffness (DeGroote muscle)
stiffness, shift = list(),list()
for mName in AID['nameMuscles']:
    if mName.split('_')[0] in ['soleus', 'gasmed', 'gaslat']: # triceps surae
        stiffness.append(15)
        shift.append( calcShift(15) )
    elif mName.split('_')[0] in ['recfem', 'vasint', 'vaslat', 'vasmed']: # quadriceps
        stiffness.append(20)
        shift.append( calcShift(20) )
    else:
        stiffness.append(35)
        shift.append( calcShift(35) )

AID['tendonStiffness']  = np.array(stiffness)
AID['tendonCurveShift'] = np.array(shift)

# parameters
iN, iM = AID['iMTL'].shape
iDt = 0.01 # time interval

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

_ML = _NML * np.tile(AID['MLopt'],[iN,1])
_MLAT = AID['iMTL'] - _NTL*np.tile(AID['TLslk'],[iN,1])
_PAcos = _MLAT / _ML

# bounds on on fiber pennation angle, min == 84.26deg or arccos(0.1)
opti.subject_to( 0.1 <= casadi.vec(_PAcos) )


########## constraints

# equality constraint on fiber geometry
opti.subject_to( _ML**2 == _MLAT**2 + np.tile(AID['height'],[iN,1])**2 )

# backward Euler method
_NML_dot = (_NML[1:,:]-_NML[:-1,:]) / iDt
_act_dot = (_act[1:,:]-_act[:-1,:]) / iDt
# my workaround, zeros for the first time frame 
_NML_dot = casadi.vertcat( np.zeros([1,iM]), _NML_dot)
_act_dot = casadi.vertcat( np.zeros([1,iM]), _act_dot)

# equality constraint on fiber length derivative and velocity
opti.subject_to( _NML_dot / np.tile(AID['CVmax'],[iN,1]) == _NMV )

# equality constraint on muscle activation derivative and activation dynamics
opti.subject_to( _act_dot == ActivationDynamics(_exc, _act, tact=0.015, tdeact=0.06, b=0.1) )

# equality constraint on fiber and tendon force equilibrium
NTF, NMFAT = calcMuscleTendonForceMultipliers(AID, _act, _NML, _NMV, _NTL, AID['iMTL'], _PAcos)
opti.subject_to( NTF == NMFAT )

# equality constraint on moment equilibrium

for ci,cName in enumerate(AID['include']):
    _tau       = AID['tau'][:,ci]
    _momentArm = AID[cName]
    _moment    = casadi.sum2( _momentArm * NMFAT*np.tile(AID['IFmax'],[iN,1]) )
    opti.subject_to( _moment == _tau )

muscleExcitation = casadi.sumsqr(_exc) # 1.0297743e+02 (32) 9.18s

opti.minimize( muscleExcitation )

# plugin_options
p_opts = {'expand':True, 
          'detect_simple_bounds':True}

# solver_options
s_opts = {'max_iter':1e+4, 
          'linear_solver':'mumps', 
          'tol':1e-6, 
          # 'hessian_approximation': 'limited-memory',
          'nlp_scaling_method':'gradient-based' }
          
opti.solver('ipopt', p_opts, s_opts)

solver = opti.solve()



