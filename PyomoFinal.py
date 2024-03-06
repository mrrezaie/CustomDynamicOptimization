from pyomo.common.timing import report_timing
# report_timing()

import pyomo.environ as pyo
import numpy as np  


def calcFiberAndTendonForceDeGrootePyomo(activity, NML, NMV, NTL, MTL, MLopt, TLslk, stiffness):

    ML  = NML * MLopt
    TL  = NTL * TLslk
    MLAT= MTL - TL
    PAcos = MLAT / ML

    # Tendon force-length multiplier (S1)
    c1=0.235; c2=1.0; c3=0.235
    NTF = c1*pyo.exp(stiffness*(NTL-c2))-c3

    # Active muscle force-length multiplier (S2)
    b11=+0.8150671134243542
    b21=+1.055033428970575
    b31=+0.162384573599574
    b41=+0.063303448465465
    b12=+0.433004984392647
    b22=+0.716775413397760
    b32=-0.029947116970696
    b42=+0.200356847296188
    b13=+0.1
    b23=+1.0
    b33=+0.5*np.sqrt(0.5)
    b43=+0.0
    NFF1 = b11*pyo.exp(-0.5*(NML-b21)**2 / (b31+b41*NML)**2)
    NFF2 = b12*pyo.exp(-0.5*(NML-b22)**2 / (b32+b42*NML)**2)
    NFF3 = b13*pyo.exp(-0.5*(NML-b23)**2 / (b33+b43*NML)**2)
    AMFLM = NFF1+NFF2+NFF3

    # Pasive muscle force multiplier (S3)
    
    kp=4.0; e0=0.6
    PMFLM = (pyo.exp(kp*(NML-1)/e0)-1) / (pyo.exp(kp)-1) 
    PMFLM = 0 # the output is not perfect yet. Exclude it at the moment

    # Active muscle force-velocity multiplier (S4)
    d1=-0.3211346127989808
    d2=-8.149
    d3=-0.374
    d4=+0.8825327733249912
    # d1=-0.318; d2=-8.149; d3=-0.374; d4=+0.886
    AMFVM = d1*pyo.log(d2*NMV+d3 + pyo.sqrt((d2*NMV+d3)**2+1))+d4

    NMFAT = (activity*AMFLM*AMFVM + PMFLM) * PAcos
    return NTF, NMFAT


def ActivationDynamicsPyomo(e,a, tact=0.015, tdeact=0.06, b=0.1):

    '''
    by Winters 1995
    De Groote et al. 2016 Eq(1) and (2)
    e       muscle excitation, 
    a       muscle activation,
    tact    activation time constant (0.015 s),
    tdeact  deactivation time constant (0.060 s),
    b       parameter determining transition smoothness (0.1).
    '''
    import pyomo.environ as pyo
    f = 0.5 * pyo.tanh(b*(e-a))
    constant = ((f+0.5)/(tact*(0.5+1.5*a)) + (0.5+1.5*a)/tdeact*(-f+0.5))
    return constant * (e-a) # == da/dt


# data
AID = dict(np.load('./dataDynamic.npz'))

# adjust tendon stiffness (DeGroote muscle)
stiffness, shift = list(),list()
for mName in AID['nameMuscles']:
    if mName.split('_')[0] in ['soleus', 'gasmed', 'gaslat']: # triceps surae
        stiffness.append(15)
    elif mName.split('_')[0] in ['recfem', 'vasint', 'vaslat', 'vasmed']: # quadriceps
        stiffness.append(20)
    else:
        stiffness.append(35)

AID['tendonStiffness']  = np.array(stiffness)

iN, nMuscles = AID['iMTL'].shape
iDt = 0.01


# define solver
ipopt_path = 'D:/Academic/Codes/Python/CustomStaticOptimization/Ipopt-3.14.14-win64-msvs2019-md/bin/ipopt.exe'
solver = pyo.SolverFactory('ipopt', executable=ipopt_path)
solver.options['tol'] = 1e-6
solver.options['max_iter'] = 10000
solver.options['nlp_scaling_method'] = 'gradient-based'


# Define the model
pmo = pyo.ConcreteModel()

pmo.i     = pyo.RangeSet(0, iN-1) # or Set(initialize = [0,1,2])
pmo.mi    = pyo.RangeSet(0, nMuscles-1)
pmo.ci    = pyo.RangeSet(0, len(AID['include'])-1)
pmo.cName = pyo.Set(initialize=AID['include'])

# dicts for initial guesses
MTL_dict  = {(i,j): AID['iMTL'][i,j]  for i in pmo.i for j in pmo.mi}

zeros_dict = {(i,j): 0.   for i in pmo.i for j in pmo.mi}
ones_dict  = {(i,j): 1.   for i in pmo.i for j in pmo.mi}
ones1_dict = {(i,j): 0.1  for i in pmo.i for j in pmo.mi}

stiffness_dict = {(j): AID['tendonStiffness'][j]  for j in pmo.mi}

IFmax_dict  = {(j): AID['IFmax'][j]  for j in pmo.mi}
MLopt_dict  = {(j): AID['MLopt'][j]  for j in pmo.mi}
TLslk_dict  = {(j): AID['TLslk'][j]  for j in pmo.mi}
CVmax_dict  = {(j): AID['CVmax'][j]  for j in pmo.mi}
height_dict = {(j): AID['height'][j] for j in pmo.mi}
volume_dict = {(j): AID['volume'][j] for j in pmo.mi}

# Create parameters for initial guesses
pmo.MTL    = pyo.Param(pmo.i, pmo.mi, initialize=MTL_dict)
pmo.IFmax  = pyo.Param(pmo.mi, initialize=IFmax_dict)
pmo.MLopt  = pyo.Param(pmo.mi, initialize=MLopt_dict)
pmo.TLslk  = pyo.Param(pmo.mi, initialize=TLslk_dict)
pmo.CVmax  = pyo.Param(pmo.mi, initialize=CVmax_dict)
pmo.height = pyo.Param(pmo.mi, initialize=height_dict)
pmo.volume = pyo.Param(pmo.mi, initialize=volume_dict)
pmo.stiffness = pyo.Param(pmo.mi, initialize=stiffness_dict)

# Define the decision variables
pmo.act = pyo.Var(pmo.i, pmo.mi, bounds=(0, 1),     initialize=ones1_dict)
pmo.exc = pyo.Var(pmo.i, pmo.mi, bounds=(0, 1),     initialize=ones1_dict)
pmo.NML = pyo.Var(pmo.i, pmo.mi, bounds=(0.1, 1.9), initialize=ones_dict)
pmo.NMV = pyo.Var(pmo.i, pmo.mi, bounds=(-1, +1),   initialize=zeros_dict)
pmo.NTL = pyo.Var(pmo.i, pmo.mi, bounds=(1, None),  initialize=ones_dict)


######################################################################
# Pyomo function-based objective and constraints
######################################################################

# Define the constraints for the fiber geometry
def NML_constraint(model, i, mi):
    return (model.MTL[i,mi]-model.NTL[i,mi]*model.TLslk[mi])**2 + model.height[mi]**2 == (model.NML[i,mi]*model.MLopt[mi])**2
pmo.fiber_geometry_constraint = pyo.Constraint(pmo.i, pmo.mi, rule=NML_constraint)

# Define the constraints for the fiber geometry and fiber length derivative equations
def NMV_constraint(model, i, mi):
    if i == 0: 
        _NMLDot = 0
    else:
        _NMLDot = (model.NML[i,mi]-model.NML[i-1,mi])/iDt
    return _NMLDot / model.CVmax[mi] == model.NMV[i,mi]
pmo.fiber_velocity_constraint = pyo.Constraint(pmo.i, pmo.mi, rule=NMV_constraint)

# equality constraint on muscle activation derivative and activation dynamics
def exc_constraint(model, i, mi):
    if i == 0:
        _actDot =2
    else:
        _actDot = ((model.act[i,mi]-model.act[i-1,mi])/iDt)
    return _actDot == ActivationDynamicsPyomo(model.exc[i,mi], model.act[i,mi])
pmo.muscle_excitation_constraint = pyo.Constraint(pmo.i, pmo.mi, rule=exc_constraint)

# calculate normalized fiber and tendon forces
pmo.NTF, pmo.NMFAT = {},{}
def build_fiber_tendon_force(model, i, mi):
    model.NTF[i,mi] , model.NMFAT[i,mi] = calcFiberAndTendonForceDeGrootePyomo(
            model.act[i,mi], model.NML[i,mi], model.NMV[i,mi], model.NTL[i,mi], \
            model.MTL[i,mi], model.MLopt[mi], model.TLslk[mi], model.stiffness[mi])
pmo.updMTForce = pyo.BuildAction(pmo.i, pmo.mi, rule=build_fiber_tendon_force)

# equality constraint on fiber and tendon force equilibrium
def fiber_tendon_equilibrium(model, i, mi):
    return model.NTF[i,mi] == model.NMFAT[i,mi]
pmo.fiber_tendon_equilibrium = pyo.Constraint(pmo.i, pmo.mi, rule=fiber_tendon_equilibrium)

# equality constraint on moment equilibrium
def joint_moment_equilibrium(model, i, ci):
    _tau       = AID['tau'][i,ci] # float
    _momentArm = AID[AID['include'][ci]][i,:] # array
    moment     = pyo.quicksum( _momentArm[mi] * model.NMFAT[i,mi]*model.IFmax[mi] for mi in model.mi)
    return moment == _tau
pmo.joint_moment_equilibrium = pyo.Constraint(pmo.i, pmo.ci, rule=joint_moment_equilibrium)


# Define the objective function
def cost_func(model):
    muscleExcitation = pyo.quicksum(model.exc[i,mi]**2 for i in model.i for mi in model.mi)
    return muscleExcitation

pmo.objective = pyo.Objective(rule=cost_func, sense=pyo.minimize)


# Solve the model
results = solver.solve(pmo, report_timing=True, tee=True)