import os
import numpy as np
import scipy
import krotov
import qutip

from qutip import tensor,qeye,ket
from multiprocessing import Pool

# define Hamiltonian : The total system has two parts a time-independent part and a control part
# the control part of the system, is where the pulses are to be applied to generate the target state!
def hamiltonian_parts():
    z = qutip.operators.sigmaz()
    x = qutip.operators.sigmax()
    e = qeye(2)
    
    x1x2 = tensor(x,x,e)
    x2x3 = tensor(e,x,x)
    z1z2 = tensor(z,z,e)
    
    w1 = 5.0
    w2 = 5.3
    w3 = 5.5
    
    k1 = 0.112358
    k2 = 0.222222
    k3 = 0.314159
    
    drift = -1/2 * ( w1*tensor(z,e,e) + w2*tensor(e,z,e) + w3*tensor(e,e,z) ) 
    coupl = k1*x1x2 + k2*z1z2 + k3*x2x3
    
    h0 = drift + coupl
    h0 = np.array(h0)*2*np.pi  # important for dimensionality. Code recognizes the 8 x 8 matrix instead of a (2x2x2) x (2x2x2) tensor
    h0 = qutip.Qobj(h0) 
    
    h2 = tensor(e,x,e)
    h2 = np.array(h2)
    h2 = qutip.Qobj(h2)
    return h0,h2

H0,H1 = hamiltonian_parts()


# define class KrotovSys
class KrotovSys:
    
    def __init__(self,final_time,t_rise,delta_t,lambda_0,target,H0,H1):
        self.final_time = final_time
        self.t_rise = t_rise
        self.delta_t = delta_t
        
        # number of grid points depending on final_time, delta_t, N = final_time/delta_t + 1
        self.tlist  = np.linspace(0, self.final_time, int( self.final_time/self.delta_t + 1) )
        
        self.T = self.tlist[-1]
        self.lambda_0 = lambda_0
        self.target = target
        
      
        self.H0 = H0
        self.H1 = H1
        #self.guess_pulse = guess_pulse
        
    def guess_pulse(self,t,args):
        self.t = t
        return 2*np.pi*np.exp(-60.0 * (t /self.T - 0.5) ** 2)*(
            np.cos(0*t) + np.cos(0.2*t) + np.cos(0.3*t) + np.cos(0.5*t) + np.cos(5.0*t) + np.cos(5.3*t) + np.cos(5.5*t) + np.cos(4.8*t) + np.cos(5.8*t) + np.cos(10.3*t) + np.cos(10.5*t) + np.cos(10.8*t) + np.cos(15.8)
        )
        
    def qub_hamiltonian(self):
        return [self.H0, [self.H1, self.guess_pulse]]
    
    #def logical_basis(self):
     #   """
        # Define logical basis (In this case the initial basis is coincidentally the canonical basis
        # eigenvectors are columns within V"
     #   """
        
      #  I = qeye(8)
        
       # eigenvals, eigenvecs = scipy.linalg.eig(I)
        
        #ndx = np.argsort(eigenvals.real)
        
        #E = eigenvals[ndx].real
        
        #V = eigenvecs[:, ndx]
        
        #return [qutip.Qobj(row) for row in V.transpose()] 
    
    def logical_basis(self):
        """
        # Define logical basis (In this case the initial basis is coincidentally the canonical basis
        # eigenvectors are columns within V"
        """
        
        return [qutip.basis(8,i) for i in range(0,8)]
      
    

    def projectors(self):
        # Define projectors, because the Krotov routine must take the basis states as density matrices!
        return [qutip.ket2dm(k) for k in self.logical_basis()]
    
    def shape(self, t):
        # Define the shape function S(t). To put it casually, this function can be understood as a template in which optimisation         is allowed. For instance, if the guess pulse is partly inside the template and partly outside, then the function
        #is only optimised inside the template and remains unchanged outside the template.
        self.t = t
        """Scales the Krotov methods update of the pulse value at the time t"""
        return krotov.shapes.flattop(t, t_start=0.0, t_stop=self.final_time, t_rise = self.t_rise, func='blackman')
 
    def krotov_routine(self):
       # The H1 part of the system, defined in hamiltonian_parts() is to be manipulated by the pulse,
       # throughout the whole krotov routine
        
        pulse_options = {self.qub_hamiltonian()[1][1]: dict(lambda_a=self.lambda_0 , update_shape=self.shape)}
        
        objectives = krotov.gate_objectives( basis_states=self.logical_basis(), 
                                            gate = self.target, H=self.qub_hamiltonian())
        
        guess_dynamics = [ objectives[x].mesolve(self.tlist, e_ops = self.projectors()) for x in [0, 1, 2, 3, 4, 5, 6, 7]]
        
        opt_result = krotov.optimize_pulses(objectives, pulse_options, self.tlist, propagator=krotov.propagators.expm,
        chi_constructor=krotov.functionals.chis_sm,
        info_hook=krotov.info_hooks.print_table(
            J_T=krotov.functionals.J_T_sm,
            show_g_a_int_per_pulse=True,
            unicode=False,
        ),
        check_convergence=krotov.convergence.Or(
            krotov.convergence.value_below(1e-4, name='J_T'),
            krotov.convergence.delta_below(1e-7),
            krotov.convergence.check_monotonic_error,
        ),
        iter_stop = 2,
    )
        print(opt_result)
        
        return opt_result
    
""" Globally defined functions """


def save_data(opt_result,path_to_file1 = None):
    
    final_time = opt_result.tlist[-1]
    final_t = str(final_time)
    
    
    path_to_file1 = f"./{final_t}_opt_result.dump"
    
    if path_to_file1:
        opt_result.dump(path_to_file1)
        
# The function "single_optimization" takes a list that consists of final_time,t_rise,grid_points,lambda_0,target as an Input and does the Krotov_routine: It optimizes towards the target state with respect to the qub_Hamiltonian and the guess_pulse
# We need to save various outputs, which is done with the save_data function


def single_optimization(L,Ham0 = H0,Ham1 = H1):
    
    system_1 = KrotovSys(L[0],L[1],L[2],L[3],L[4],Ham0,Ham1)
    
    opt_result = system_1.krotov_routine()
    
    save_data(opt_result)
                  
# the function "do_final_times"  is a very important function, since we want to see if the optimization can be succesfully performed for shorter final times. Hence we create an numpy array that consists of multiple final times in decreasing order.
# The idea is, if the optimization was succesful for a certain final_time t_f, we want to observe if it is succesful for the final time t_f - b and subsequently for t_f - 2*b, all the way until t_f - c*b.


def do_final_times(a, b, c):
    """
    a = final time
    b = decreasing parameter
    c = how often shall b be reduced from a
    """
    return np.arange(a, 0, -b)[:c]


# do_optimization_input: Here, we create c Lists, where each list consists of final_time, t_rise, delta_t, lambda_0 and the target. For each final_time that was created in "do_final_times" exists a list P. Hence, there are c Lists in total! Each list P contains information that is necessary for doing the "single optimization - routine"


def do_optimization_input(L,b,c):
 
    a = L[0]
    P = [[m] + L[1:5] for m in do_final_times(a, b, c)]
    
    return P

# do_krotov_parallel: The Idea here is to make life easier and safe time, therefore we want to parallelize the optimizations! 
# In the parallelization, we simultaneously perform the krotov algorithm with respect to each of the c Lists 
# More technical, for each of the c Lists that was created in "do_optimization_input", we can perforn an optimization using "single_optimization"


def do_krotov_parallel(L,b,c,n=16):
    P = do_optimization_input(L,b,c)    #n number of threads
    with  Pool(n) as p:
        results = p.map(single_optimization,P)
        
#define TARGET
HAD = qutip.operations.hadamard_transform(N=1)
E = qeye(2)
TARGET = qutip.Qobj( np.array(tensor(HAD,E,E)))

        
L = [14, 5 ,0.01 ,0.2 ,TARGET]

do_krotov_parallel(L,1,1)


