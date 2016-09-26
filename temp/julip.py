"""
ASE to JuLIP.jl interface

Requires `pyjulia` package from https://github.com/JuliaPy/pyjulia
"""

import numpy as np
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer

from julia import Julia
julia = Julia()
julia.using("JuLIP")

# Workaround limitiation in PyCall that does not allow types to be called
#   https://github.com/JuliaPy/PyCall.jl/issues/319

ASEAtoms = julia.eval('ASEAtoms(a) = JuLIP.ASE.ASEAtoms(a)')
ASECalculator = julia.eval('ASECalculator(c) = JuLIP.ASE.ASECalculator(c)')
fixedcell = julia.eval('fixedcell(a) = JuLIP.Constraints.FixedCell(a)')

class JulipCalculator(Calculator):
    """
    ASE-compatible Calculator that calls JuLIP.jl for forces and energy
    """
    implemented_properties = ['forces', 'energy']
    default_parameters = {}
    name = 'JulipCalculator'

    def __init__(self, julip_calculator):
        Calculator.__init__(self)
        self.julip_calculator = julia.eval(julip_calculator)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        julia_atoms = ASEAtoms(atoms)
        self.results = {}
        if 'energy' in properties:
            self.results['energy'] = julia.energy(self.julip_calculator, julia_atoms)
        if 'forces' in properties:
            self.results['forces'] = np.array(julia.forces(self.julip_calculator, julia_atoms))


class JulipOptimizer(Optimizer):
    """
    Geometry optimize a structure using JuLIP.jl and Optim.jl
    """

    def __init__(self, atoms, restart=None, logfile='-',
                 trajectory=None, master=None, optimizer='JuLIP.Solve.ConjugateGradient'):
        """Parameters:

        atoms: Atoms object
            The Atoms object to relax.
        """
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)
        self.optimizer = julia.eval(optimizer)

    def run(self, fmax=0.05):
        """
        Run the optimizer to convergence
        """
        julia_atoms = ASEAtoms(self.atoms)

        # FIXME - if calculator is actually implemented in Julia, can skip this
        julia_calc = ASECalculator(self.atoms.get_calculator())
        julia.set_calculator_b(julia_atoms, julia_calc)

        if self.atoms.constraints != []:
            raise NotImplementedError("No support for ASE constraints yet!")
        julia.set_constraint_b(julia_atoms, fixedcell(julia_atoms))

        results = julia.minimise_b(julia_atoms, gtol=fmax, verbose=2)
        return results
