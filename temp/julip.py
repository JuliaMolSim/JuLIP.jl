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
variablecell = julia.eval('variablecell(a) = JuLIP.Constraints.VariableCell(a)')

class JulipCalculator(Calculator):
    """
    ASE-compatible Calculator that calls JuLIP.jl for forces and energy
    """
    implemented_properties = ['forces', 'energy', 'stress']
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
        if 'stress' in properties:
            self.results['stress'] = np.array(julia.stress(self.julip_calculator, julia_atoms))


class JulipOptimizer(Optimizer):
    """
    Geometry optimize a structure using JuLIP.jl and Optim.jl
    """

    def __init__(self, atoms, restart=None, logfile='-',
                 trajectory=None, master=None, variable_cell=False,
                 optimizer='JuLIP.Solve.ConjugateGradient'):
        """Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart, logfile ,trajector master : as for ase.optimize.optimize.Optimzer

        variable_cell : bool
            If true optimize the cell degresses of freedom as well as the
            atomic positions. Default is False.
        """
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)
        self.optimizer = julia.eval(optimizer)
        self.variable_cell = variable_cell

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
        if self.variable_cell:
            julia.set_constraint_b(julia_atoms, variablecell(julia_atoms))
        else:
            julia.set_constraint_b(julia_atoms, fixedcell(julia_atoms))

        # FIXME - should send output to logfile, and also add callback hook
        # to write frames to trajectory (Callbacks not yet supported by Optim.jl).
        results = julia.minimise_b(julia_atoms, gtol=fmax, verbose=2)
        return results
