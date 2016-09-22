import numpy as np
from ase.calculators.calculator import Calculator

class JulipCalculator(Calculator):
    """
    ASE to JuLIP bridge

    Requires `pyjulia` package from https://github.com/JuliaPy/pyjulia
    """
    implemented_properties = ['forces', 'energy']
    default_parameters = {}
    name = 'JulipCalculator'

    def __init__(self, julip_calculator):
        Calculator.__init__(self)
        from julia import Julia
        self.julia = Julia()
        self.julia.using("JuLIP")
        self.julip_calculator = self.julia.eval(julip_calculator)

        # Workaround limitiation in PyCall that does not allow types to be called
        #   https://github.com/JuliaPy/PyCall.jl/issues/319
        self.convert_atoms = self.julia.eval('''
function convert_atoms(a)
   JuLIP.ASE.ASEAtoms(a)
end''')

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        julia_atoms = self.convert_atoms(atoms)
        self.results = {}
        if 'energy' in properties:
            self.results['energy'] = self.julia.energy(self.julip_calculator, julia_atoms)
        if 'forces' in properties:
            self.results['forces'] = np.array(self.julia.forces(self.julip_calculator, julia_atoms))

if __name__ == '__main__':
    from ase.build import bulk
    at = bulk("Cu")
    calc = JulipCalculator("JuLIP.Potentials.LennardJones()")
    at.set_calculator(calc)
    print at.get_potential_energy()
    print at.get_forces()
