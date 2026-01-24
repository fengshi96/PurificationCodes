"""
implement extended Kitaev on a honeycomb lattice
Sign Convention:
H = J1 SS + J2 SS
"""
from tenpy.networks.site import SpinHalfSite
# from tenpy.linalg import np_conserved as npc
from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Triangular
import math



class J1J2(CouplingMPOModel):
    r"""
    J1 SS + J2 SS
    """
    def init_lattice(self, model_params):
        Lx = model_params.get('Lx', 0.)
        Ly = model_params.get('Ly', 0.)
        conserve=model_params.get('conserve', 'Sz')
        bc_MPS = model_params.get('bc_MPS', 'finite')
        bc = model_params.get('bc', ['open', 'periodic'])
        order = model_params.get('order', 'default')
        site = SpinHalfSite(conserve=conserve)
        lat = Triangular(Lx, Ly, site, bc=bc, bc_MPS=bc_MPS, order=order)
        return lat

    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_terms(self, model_params):
        # See https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.033011 (Eq. 2)
        # 0) read out/set default parameters
        J1 = model_params.get('J1', 1.)
        J2 = model_params.get('J2', 1.)  
        Fz = model_params.get('Fz', 0.)  # magnetic field in z

        # nearest_neigbor Heisenberg terms
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(J1, u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling(J1/2, u1, 'Sp', u2, 'Sm', dx)
            self.add_coupling(J1/2, u1, 'Sm', u2, 'Sp', dx)



        # next_nearest_neigbor Heisenberg terms
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling(J2, u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling(J2/2, u1, 'Sp', u2, 'Sm', dx)
            self.add_coupling(J2/2, u1, 'Sm', u2, 'Sp', dx)


        # magnetic field:
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(Fz, u, 'Sz')



    def draw(self, ax, coupling=None, wrap=False):
        # for drawing lattice with MPS indices
        self.lat.plot_coupling(ax, coupling=None, wrap=False)
        self.lat.plot_basis(ax, origin=(-0.0, -0.0), shade=None)
        print(self.lat.bc)
        # self.lat.plot_bc_identified(ax, direction=-1, origin=None, cylinder_axis=True)
        self.lat.plot_order(ax)

    def positions(self):
        # return space positions of all sites
        return self.lat.mps2lat_idx(range(self.lat.N_sites))




# ----------------------------------------------------------
# ----------------- Debug Kitaev_Extended() ----------------
# ----------------------------------------------------------
def test(**kwargs):
    print("test module")
    Lx = kwargs['Lx']
    Ly = kwargs['Ly']
    order = kwargs.get('order', 'default')
    J1 = kwargs['J1']
    J2 = kwargs['J2']
    Fz = kwargs['Fz']
    conserve = kwargs.get('conserve', 'Sz')
    # bc = 'open'

    model_params = dict(Lx=Lx, Ly=Ly, order=order, conserve=conserve, J1=J1, J2=J2, Fz=Fz)
    M = J1J2(model_params)
    
    pos = M.positions()
    print("pos:\n", pos, "\n")
    
    test = np.array([(str(i)) for i in pos])
    retest = test.reshape(model_params['Lx'],-1).T
    print(retest,'\n')
    print(retest[::2,:])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1,  figsize=(8,6))     
    M.draw(ax)
    plt.savefig("./figure.pdf", dpi=300, bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    import numpy as np
    import sys
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    # print(cmdargs[1])

    Lx = 24
    Ly = 6
    J1 = 1.0
    J2 = 1.0
    Fx = 0
    Fy = 0
    Fz = 0
    conserve = 'Sz'
    order = 'default' 
    bc = ['open', 'periodic']
    test(Lx=Lx, Ly=Ly, order=order, conserve=conserve, bc=bc, J1=J1, J2=J2, Fz=Fz)
