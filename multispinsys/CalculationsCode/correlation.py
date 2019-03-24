import functools
import multiprocessing
import numpy as np
import re
import os
from spinsys import exceptions, utils
from scipy import sparse
import hamiltonians.triangular_lattice_model as t


@functools.lru_cache(maxsize=None)
def _sites(Nx, Ny, l):
    vec = t.SiteVector((0, 0), Nx, Ny)
    bonds = []
    xstride = l % Nx
    ystride = l // Nx
    for _ in range(Ny):
        for _ in range(Nx):
            bonds.append((vec.lattice_index, vec.xhop(xstride).yhop(ystride)
                          .lattice_index))
            vec = vec.xhop(1)
        vec = vec.yhop(1)
    site1, site2 = np.array(bonds).T
    return 2 ** site1, 2 ** site2


def sigmaz_op_consvd_k_reduced_basis_elements(Nx, Ny, kx, ky, i, l):
    ind_to_dec, dec_to_ind = t._gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
    state = ind_to_dec[i]
    site1, site2 = _sites(Nx, Ny, l)
    same_dir = 0
    for s1, s2 in zip(site1, site2):
        upup, downdown = t._repeated_spins(state.lead, s1, s2)
        same_dir += upup + downdown
    diff_dir = len(site1) - same_dir
    return 0.25 * (same_dir - diff_dir)


def sigmapm_op_consvd_k_reduced_basis_elements(Nx, Ny, kx, ky, i, l):
    ind_to_dec, dec_to_ind = t._gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
    orig_state = ind_to_dec[i]
    j_element = {}
    sites = _sites(Nx, Ny, l)
    for s1, s2 in zip(*sites):
        updown, downup = t._exchange_spin_flips(orig_state.lead, s1, s2)
        if updown or downup:
            if updown:  # if updown == True
                new_dec = orig_state.lead - s1 + s2
            elif downup:  # if downup == True
                new_dec = orig_state.lead + s1 - s2

            try:
                cntd_state, phase = t._find_leading_state(Nx, Ny, kx, ky, new_dec)
                j = dec_to_ind[cntd_state.lead]
                coeff = phase * t._coeff(Nx, Ny, kx, ky, orig_state, cntd_state)
                try:
                    j_element[j] += 0.5 * coeff
                except KeyError:
                    j_element[j] = 0.5 * coeff
            except exceptions.NotFoundError:  # connecting to a zero state
                pass
    return j_element


@utils.cache.matcache
def sigmaz_op_consvd_k_reduced_basis_matrix(Nx, Ny, kx, ky, l):
    return t._diag_components(Nx, Ny, kx, ky, l,
                              sigmaz_op_consvd_k_reduced_basis_elements)


@utils.cache.matcache
def sigmapm_op_consvd_k_reduced_basis_matrix(Nx, Ny, kx, ky, l):
    return t._offdiag_components(Nx, Ny, kx, ky, l,
                                 sigmapm_op_consvd_k_reduced_basis_elements)


def sz_corr(Nx, Ny, kx, ky, l, ψ):
    σz = sigmaz_op_consvd_k_reduced_basis_matrix(Nx, Ny, kx, ky, l)
    corr = ψ.T.conjugate().dot(σz).dot(ψ)[0, 0].real
    return corr / (Nx * Ny)


def spm_corr(Nx, Ny, kx, ky, l, ψ):
    σpm = sigmapm_op_consvd_k_reduced_basis_matrix(Nx, Ny, kx, ky, l)
    corr = ψ.T.conjugate().dot(σpm).dot(ψ)[0, 0].real
    return corr / (Nx * Ny)


def f(J_ppmm):
    print('Working on J2 = {}, J++--={}'.format(J2, round(J_ppmm, 4)))
    func = {'z': sz_corr, 'pm': spm_corr}
    path = '{}/eigvec_4x6_kx=0_ky=0_Jz=1_J+-=0.5_J++--={}_n={}.npy'
    # For some reason the saved vectors are 3-dimensional although only
    # one dimension is actually meaningful
    for n in range(2):
        V = np.load(path.format(src_dat_dir, round(J_ppmm, 4), n)).reshape(-1, 1)
        ψ = sparse.csc_matrix(V)
        data = np.empty((N, 3))
        for l in range(N):
            ix = l % Nx
            iy = l // Nx
            data[l, 0] = ix
            data[l, 1] = iy
            data[l, 2] = func[direction](Nx, Ny, kx, ky, l, ψ)
        fname = '{}/corr_{}_{}x{}_i={}_Jz={}_J+-={}_J++--={}_n={}.txt' \
            .format(save_dir, direction, Nx, Ny, i, J_z, J_pm, round(J_ppmm, 4), n)
        local_header = header.format(J_z, J_pm, round(J_ppmm, 4), direction)
        np.savetxt(fname, data, fmt='%10.7f', header=local_header,
                   footer=' ', comments='#')
        # print('load path: ', path.format(src_dat_dir, round(J_ppmm, 4), n), '\nsave path: ', fname)


Nx = 4
Ny = 6
N = Nx * Ny
kx = ky = 0

J_z = 1
J_pm = 0.5

i = 0

utils.cache.tmpdir = '/home/mlee/.cache'
J2s = np.linspace(0.05, 1.0, 20)
dat_dir_tmp = '/home/mlee/{}/{}x{}_Jz={}_J+-={}_J2={}'

header = 'Jz={}, J+-={}, J++--={}\nix         iy         S{} corr'
direction = 'pm'
nthreads = 11

for J2 in J2s:
    J2 = round(J2, 2)
    print('J2 = {}'.format(J2))
    src_dat_dir = dat_dir_tmp.format('eigvecs', Nx, Ny, J_z, J_pm, J2)
    save_dir = dat_dir_tmp.format('correlations', Nx, Ny, J_z, J_pm, J2)
    # if not os.path.exists(src_dat_dir):
    #     os.mkdir(src_dat_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    J_ppmms = re.findall(r'\S+J\+\+--=([0-9.-]+)\S+', ' '.join(os.listdir(src_dat_dir)))
    J_ppmms = sorted(map(float, set(J_ppmms)))
    # print(J_ppmms)

    os.environ['MKL_NUM_THREADS'] = str(nthreads)
    f(J_ppmms[0])
    os.environ['MKL_NUM_THREADS'] = '1'
    with multiprocessing.Manager() as manager:
        with multiprocessing.Pool(nthreads) as p:
            p.map(f, J_ppmms[1:])

    print('Done.')
