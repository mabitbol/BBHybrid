import numpy as np
from utils import *
import noise_calc as nc
import sacc
import healpy as hp
import pymaster as nmt
import glob

for sk in range(4):
    residuals = False
    masked = False
    nsims = 21

    nside = 256
    npix = hp.nside2npix(nside)

    fdir = f'/mnt/zfsusers/mabitbol/simdata/sims_gauss_fullsky_ns256_csd_std0.{sk}_gm3/'
    prefix_out = f'data/sim0{sk}/'

    if residuals:
        sname = 'residual'
        if masked:
            fnames = glob.glob(f'{prefix_out}masked_residualmaps*.fits')
        else: 
            fnames = glob.glob(f'{prefix_out}full_residualmaps*.fits')
    else:
        sname = 'baseline'
        fnames = glob.glob(f'{fdir}s*/maps_sky_signal.fits')
    fnames.sort()

    if masked:
        sname += '_masked'
        sat_mask = hp.ud_grade(hp.read_map('mask_apodized.fits'), nside)
    else: 
        sat_mask = np.ones(npix)

    nfreqs = 6
    npol = 2
    sens = 2
    knee = 1
    ylf = 1
    fsky = 0.1

    # Bandpasses
    bpss = {n: Bpass(n,f'data/bandpasses/{n}.txt') for n in band_names}

    # Bandpowers
    dell = 10
    nbands = 76
    lmax = 2+nbands*dell
    larr_all = np.arange(lmax+1)
    lbands = np.linspace(2, lmax, nbands+1, dtype=int)
    leff = 0.5*(lbands[1:]+lbands[:-1])
    windows = np.zeros([nbands, lmax+1])
    for b,(l0,lf) in enumerate(zip(lbands[:-1], lbands[1:])):
        windows[b,l0:lf] = (larr_all * (larr_all + 1)/(2*np.pi))[l0:lf]
        windows[b,:] /= dell
    s_wins = sacc.BandpowerWindow(larr_all, windows.T)

    # Precompute coupling matrix for namaster
    b = nmt.NmtBin.from_nside_linear(nside, dell, is_Dell=True)
    empty_field = nmt.NmtField(sat_mask, maps=None, spin=2, purify_e=False, purify_b=True)
    w_yp = nmt.NmtWorkspace()
    w_yp.compute_coupling_matrix(empty_field, empty_field, b)

    # Beams
    beams = {band_names[i]: b for i, b in enumerate(nc.Simons_Observatory_V3_SA_beams(larr_all))}

    # N_ell
    nell=np.zeros([nfreqs, lmax+1])
    _, nell[:,2:], _ = nc.Simons_Observatory_V3_SA_noise(sens, knee, ylf, fsky, lmax+1, 1)
    n_bpw = np.sum(nell[:, None, :]*windows[None, :, :], axis=2)
    bpw_freq_noi = np.zeros((nfreqs, npol, nfreqs, npol, nbands))
    for ib,n in enumerate(n_bpw):
        bpw_freq_noi[ib, 0, ib, 0, :] = n_bpw[ib, :]
        bpw_freq_noi[ib, 1, ib, 1, :] = n_bpw[ib, :]
    bpw_freq_noi_re = bpw_freq_noi.reshape([nfreqs*2, nfreqs*2, nbands])

    for kn in range(nsims):
        x = hp.read_map(fnames[kn], field=np.arange(nfreqs*npol), verbose=False)
        x[:, sat_mask==0] = 0
        y = x.reshape((nfreqs, npol, -1))
        T = np.ones((nfreqs, 1, x.shape[-1]))
        z = np.hstack((T, y))

        bpw_freq_sig = np.zeros((nfreqs, npol, nfreqs, npol, nbands))
        for i in range(nfreqs):
            for j in range(nfreqs):
                f2_1 = nmt.NmtField(sat_mask, y[i], purify_e=False, purify_b=True)
                f2_2 = nmt.NmtField(sat_mask, y[j], purify_e=False, purify_b=True)
                cl_coupled = nmt.compute_coupled_cell(f2_1, f2_2)
                cl_decoupled = w_yp.decouple_cell(cl_coupled)
                bpw_freq_sig[i, 0, j, 0] = cl_decoupled[0]
                bpw_freq_sig[i, 0, j, 1] = cl_decoupled[1]
                bpw_freq_sig[i, 1, j, 0] = cl_decoupled[2]
                bpw_freq_sig[i, 1, j, 1] = cl_decoupled[3]

        # Add to signal
        bpw_freq_tot = bpw_freq_sig+bpw_freq_noi
        bpw_freq_tot = bpw_freq_tot.reshape([nfreqs*2, nfreqs*2, nbands])
        bpw_freq_sig = bpw_freq_sig.reshape([nfreqs*2, nfreqs*2, nbands])

        # Creating Sacc files
        s_d = sacc.Sacc()
        s_f = sacc.Sacc()
        s_n = sacc.Sacc()

        # Adding tracers
        print("Adding tracers")
        for ib, n in enumerate(band_names):
            bandpass = bpss[n]
            beam = beams[n]
            for s in [s_d, s_f, s_n]:
                s.add_tracer('NuMap', 'band%d' % (ib+1),
                             quantity='cmb_polarization',
                             spin=2,
                             nu=bandpass.nu,
                             bandpass=bandpass.bnu,
                             ell=larr_all,
                             beam=beam,
                             nu_unit='GHz',
                             map_unit='uK_CMB')

        # Adding power spectra
        nmaps = 2*nfreqs
        ncross = (nmaps*(nmaps+1))//2
        indices_tr = np.triu_indices(nmaps)
        map_names = []
        for ib, n in enumerate(band_names):
            map_names.append('band%d' % (ib+1) + '_E')
            map_names.append('band%d' % (ib+1) + '_B')
        for ii, (i1, i2) in enumerate(zip(indices_tr[0], indices_tr[1])):
            n1 = map_names[i1][:-2]
            n2 = map_names[i2][:-2]
            p1 = map_names[i1][-1].lower()
            p2 = map_names[i2][-1].lower()
            cl_type = f'cl_{p1}{p2}'
            s_d.add_ell_cl(cl_type, n1, n2, leff, bpw_freq_sig[i1, i2, :], window=s_wins)
            s_f.add_ell_cl(cl_type, n1, n2, leff, bpw_freq_sig[i1, i2, :], window=s_wins)
            s_n.add_ell_cl(cl_type, n1, n2, leff, bpw_freq_noi_re[i1, i2, :], window=s_wins)

        # Add covariance
        cov_bpw = np.zeros([ncross, nbands, ncross, nbands])
        factor_modecount = 1./((2*leff+1)*dell*fsky)
        for ii, (i1, i2) in enumerate(zip(indices_tr[0], indices_tr[1])):
            for jj, (j1, j2) in enumerate(zip(indices_tr[0], indices_tr[1])):
                covar = (bpw_freq_tot[i1, j1, :]*bpw_freq_tot[i2, j2, :]+
                         bpw_freq_tot[i1, j2, :]*bpw_freq_tot[i2, j1, :]) * factor_modecount
                cov_bpw[ii, :, jj, :] = np.diag(covar)
        cov_bpw = cov_bpw.reshape([ncross * nbands, ncross * nbands])
        s_d.add_covariance(cov_bpw)

        # Write output
        print("Writing "+str(kn))
        s_d.save_fits(f'{prefix_out}cls_coadd_{sname}_{kn}.fits', overwrite=True)
        s_f.save_fits(f'{prefix_out}cls_fid_{sname}_{kn}.fits', overwrite=True)
        s_n.save_fits(f'{prefix_out}cls_noise_{sname}_{kn}.fits', overwrite=True)


