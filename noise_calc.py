from __future__ import print_function
import numpy as np

####################################################################
####################################################################
### SAT CALCULATOR ###
####################################################################
####################################################################
def Simons_Observatory_V3_SA_bands():
    ## returns the band centers in GHz for a CMB spectrum
    ## if your studies require color corrections ask and we can estimate these for you
    return(np.array([27.,39.,93.,145.,225.,280.]))

def Simons_Observatory_V3_SA_beam_FWHM():
    ## returns the SAT beams in arcminutes
    beam_SAT_27 = 91.
    beam_SAT_39 = 63.
    beam_SAT_93 = 30.
    beam_SAT_145 = 17.
    beam_SAT_225 = 11.
    beam_SAT_280 = 9.
    return(np.array([beam_SAT_27,beam_SAT_39,beam_SAT_93,beam_SAT_145,beam_SAT_225,beam_SAT_280]))

def Simons_Observatory_V3_SA_beams(ell):
    SA_beams = Simons_Observatory_V3_SA_beam_FWHM() / np.sqrt(8. * np.log(2)) /60. * np.pi/180.
    ## SAT beams as a sigma expressed in radians
    return [np.exp(-0.5*ell*(ell+1)*sig**2.) for sig in SA_beams]

def Simons_Observatory_V3_SA_noise(sensitivity_mode,one_over_f_mode,SAT_yrs_LF,f_sky,ell_max,delta_ell,
                                   include_kludge=True, include_beam=True):
    ## returns noise curves in polarization only, including the impact of the beam, for the SO small aperture telescopes
    ## noise curves are polarization only
    # sensitivity_mode
    #     1: baseline, 
    #     2: goal
    # one_over_f_mode
    #     0: pessimistic
    #     1: optimistic
    # SAT_yrs_LF: 0,1,2,3,4,5:  number of years where an LF is deployed on SAT
    # f_sky:  number from 0-1
    # ell_max: the maximum value of ell used in the computation of N(ell)
    # delta_ell: the step size for computing N_ell
    ####################################################################
    ####################################################################
    ###                        Internal variables
    ## SMALL APERTURE
    # ensure valid parameter choices
    assert( sensitivity_mode == 1 or sensitivity_mode == 2)
    assert( one_over_f_mode == 0 or one_over_f_mode == 1)
    assert( SAT_yrs_LF <= 5) #N.B. SAT_yrs_LF can be negative
    assert( f_sky > 0. and f_sky <= 1.)
    assert( ell_max <= 2e4 )
    assert( delta_ell >= 1 )
    # configuration
    if (SAT_yrs_LF > 0):
        NTubes_LF  = SAT_yrs_LF/5. + 1e-6  ## regularized in case zero years is called
        NTubes_MF  = 2 - SAT_yrs_LF/5.
    else:
        NTubes_LF  = np.fabs(SAT_yrs_LF)/5. + 1e-6  ## regularized in case zero years is called
        NTubes_MF  = 2 
    NTubes_UHF = 1.
    # sensitivity
    # N.B. divide-by-zero will occur if NTubes = 0
    # handle with assert() since it's highly unlikely we want any configurations without >= 1 of each tube type
    assert( NTubes_LF > 0. )
    assert( NTubes_MF > 0. )
    assert( NTubes_UHF > 0.)
    S_SA_27  = np.array([1.e9,21,15])    * np.sqrt(1./NTubes_LF)
    S_SA_39  = np.array([1.e9,13,10])    * np.sqrt(1./NTubes_LF)
    S_SA_93  = np.array([1.e9,3.4,2.4]) * np.sqrt(2./(NTubes_MF))
    S_SA_145 = np.array([1.e9,4.3,2.7]) * np.sqrt(2./(NTubes_MF))
    S_SA_225 = np.array([1.e9,8.6,5.7])  * np.sqrt(1./NTubes_UHF)
    S_SA_280 = np.array([1.e9,22,14])    * np.sqrt(1./NTubes_UHF)
    # 1/f polarization noise
    # see Sec. 2.2 of the SO science goals paper
    f_knee_pol_SA_27  = np.array([30.,15.])
    f_knee_pol_SA_39  = np.array([30.,15.])  ## from QUIET
    f_knee_pol_SA_93  = np.array([50.,25.])
    f_knee_pol_SA_145 = np.array([50.,25.])  ## from ABS, improvement possible by scanning faster
    f_knee_pol_SA_225 = np.array([70.,35.])
    f_knee_pol_SA_280 = np.array([100.,40.])
    alpha_pol =np.array([-2.4,-2.4,-2.5,-3,-3,-3])
    
    ####################################################################
    ## calculate the survey area and time
    t = 5* 365. * 24. * 3600    ## five years in seconds
    t = t * 0.2  ## retention after observing efficiency and cuts
    if include_kludge:
        t = t* 0.85  ## a kludge for the noise non-uniformity of the map edges
    A_SR = 4 * np.pi * f_sky  ## sky area in steradians
    A_deg =  A_SR * (180/np.pi)**2  ## sky area in square degrees
    A_arcmin = A_deg * 3600.
    #print("sky area: ", A_deg, "degrees^2")
    #print("Note that this code includes a factor of 1/0.85 increase in the noise power, corresponding to assumed mode loss due to map depth non-uniformity.")
    #print("If you have your own N_hits map that already includes such non-uniformity, you should increase the total integration time by a factor of 1/0.85 when generating noise realizations from the power spectra produced by this code, so that this factor is not mistakenly introduced twice.")
    
    ####################################################################
    ## make the ell array for the output noise curves
    ell = np.arange(2,ell_max,delta_ell)
    
    ####################################################################
    ###   CALCULATE N(ell) for Temperature
    ## calculate the experimental weight
    W_T_27  = S_SA_27[sensitivity_mode]  / np.sqrt(t)
    W_T_39  = S_SA_39[sensitivity_mode]  / np.sqrt(t)
    W_T_93  = S_SA_93[sensitivity_mode]  / np.sqrt(t)
    W_T_145 = S_SA_145[sensitivity_mode] / np.sqrt(t)
    W_T_225 = S_SA_225[sensitivity_mode] / np.sqrt(t)
    W_T_280 = S_SA_280[sensitivity_mode] / np.sqrt(t)
    
    ## calculate the map noise level (white) for the survey in uK_arcmin for temperature
    MN_T_27  = W_T_27  * np.sqrt(A_arcmin)
    MN_T_39  = W_T_39  * np.sqrt(A_arcmin)
    MN_T_93  = W_T_93  * np.sqrt(A_arcmin)
    MN_T_145 = W_T_145 * np.sqrt(A_arcmin)
    MN_T_225 = W_T_225 * np.sqrt(A_arcmin)
    MN_T_280 = W_T_280 * np.sqrt(A_arcmin)
    Map_white_noise_levels = np.array([MN_T_27,MN_T_39,MN_T_93,MN_T_145,MN_T_225,MN_T_280])
    #print("white noise levels (T): ",Map_white_noise_levels ,"[uK-arcmin]")
    
    ####################################################################
    ###   CALCULATE N(ell) for Polarization
    ## calculate the atmospheric contribution for P
    ## see Sec. 2.2 of the SO science goals paper
    AN_P_27  = (ell / f_knee_pol_SA_27[one_over_f_mode] )**alpha_pol[0] + 1.  
    AN_P_39  = (ell / f_knee_pol_SA_39[one_over_f_mode] )**alpha_pol[1] + 1. 
    AN_P_93  = (ell / f_knee_pol_SA_93[one_over_f_mode] )**alpha_pol[2] + 1.   
    AN_P_145 = (ell / f_knee_pol_SA_145[one_over_f_mode])**alpha_pol[3] + 1.   
    AN_P_225 = (ell / f_knee_pol_SA_225[one_over_f_mode])**alpha_pol[4] + 1.   
    AN_P_280 = (ell / f_knee_pol_SA_280[one_over_f_mode])**alpha_pol[5] + 1.  

    ## calculate N(ell)
    N_ell_P_27   = (W_T_27  * np.sqrt(2))**2.* A_SR * AN_P_27
    N_ell_P_39   = (W_T_39  * np.sqrt(2))**2.* A_SR * AN_P_39
    N_ell_P_93   = (W_T_93  * np.sqrt(2))**2.* A_SR * AN_P_93
    N_ell_P_145  = (W_T_145 * np.sqrt(2))**2.* A_SR * AN_P_145
    N_ell_P_225  = (W_T_225 * np.sqrt(2))**2.* A_SR * AN_P_225
    N_ell_P_280  = (W_T_280 * np.sqrt(2))**2.* A_SR * AN_P_280

    if include_beam:
        ## include the impact of the beam
        SA_beams = Simons_Observatory_V3_SA_beams(ell)
        ## SAT beams as a sigma expressed in radians
        N_ell_P_27  /= SA_beams[0]**2
        N_ell_P_39  /= SA_beams[1]**2.
        N_ell_P_93  /= SA_beams[2]**2.
        N_ell_P_145 /= SA_beams[3]**2.
        N_ell_P_225 /= SA_beams[4]**2.
        N_ell_P_280 /= SA_beams[5]**2.
    
    ## make an array of noise curves for P
    N_ell_P_SA = np.array([N_ell_P_27,N_ell_P_39,N_ell_P_93,N_ell_P_145,N_ell_P_225,N_ell_P_280])
    
    ####################################################################
    return(ell,N_ell_P_SA,Map_white_noise_levels)
