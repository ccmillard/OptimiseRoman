import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import scipy.stats as stats
import astropy.cosmology as ac
import pandas as pd
import sys
import astropy.constants as cst
from tqdm import tqdm
from tqdm.notebook import tqdm
from shutil import which

import fisher_matrix_analysis as fma

# ==================================================
# ================= PSD UTILITIES ==================
# ==================================================

def nearest_psd(A, eps=1e-10):
    """
    Project matrix onto symmetric PSD cone.
    """
    A = 0.5 * (A + A.T)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals[eigvals < eps] = eps # eigenvalue clipping
    A_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return 0.5 * (A_psd + A_psd.T)

def build_covariance(dist, dist_base, Cov_stat_base, Cov_sys, eps_psd=1e-10):
    """
    Build covariance from ORIGINAL statistical covariance
    and CURRENT distribution.
    """

    DD = np.diag(np.sqrt(dist_base / dist))

    Cov_stat = DD @ Cov_stat_base @ DD

    Cov = Cov_stat + Cov_sys

    Cov = nearest_psd(Cov, eps=eps_psd)

    return Cov

# =================================================
# ================= OPTIMIZATION ==================
# =================================================

def optimize_bins_gen(
    nb_iter,
    perturbation,
    dist_roman,
    Cov_roman,
    Cov_roman_stat,
    Cov_roman_sys,
    FOM,
    H0,
    z_roman,
    tt,
    fiducial_cosmo,
    cosmo_name,
    param_tuple, 
    param_names,
    use_marginalization=True,
    verbose=True
):
    """
    Optimization of the distribution of systems in redshift bins to maximize a figure of merit (FOM) 
    related to the cosmological constraints, under a fixed time constraint.

    Parameters:
        - nb_iter: number of iterations of the optimization process
        - perturbation: number of systems to remove from the lowest leverage bin at each iteration  
        - dist_roman: initial distribution of systems in redshift bins (array of shape (nb_bins,))
        - Cov_roman: initial covariance matrix (array of shape (nb_bins, nb_bins))
        - Cov_roman_stat: initial statistical covariance matrix (array of shape (nb_bins, nb_bins))
        - Cov_roman_sys: initial systematic covariance matrix (array of shape (nb_bins, nb_bins))
        - FOM: initial figure of merit to maximize (scalar)
        - H0: Hubble constant (scalar)
        - z_roman: array of redshift bin centers (array of shape (nb_bins,))
        - tt: array of observation time per system in each redshift bin (array of shape (nb_bins,))
        - fiducial_cosmo: fiducial cosmological parameters (array of shape (nb_params,))
        - cosmo_name: name of the cosmological model (string, e.g. 'lcdm' or 'fcpl')
        - param_tuple: tuple of parameter to keep in the Fisher matrix (e.g. ('w0', 'wa') for fCPL, or ('Om0', 'Ode0') for LCDM)
        - param_names: list of all parameter names in the Fisher matrix (e.g. ['Om0', 'w0', 'wa', 'Mb'])
        - use_marginalization: whether to marginalize the Fisher matrix over parameters not in param_tuple when computing the FOM (boolean)
        - verbose: whether to print information at each iteration (boolean) 

    Returns:
        - dist_optimized: optimized distribution of systems in redshift bins (array of shape (nb_bins,))
        - Dictionary containing the tracking of the optimization process, with keys:
            - "track_distribution": distribution of systems in redshift bins at each iteration (array of shape (nb_iter, nb_bins))
            - "track_delta_n": number of systems reallocated to each bin at each iteration (array of shape (nb_iter, nb_bins))
            - "track_Cov": covariance matrix at each iteration (array of shape (nb_iter, nb_bins, nb_bins))
            - "track_dFOM": dFOM for each bin at each iteration (array of shape (nb_iter, nb_bins))
            - "track_dFOM_triche": dFOM for each bin at each iteration, with bins that cannot be safely removed (between 2*perturbation and 2*perturbation + 1 systems) set to np.nan to ignore them in the choice of kk (array of shape (nb_iter, nb_bins))
            - "track_kk": index of the lowest leverage bin (kk) at each iteration (array of shape (nb_iter,))
            - "track_FF": Fisher matrix at each iteration (array of shape (nb_iter, nb_params, nb_params))
            - "track_ellipse": ellipse parameters at each iteration (array of shape (nb_iter, 4))
    """
    


    ss = len(fiducial_cosmo)

    # --------------------------------------------------
    # ----------------- INITIALISATION -----------------
    # --------------------------------------------------

    #----------
    # Trackers
    #----------
    track_ellipse = np.zeros((nb_iter, 4))
    track_FF = np.zeros((nb_iter, ss, ss))
    track_distribution = np.zeros((nb_iter, dist_roman.shape[0]))
    track_delta_n = np.zeros((nb_iter, dist_roman.shape[0]))
    track_Cov = np.zeros((nb_iter, Cov_roman.shape[0], Cov_roman.shape[1]))
    track_dFOM = np.zeros((nb_iter, len(z_roman)))
    track_dFOM_triche = np.zeros((nb_iter, len(z_roman)))
    track_kk = np.zeros((nb_iter))

    #-------------------------
    # Fixed baseline objects
    #-------------------------
    dist_base = np.copy(dist_roman).astype(float)
    Cov_stat_base = nearest_psd(np.copy(Cov_roman_stat))
    Cov_sys = nearest_psd(np.copy(Cov_roman_sys))

    # --------------------------------------------------
    # Current state
    # --------------------------------------------------

    dist_reference = np.copy(dist_roman).astype(float)
    #Cov_reference = np.copy(Cov_roman)
    II = np.eye(Cov_roman.shape[0])
    #Cinv_reference = np.linalg.solve(Cov_reference, II) 
    FOM_reference = np.copy(FOM)

    # Progress bar
    pbar = tqdm(range(nb_iter), desc="Optimizing bins")
    for nn in pbar:

        # ------------------------------------------------------
        # ---------------------- GRADIENT ----------------------
        # ------------------------------------------------------

        #------------------------------------
        # Initialise dFOM for this iteration
        #------------------------------------
        dFOM = np.zeros_like(z_roman)
        dFOM_triche = np.zeros_like(z_roman) # this will identify bins < 3*perturbation systems with np.nan, 
                                            # which we want to ignore in the choice of the lowest leverage bin (kk)

        #---------------------------------------------
        # Loop over bins to compute dFOM for each bin
        #---------------------------------------------
        for i in range(len(z_roman)):

            # initialize perturbed distribution as reference
            dist_perturbed = np.copy(dist_reference)

            min_bin_population = 2 * perturbation

            # if the bin has more than 2*perturbation systems, we can remove perturbation safely
            if dist_perturbed[i] > min_bin_population + 1: 
                time_freed = tt[i] * perturbation # time freed by 
                dist_perturbed[i] -= perturbation # removing perturbation systems from bin i

                # reallocation
                weights = np.ones_like(z_roman, dtype=float)
                weights[i] = 0  # cannot give back to itself
                weights /= weights.sum() # normalize weights
                for j in range(len(z_roman)):
                    if j != i:
                        delta_j = (time_freed * weights[j]) / tt[j] # uniform reallocation of freed time 
                        dist_perturbed[j] += delta_j
           
                # Propagate the perturbation to the covariance matrix
                Cov_perturbed = build_covariance(dist_perturbed, dist_base, Cov_stat_base, Cov_sys)
                Cov_perturbed = 0.5 * (Cov_perturbed + Cov_perturbed.T) # ensure symmetry
                Cinv_perturbed = np.linalg.solve(Cov_perturbed, II)

                # Compute the perturbed Fisher matrix and FOM
                FF_perturbed = fma.fisher_matrix_observable(
                    fiducial_cosmo,
                    Cinv_perturbed,
                    z_roman,
                    cosmo_name,
                    H0
                )

                FF_perturbed = nearest_psd(FF_perturbed)

                if use_marginalization:
                    # fCPL
                    CC_perturbed = fma.marginalize_fisher_matrix(
                        FF_perturbed, 
                        param_tuple, 
                        param_names)
                else:
                    CC_perturbed = np.linalg.inv(FF_perturbed)
                CC_perturbed = nearest_psd(CC_perturbed)
                ellipse = fma.ellipse_parameters(CC_perturbed, 0.32)
                FOM_perturbed = 1 / ellipse[3]

                # update dFOM and dFOM_triche       
                dFOM[i] = - (FOM_perturbed - FOM_reference) / tt[i]
                dFOM_triche[i] = dFOM[i] # system can be removed safely in bin i (without reaching 2*perturbation systems)

            # if the bin has between 2*perturbation and 2*perturbation + 1 systems, 
            # we can compute dFOM, but we will ignore the bin in the choice of kk by setting dFOM_triche to np.nan, 
            # because if we remove perturbation systems from this bin, we will reach the limit of 2*perturbation systems 
            # and won't be able to remove any more system from this bin in the next iterations
            # It is still interesting to have dFOM for the re-allocation once kk have been determined: otherwise, a bin reaching 
            # 2*perturbation systems would stay stuck to this value without ever being realloacted time.
            elif min_bin_population <= dist_perturbed[i] <= min_bin_population + 1: 
                # Compute dFOM as previously, but set dFOM_triche to np.nan to ignore this bin in the choice of kk
                time_freed = tt[i] * perturbation
                dist_perturbed[i] -= perturbation

                # rellocation
                weights = np.ones_like(z_roman, dtype=float)
                weights[i] = 0  # cannot give back to itself
                weights /= weights.sum() # normalize weights
                for j in range(len(z_roman)):
                    if j != i:
                        delta_j = (time_freed * weights[j]) / tt[j] # uniform reallocation of freed time 
                        dist_perturbed[j] += delta_j
                
                # Propagate the perturbation to the covariance matrix
                Cov_perturbed = build_covariance(dist_perturbed, dist_base, Cov_stat_base, Cov_sys)
                Cinv_perturbed = np.linalg.solve(Cov_perturbed, II)

                # Compute the perturbed Fisher matrix and FOM
                FF_perturbed = fma.fisher_matrix_observable(
                    fiducial_cosmo,
                    Cinv_perturbed,
                    z_roman,
                    cosmo_name,
                    H0
                )
                FF_perturbed = nearest_psd(FF_perturbed)
                if use_marginalization:
                    # fCPL
                    CC_perturbed = fma.marginalize_fisher_matrix(
                        FF_perturbed, 
                        param_tuple, 
                        param_names)
                else:
                    CC_perturbed = np.linalg.inv(FF_perturbed)
                CC_perturbed = nearest_psd(CC_perturbed)
                ellipse = fma.ellipse_parameters(CC_perturbed, 0.32)
                FOM_perturbed = 1 / ellipse[3]
                
                # update dFOM and dFOM_triche 
                dFOM[i] = -(FOM_perturbed - FOM_reference) / tt[i]
                dFOM_triche[i] = np.nan # ignore this bin in the choice of kk
            
            # if the bin has less than 2*perturbation systems
            # all bins are initialized with more than 2*perturbation systems, and we remove at most perturbation systems at each 
            # iteration, so if we are here it means that the input distribution has bins with less than 2*perturbation systems, 
            # and the optimization process cannot start properly
            else:
                print('Bin with less than 2 systems, change input distribution')
                break

        # ----------------------------------------------------------
        # ---------------------- OPTIMIZATION ----------------------
        # ----------------------------------------------------------

        #------------------------------------
        # Bin with lowest leverage: kk
        #------------------------------------

        kk = np.nanargmin(dFOM_triche)

        if verbose == True:
            print("Best index:", kk)
            print("Minimum dFOM:", dFOM[kk])

        if np.isnan(dFOM_triche).all():
            print(f"All entries in dFOM_triche are NaN at iteration {nn}")
            # this is absolutely not supposed to happen, if you get there someting went really wrong 
        
        # update trackers
        track_kk[nn] = kk
        track_dFOM[nn] = dFOM
        track_dFOM_triche[nn] = dFOM_triche
        
        #------------------------------------------------------------------------------------------
        # Choose bin to reallocate time to: only among bins with negative dFOM that are not bin kk
        #------------------------------------------------------------------------------------------

        # Find bins with negative dFOM
        gg = np.where(dFOM < 0)[0] 
        ignore = np.unique(np.sort(np.append(gg, kk))) # in the re-allocation, ignore bins with negative dFOM and bin kk 

        if verbose:
            print(f"At iter {nn}")
            print(f"Lowest leverage bin: {kk}")

        #-----------------------------------------
        # Remove perturbation systems from bin kk
        #-----------------------------------------
        
        #initialize new distribution as reference distribution
        dist_new = np.copy(dist_reference)

        if dist_new[kk] > perturbation: # safety check, should always be true because of the way we compute kk with dFOM_triche
            dist_new[kk] -= perturbation
            time_saved = tt[kk] * perturbation

            #------------------------------------------------------------------------------------------------------
            # Reallocate the time saved to other bins with negative dFOM, excluding kk and bins with positive dFOM
            #------------------------------------------------------------------------------------------------------

            # initialize delta_n for this iteration, which will store the number of systems reallocated to each bin 
            delta_n = np.zeros_like(z_roman)

            # define array of bins to reallocate time to: only bins with negative dFOM that are not bin kk (not in ignore)
            reallocate = np.array([
                j for j in range(len(z_roman))
                if j not in ignore
            ], dtype=int)

            denom = np.sum(dFOM[reallocate])
            for j in reallocate:
                delta_n[j] = (
                        perturbation * tt[kk] / tt[j]
                        * dFOM[j] / denom
                    )
                dist_new[j] = dist_new[j] + delta_n[j]
            time_reallocated = (delta_n * tt).sum()

            if verbose:
                print("Time saved:", time_saved)
                print("Time reallocated:", time_reallocated)
            
            # Propagate the perturbation to the covariance matrix
            Cov_new = build_covariance(dist_new, dist_base, Cov_stat_base, Cov_sys)
            Cinv_new = np.linalg.solve(Cov_new, II)

            if not fma.is_positive_definite(Cov_new):
                print("Cov not PSD at iter", nn)

            # Compute the new Fisher matrix and FOM
            FF_new = fma.fisher_matrix_observable(
                fiducial_cosmo,
                Cinv_new,
                z_roman,
                cosmo_name,
                H0
            )
            FF_new = nearest_psd(FF_new)

            # if model with more than 2 parameters, marginalize over all parameters you are not interested in 
            # right now, this is only designed for the fCPL case
            if use_marginalization:
                CC_new = fma.marginalize_fisher_matrix(
                    FF_new,
                    param_tuple,
                    param_names
                )
            else:
                CC_new = np.linalg.inv(FF_new)
            CC_new = nearest_psd(CC_new)
            ellipse_new = fma.ellipse_parameters(CC_new, 0.32)
            FOM_new = 1 / ellipse_new[3]

            # tracking
            track_delta_n[nn] = delta_n
            track_distribution[nn] = dist_new
            track_Cov[nn] = Cov_new
            track_ellipse[nn] = ellipse_new
            track_FF[nn] = FF_new

            # prepare next iteration
            dist_reference = dist_new
            Cov_reference = Cov_new
            FOM_reference = FOM_new

    optimized_dist = np.round(dist_reference).astype('int64')

    return {
        "optimized_dist": optimized_dist,
        "track_distribution": track_distribution,
        "track_delta_n": track_delta_n,
        "track_Cov": track_Cov,
        "track_dFOM": track_dFOM,
        "track_dFOM_triche": track_dFOM_triche,
        "track_kk": track_kk,
        "track_FF": track_FF,
        "track_ellipse": track_ellipse,
    }

