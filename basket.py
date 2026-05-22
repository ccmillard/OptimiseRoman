import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy as sc
import scipy.stats as stats
import astropy.cosmology as ac
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import numdifftools as nd
import sys
import random
from tqdm import tqdm
import time
import fisher_matrix_analysis as fma

# time cost function
def T_ngrt(z) :
    """Gives an estimation of the exposure time in seconds for a SNIa of redshift z.

    Parameter:
        z (float): redshift

    Returns:
        T(z): exposure time in s
    """
    
    z_bins=np.arange(0.15,1.7,0.1)
    exposure_time = np.array([335.8, 592.2, 939.2, 1422.2, 2002.2, 2304.4, 2631.6, 2914.2, 
                              3442.5, 4186.8, 4889.3, 5757.3, 6733.8, 7603.9, 8425.1, 8826.4])

    # quadratic fit
    res=np.polyfit(z_bins,exposure_time,2)
    a, b, c = res

    return a*z**2+b*z+c

def T_lsst(z):
    """Gives an estimation of the exposure time in seconds for a SNIa of redshift z.

    Parameter:
        z (float): redshift

    Returns:
        T(z): exposure time 
    """
    return (1+z)**8

def redistribute(distribution,z,time_basket,time_cost_function) :
    """Redistribute randomly N = size_basket SNIa across all redshift bins

    Parameters:
        distribution (np.ndarray): number of SNIa per redshift bin
        z (np.ndarray): mean value of redshift bins
        time_basket (float): SNIa 'cost' to redistribute
        time_cost_function (function): time cost function T(z) 

    Returns:
        alt_d (np.ndarray): altered distribution 
    """

    # copy distribution to be altered
    alt_d=np.copy(distribution)

    # remove SNIa
    t=0
    while t<time_basket :

        # randomly choose a bin
        i=random.randrange(0,len(distribution))

        if alt_d[i]>1: # bin can never drop to 0

            # remove one star from the bin
            alt_d[i]-=1

            # update observation time basket
            t+=time_cost_function(z[i])

        else:
            # chose a new bin
            pass
    
    # redistribute observation time
    t=0
    while t<time_basket :

        # randomly choose a bin
        i=random.randrange(0,len(distribution))

        # add one star to the bin
        alt_d[i]+=1

        # observation time spent
        t+=time_cost_function(z[i])

    return alt_d

def optimise_fisher_area(original_distribution, time_basket, time_cost_function, n_iter,
                         chi2, param_values, cosmology, H0, z, data_obs,
                         Cov_stat, Cov_sys, epsilon=1e-12, h=None, time_threshold=None):
    """
    Optimizes the SNIa redshift distribution to maximize the area of the Fisher matrix ellipse.

    Parameters:
        original_distribution (np.ndarray): Initial number of SNIa per redshift bin to optimize.
        time_basket (float): Observation time budget for redistribution.
        time_cost_function (function): Time cost function T(z).
        n_iter (int): Number of successful iterations to perform.
        chi2 : The chi-squared function for likelihood evaluation.
        param_values : List of cosmological parameters at fiducial values (e.g., [Om0, w0, wa, Mb]).
        cosmology : Cosmological model ('fLCDM', 'kLCDM', etc.).
        H0 : Hubble constant.
        z (np.ndarray): Redshift values.
        data_obs (np.ndarray): Observational data vector.
        Cov_stat (np.ndarray): Statistical covariance matrix.
        Cov_sys (np.ndarray): Systematic covariance matrix.
        epsilon (float): Small value to clip eigenvalues at (default: 1e-12).
        h (float or list): Step sizes for finite differences (default: 1e-4 for each parameter).
        time_threshold (float): Maximum allowed time (in seconds) to find one successful iteration. If exceeded, stops early.

    Returns:
        distribution (np.ndarray): Final optimized redshift distribution.
        list_of_covariance (np.ndarray): List of covariance matrices after each successful iteration.
        list_of_area (np.ndarray): List of Fisher ellipse areas after each iteration.
        list_of_cond_nb (np.ndarray): List of condition numbers of covariance matrices per iteration.
        list_of_fisher (list of np.ndarray): History of marginalised Fisher matrices.
    """

    # Initialization
    distribution = np.copy(original_distribution)
    Cov_tot = Cov_stat + Cov_sys

    F = fma.fisher_matrix(chi2, param_values, cosmology, H0, z, data_obs, Cov_tot, h=h)
    F_marginalised = fma.marginalize_fisher_matrix(F, ['w0', 'wa'], ['Om0', 'w0', 'wa', 'Mb'])
    C = np.linalg.inv(F_marginalised)
    area = fma.ellipse_parameters(C, 0.32, epsilon)[3]

    # track fisher evolution
    list_of_distribution = []
    list_of_C = []
    list_of_area = []
    list_of_cond_nb = []
    list_of_fisher = []

    # track Cov evolution
    list_of_Cov = []
    list_of_det = []
    list_of_Tr = []
    list_of_norm = []


    tqdm._instances.clear()
    progress = tqdm(total=n_iter, desc="Optimizing Fisher Matrix", unit="successful iter")

    n = 0

    while n < n_iter:
        search_start = time.time()  # Start timing the search for the next successful redistribution

        while True:
            # Check elapsed time during search
            elapsed_search = time.time() - search_start
            if time_threshold is not None and elapsed_search > time_threshold:
                print(f"Search for next successful iteration took {elapsed_search:.2f}s, exceeding {time_threshold:.2f}s. Stopping optimization early.")
                progress.close()
                return list_of_distribution, list_of_C, list_of_area, list_of_cond_nb, list_of_fisher, list_of_Cov, list_of_det, list_of_Tr, list_of_norm

            # Attempt a redistribution
            alt_distribution = redistribute(distribution, z, time_basket, time_cost_function)

            A = distribution[:, np.newaxis] @ distribution[np.newaxis, :]
            B = alt_distribution[:, np.newaxis] @ alt_distribution[np.newaxis, :]

            # modify stat and sys sperately
            alt_Cov_stat = np.sqrt(A / B) * Cov_stat
            alt_Cov_tot = alt_Cov_stat + Cov_sys

            # # modify them together
            # alt_Cov_tot = np.sqrt(A / B) * Cov_tot

            try:
                alt_F = fma.fisher_matrix(chi2, param_values, cosmology, H0, z, data_obs, alt_Cov_tot, h=h)
                alt_F_marginalised = fma.marginalize_fisher_matrix(alt_F, ['w0', 'wa'], ['Om0', 'w0', 'wa', 'Mb'])
                alt_C = np.linalg.inv(alt_F_marginalised)
                alt_area = fma.ellipse_parameters(alt_C, 0.32, epsilon)[3]
            except (np.linalg.LinAlgError, ValueError):
                # Numerical instability, try again
                continue

            # Accept if the new distribution is better (smaller area)
            if area >= alt_area:
                distribution = alt_distribution
                Cov_stat = alt_Cov_stat
                Cov_tot = alt_Cov_tot
                F = alt_F
                F_marginalised = alt_F_marginalised
                C = alt_C
                area = alt_area

                try:
                    cond_number = np.linalg.cond(C)
                    list_of_distribution.append(distribution)
                    list_of_C.append(C)
                    list_of_area.append(area)
                    list_of_cond_nb.append(cond_number)
                    list_of_fisher.append(F_marginalised)

                    list_of_Cov.append(Cov_stat)
                    list_of_det.append(np.linalg.det(Cov_stat))
                    list_of_Tr.append(np.trace(Cov_stat))
                    list_of_norm.append(np.linalg.norm(Cov_stat))

                except np.linalg.LinAlgError:
                    # If inversion fails, stop optimization
                    print("LinAlgError during condition number computation. Stopping optimization early.")
                    progress.close()
                    return list_of_distribution, list_of_C, list_of_area, list_of_cond_nb, list_of_fisher, list_of_Cov, list_of_det, list_of_Tr, list_of_norm


                n += 1
                progress.update(1)
                break  # Exit search loop and continue to next successful iteration

    progress.close()
    return list_of_distribution, list_of_C, list_of_area, list_of_cond_nb, list_of_fisher, list_of_Cov, list_of_det, list_of_Tr, list_of_norm



    

