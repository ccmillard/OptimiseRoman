import jax
jax.config.update("jax_enable_x64", True) #MAYBE CHANGE FOR FASTER CALCULATIONS ????

import jax.numpy as jnp
import jax_cosmo as jc
import jax.scipy as jsp
from jaxopt import LBFGS
import pandas as pd

from astropy.io import fits
import time
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


"""

#LOADING MOCK DATA:
d = np.load("/Users/edwyn.howarth/Desktop/SNIa/NGR_fLCDM.npz")

z = jnp.array(d["z"])
mb_panth = jnp.array(d["mb"])

hdul = fits.open("/Users/edwyn.howarth/Desktop/SNIa/roman_cov.fits") #for covariance matrix

A = hdul[0].data
A = np.array(A, dtype=np.float64)   # convert endian + dtype
C = A[1:, :]     # remaining rows

cov_jc = jnp.array(C)



Omega_m = 0.31
sigma8 = 0.8
M_B = -19.5
h = 0.70

params = jnp.array([Omega_m, M_B])

@jax.jit
def theory_mb_jc(params,z):
    cosmo = jc.Cosmology(
        Omega_c=params[0]-0.05,
        Omega_b=0.05,
        h=h,
        sigma8=sigma8,
        n_s=0.96,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0
    )

    # scale factor
    a = 1.0 / (1.0 + z)

    chi = jc.background.radial_comoving_distance(cosmo, a, steps=10000) #Mpc/h, FUNCTION MAKES GRID AND THEN USES INTERPOLATION, SO NUMBER OF STEPS IS IMPORTANT!!!! 4096 256
    chi = chi / h 
    dL = chi / a 

    #jax.debug.print("dL = {}", dL)

    mu = 5.0 * jnp.log10(dL) + 25.0

    return mu + params[1]


cov_inv_jc = jnp.linalg.inv(cov_jc)

@jax.jit
def chi2_jc(params):
    mb_theory = theory_mb_jc(params)
    r = mb_panth - mb_theory
    chi2 = r @ cov_inv_jc @ r
    return chi2

params0 = params
params1 = jnp.array([Omega_m+0.01, M_B+0.1])


#start1 = time.perf_counter()

#solver = LBFGS(fun=chi2_jc, maxiter=100)
#res_jc = solver.run(params0)
#best_jc = res_jc.params

#end1 = time.perf_counter()

#print("chi2_jc : ",chi2_jc(best_jc),"  avec Omega_m : ",best_jc[0],"  et M_B : ",best_jc[1])
#print(f"solver Execution time: {end1 - start1:.4f} seconds")




@jax.jit
def loglike_jc(params):
    yess = chi2_jc(params)
    return -0.5*yess

@jax.jit
def fisher_matrix(params):
    hessian_loglike = jax.jit(jax.hessian(loglike_jc))
    return - hessian_loglike(params)


start2 = time.perf_counter()
F1 = fisher_matrix(params0)
jax.block_until_ready(F1)
end2 = time.perf_counter()
print(f"FISHER MATRIX JC Execution time: {end2 - start2:.4f} seconds")

start2 = time.perf_counter()
F = fisher_matrix(params1)
jax.block_until_ready(F)
end2 = time.perf_counter()
print(f"FISHER MATRIX JC Execution time: {end2 - start2:.4f} seconds")

print(F1)

import jax.scipy.stats as jsps
import jax.scipy.special as jspsp


@jax.jit
def ellipse_area(cov2d, c=5.99):
    
    #Area of 2D Gaussian confidence ellipse.

    #Parameters
    #----------
    #cov2d : (2,2)
    #    Covariance matrix

    #c : float
    #    Chi^2 contour value
    #    68% -> 2.30
    #    95% -> 5.99
    

    # eigenvalues
    vals = jnp.linalg.eigvalsh(cov2d)

    # numerical stability
    vals = jnp.maximum(vals, 1e-12)

    # semi-axis lengths
    width  = jnp.sqrt(vals[1] * c)
    height = jnp.sqrt(vals[0] * c)

    # ellipse area
    area = jnp.pi * width * height

    return area


@jax.jit
def fom_from_fisher(F):

    C = jnp.linalg.inv(F)

    area = ellipse_area(C)

    return 1.0 / area

print(fom_from_fisher(F))
print(fom_from_fisher(F1))



"""











@jax.jit
def theory_mb_jc(params, z, H0):

    #
    # parameter ordering:
    #
    # params[0] = Om0
    # params[1] = w0
    # params[2] = wa
    # params[3] = Mb
    #

    Om0 = params[0]
    w0 = params[1]
    wa = params[2]
    Mb = params[3]

    cosmo = jc.Cosmology(

        Omega_c=Om0 - 0.05,
        Omega_b=0.05,

        h=H0 / 100.0,

        sigma8=0.8,
        n_s=0.96,

        Omega_k=0.0,

        w0=w0,
        wa=wa,
    )

    # --------------------------------------------------------
    # scale factor
    # --------------------------------------------------------

    a = 1.0 / (1.0 + z)

    # --------------------------------------------------------
    # comoving distance
    # --------------------------------------------------------

    chi = jc.background.radial_comoving_distance(
        cosmo,
        a,
        steps=10000,
    )

    #
    # jax-cosmo returns Mpc/h
    #

    chi = chi / cosmo.h

    # --------------------------------------------------------
    # luminosity distance
    # --------------------------------------------------------

    dL = chi / a

    # --------------------------------------------------------
    # distance modulus
    # --------------------------------------------------------

    mu = 5.0 * jnp.log10(dL) + 25.0

    return mu + Mb

@jax.jit
def chi2(params,z,data_obs,cov,H0):
    mb_theory = theory_mb_jc(params,z,H0)
    mb_panth = data_obs
    cov_inv = jnp.linalg.inv(cov)      #take this oout if possible
    r = mb_panth - mb_theory
    chi2 = r @ cov_inv @ r    
    return chi2



def loglike_jc(params,z,data_obs,cov,H0):
    yess = chi2(params,z,data_obs,cov,H0)
    return -0.5*yess



def fisher_matrix(
    chi2_fn,
    param_values,
    H0,
    z,
    data_obs,
    cov,
    h=None,
):

    def loglike_local(params):

        return -0.5 * chi2_fn(
            params,
            z,
            data_obs,
            cov,
            H0,
        )

    H = jax.hessian(loglike_local)(param_values)

    return -H


def fisher_matrix_fd(
    chi2_fn,
    param_values,
    H0,
    z,
    data_obs,
    cov,
    h=None,
):

    def loglike_local(params):

        return -0.5 * chi2_fn(
            params,
            z,
            data_obs,
            cov,
            H0,
        )

    H = jax.hessian(loglike_local)(param_values)

    return -H

fisher_matrix_fd = jax.jit(
    fisher_matrix_fd,
    static_argnums=(0,),
)


def marginalize_fisher_matrix(
    fisher_matrix,
    param_tuple,
    param_names,
):
    """
    JAX-compatible Fisher matrix marginalization.

    Parameters
    ----------
    fisher_matrix : jnp.ndarray
        Full Fisher matrix of shape (N, N).

    param_tuple : tuple[str]
        Parameters to keep.
        Example:
            ('Om0', 'w0')

    param_names : list[str]
        Ordered parameter names corresponding
        to Fisher matrix ordering.

    Returns
    -------
    cov_submatrix : jnp.ndarray
        Marginalized covariance submatrix.
    """

    # ------------------------------------------------------------
    # invert Fisher matrix
    # ------------------------------------------------------------

    fisher_cov = jnp.linalg.inv(fisher_matrix)

    # ------------------------------------------------------------
    # parameter lookup
    # ------------------------------------------------------------

    param_index = {
        name: i
        for i, name in enumerate(param_names)
    }

    selected_indices = jnp.array(
        [param_index[p] for p in param_tuple]
    )

    # ------------------------------------------------------------
    # extract marginalized covariance block
    # ------------------------------------------------------------

    cov_submatrix = fisher_cov[
        selected_indices[:, None],
        selected_indices[None, :]
    ]

    return cov_submatrix



def is_positive_definite(C):
    """
    JAX-compatible PSD check.
    """
    eigvals = jnp.linalg.eigvalsh(C)
    return jnp.all(eigvals > 0)



def ellipse_parameters(
    C,
    alpha,
    epsilon=1e-12,
):
    """
    JAX-compatible ellipse parameter computation.

    Parameters
    ----------
    C : jnp.ndarray
        2x2 covariance matrix.

    alpha : float
        Confidence level exclusion probability.
        Example:
            alpha = 0.32  -> 68% contour
            alpha = 0.05  -> 95% contour

    epsilon : float
        Eigenvalue floor for numerical stability.

    Returns
    -------
    width : float
        Major semi-axis.

    height : float
        Minor semi-axis.

    theta : float
        Rotation angle in degrees.

    area : float
        Ellipse area.
    """

    # ------------------------------------------------------------
    # symmetrize for numerical stability
    # ------------------------------------------------------------

    C = 0.5 * (C + C.T)

    # ------------------------------------------------------------
    # eigendecomposition
    # ------------------------------------------------------------

    eigenvalues, eigenvectors = jnp.linalg.eigh(C)

    # ------------------------------------------------------------
    # sort descending
    # ------------------------------------------------------------

    order = jnp.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # ------------------------------------------------------------
    # numerical stabilization
    # ------------------------------------------------------------

    eigenvalues = jnp.maximum(eigenvalues, epsilon)

    lambda1 = eigenvalues[0]
    lambda2 = eigenvalues[1]

    # ------------------------------------------------------------
    # chi2 quantile
    # ------------------------------------------------------------

    #
    # scipy equivalent:
    #
    # chi2.ppf(1-alpha, df=2)
    #
    # implemented via inverse incomplete gamma
    #

    chi2_quantile = -2.0 * jnp.log(alpha)

    # ------------------------------------------------------------
    # ellipse geometry
    # ------------------------------------------------------------

    width = jnp.sqrt(chi2_quantile * lambda1)

    height = jnp.sqrt(chi2_quantile * lambda2)

    theta = jnp.degrees(
        jnp.arctan2(
            eigenvectors[1, 0],
            eigenvectors[0, 0],
        )
    )

    area = jnp.pi * width * height

    return jnp.array([
    width,
    height,
    theta,
    area,
    ])

@jax.jit
def derivative_vector(
    params,
    idx,
    h,
    z,
    H0,
):
    """
    Central finite-difference derivative vector.

    Computes:

        dmu/dtheta_i

    for parameter index idx.
    """

    # --------------------------------------------------------
    # parameter perturbation vector
    # --------------------------------------------------------

    shift = jnp.zeros_like(params)

    shift = shift.at[idx].set(h)

    # --------------------------------------------------------
    # forward / backward models
    # --------------------------------------------------------

    mu_plus = theory_mb_jc(
        params + shift,
        z,
        H0,
    )

    mu_minus = theory_mb_jc(
        params - shift,
        z,
        H0,
    )

    # --------------------------------------------------------
    # central finite difference
    # --------------------------------------------------------

    dmu = (
        mu_plus - mu_minus
    ) / (2.0 * h)

    return dmu


@jax.jit
def fisher_matrix_observable(
    params,
    Cinv,
    z,
    H0,
    h=None,
):
    """
    Fisher matrix from observable derivatives.

    Parameters
    ----------
    params : ndarray
        Fiducial parameter vector.

    Cinv : ndarray
        Inverse covariance matrix.

    z : ndarray
        Redshift array.

    H0 : float
        Hubble constant.

    h : ndarray or None
        Finite-difference step sizes.

    Returns
    -------
    F : ndarray
        Fisher matrix.
    """

    # --------------------------------------------------------
    # parameter vector
    # --------------------------------------------------------

    params = jnp.asarray(
        params,
        dtype=jnp.float64,
    )

    npar = params.shape[0]

    # --------------------------------------------------------
    # finite difference step sizes
    # --------------------------------------------------------

    if h is None:

        h = jnp.full(
            (npar,),
            1e-4,
            dtype=jnp.float64,
        )

    else:

        h = jnp.asarray(
            h,
            dtype=jnp.float64,
        )

    # --------------------------------------------------------
    # derivative vectors
    # --------------------------------------------------------

    def single_derivative(i):

        return derivative_vector(
            params,
            i,
            h[i],
            z,
            H0,
        )

    derivs = jax.vmap(single_derivative)(
        jnp.arange(npar)
    )

    # --------------------------------------------------------
    # Fisher matrix
    # --------------------------------------------------------

    #
    # F_ij = d_i^T Cinv d_j
    #

    F = derivs @ Cinv @ derivs.T

    # --------------------------------------------------------
    # explicit symmetrization
    # --------------------------------------------------------

    F = 0.5 * (F + F.T)

    return F