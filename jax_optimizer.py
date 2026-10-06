import numpy as np
import sys
import astropy.constants as cst
import astropy.units as u
from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax_cosmo as jc
jax.config.update("jax_enable_x64", True)
from jax import lax
from tqdm.auto import tqdm

#--------------------------------------------------------------------------------------------------------
#-------------------------------------- FISHER MATRIX ANALYSIS ------------------------------------------
#--------------------------------------------------------------------------------------------------------

@jax.jit
def theory_mb_jc(params, z, H0):
    """JAX computation of the apparent magnitude a redshift z. 
    The flat CPL cosmology is hard-coded to avoid the use of stR.
    
    Parameters:
        params (jnp.array): array of parameters Om0, w0, wa, Mb.
        z (jnp.array): array of redshifts.
        H0: Hubble constant in km/s/Mpc.
        
    Returns:
        mb (jnp.array): theoretical apparent magnitude.
    """

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

    # scale factor
    a = 1.0 / (1.0 + z)

    # comoving distance
    dc = jc.background.radial_comoving_distance(
        cosmo,
        a,
        steps=10000,
    )
    dc = dc / cosmo.h # jax-cosmo returns Mpc/h

    # luminosity distance
    dL = dc / a

    # distance modulus
    mu = 5.0 * jnp.log10(dL) + 25.0

    return mu + Mb # apparent magnitude


@jax.jit
def chi2(params, z, data_obs, cov_inv, H0):
    """
    Jax computation of the chi-square. The flat CPL cosmology is hard-coded.
    
    Parameters:
        params (jnp.array): parameters Om0, w0, wa, Mb.
        z (jnp.array): redshift array.
        data_obs (jnp.array): observed dataset, here: apparent magnitude of SNIa (mb)
        cov_inv (jnp.array): inverse covariance matrix.
        H0: Hubble constant in km/s/Mpc.


    Returns:
        chi2 (float): chi square
    """

    # theory
    mb_theory = theory_mb_jc(
        params,
        z,
        H0,
    )

    # residuals 
    r = data_obs - mb_theory

    # chi2
    chi2 = r @ cov_inv @ r

    return chi2


@jax.jit
def fisher_matrix_hessian(
    chi2_fn,
    params,
    z,
    data_obs,
    cov,
    cosmology,
    H0
):
    @jax.jit
    def loglike_local(params):
        return -0.5 * chi2_fn(
            params,
            z,
            data_obs,
            cov,
            cosmology,
            H0
        )

    hessian_loglike = jax.jit(
        jax.hessian(loglike_local)
    )

    F = -jnp.asarray(
        hessian_loglike(
            jnp.asarray(params)
        )
    )

    return F


@jax.jit
def fisher_matrix_observable(
    params,
    cov_inv,
    z,
    H0,
):
    """
    Fisher matrix computed using JAX automatic differentiation.

    Parameters:
        params (jnp.array): fiducial parameter vector.
        cov_inv (jnp.array): inverse covariance matrix.
        z (jnp.array): redshift array.
        H0: Hubble constant in km/s/Mpc.

    Returns:
    F (jnp.array): Fisher matrix.
    """

    # parameter vector
    params = jnp.asarray(
        params,
        dtype=jnp.float64,
    )

    # Jacobian
    # J[a, i] = dmu_a / dtheta_i
    # shape = (ndata, npar)
    J = jax.jacrev(
        lambda p: theory_mb_jc(
            p,
            z,
            H0,
        )
    )(params)

    # Fisher matrix
    # F_ij = (dmu/dtheta_i)^T
    #        cov_inv
    #        (dmu/dtheta_j)

    F = J.T @ cov_inv @ J

    # explicit symmetrization
    F = 0.5 * (F + F.T)

    return F


def marginalize_fisher_matrix(
    fisher_matrix,
    param_tuple,
    param_names,
):
    """
    JAX-compatible Fisher matrix marginalization.

    Parameters:
        fisher_matrix (jnp.array): Full Fisher matrix of shape (N, N).
        param_tuple (tuple, str): Parameters to keep.
            Example: ('Om0', 'w0')
        param_names (list, str): Ordered parameter names corresponding to Fisher matrix ordering.

    Returns:
        cov_submatrix (jnp.array): Marginalized covariance submatrix.
    """

    # invert Fisher matrix
    fisher_cov = jnp.linalg.inv(
        fisher_matrix
    )

    # parameter lookup
    param_index = {
        name: i
        for i, name in enumerate(param_names)
    }

    selected_indices = jnp.array(
        [
            param_index[p]
            for p in param_tuple
        ]
    )

    # extract marginalized covariance block
    cov_submatrix = fisher_cov[
        selected_indices[:, None],
        selected_indices[None, :]
    ]

    return cov_submatrix


def is_positive_definite(C):
    """
    JAX-compatible PSD check.

    Parameters:
        C (jnp.array): Covariance matrix of the parameters.

    Returns:
        (bool): True if C is positive semi-definite.
    """

    eigvals = jnp.linalg.eigvalsh(C)

    return jnp.all(
        eigvals > 0
    )


def ellipse_parameters(
    C,
    alpha,
    epsilon=1e-12,
):
    """
    JAX-compatible ellipse parameter computation.

    Parameters:
        C (jnp.array): 2x2 covariance matrix.
        alpha (float): Confidence level exclusion probability.
            Example:
                alpha = 0.32  -> 68% contour
                alpha = 0.05  -> 95% contour
        epsilon (float): Eigenvalue floor for numerical stability.

    Returns:
        width (float): Major semi-axis.
        height (float): Minor semi-axis.
        theta (float): Rotation angle in degrees.
        area (float): Ellipse area.
    """

    # symmetrize for numerical stability
    C = 0.5 * (C + C.T)

    # eigendecomposition
    eigenvalues, eigenvectors = jnp.linalg.eigh(C)

    # sort descending
    order = jnp.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # numerical stabilization
    eigenvalues = jnp.maximum(
        eigenvalues,
        epsilon
    )

    lambda1 = eigenvalues[0]
    lambda2 = eigenvalues[1]

    # chi2 quantile
    # scipy equivalent: chi2.ppf(1-alpha, df=2)
    # implemented via inverse incomplete gamma

    chi2_quantile = -2.0 * jnp.log(alpha)

    # ellipse geometry
    width = jnp.sqrt(
        chi2_quantile * lambda1
    )

    height = jnp.sqrt(
        chi2_quantile * lambda2
    )

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


#--------------------------------------------------------------------------------------------------------
#--------------------------------------------- OPTIMIZER ------------------------------------------------
#--------------------------------------------------------------------------------------------------------

@jax.jit
def nearest_psd(
    A,
    eps=1e-10
):
    """
    Changes a matrix to the nearest positive semi-definite alternative.
    
    Parameters:
        A (jnp.array): matrix.
        eps (float): eigenvalue clipping. Default 1e-10.
        
    Returns:
        (jnp.array): nearest psd matrix
    """

    A = 0.5 * (
        A + A.T
    )

    eigvals, eigvecs = jnp.linalg.eigh(
        A
    )

    eigvals = jnp.maximum(
        eigvals,
        eps
    )

    A_psd = (
        eigvecs
        @ jnp.diag(eigvals)
        @ eigvecs.T
    )

    return 0.5 * (
        A_psd + A_psd.T
    )


@jax.jit
def build_covariance(
    dist,
    dist_base,
    cov_stat_base,
    cov_sys,
    eps_psd=1e-10,
):
    """
    Build the covariance matrix of an altered distribution by rescaling the statistical 
    part of the original covariance matrix.
    
    Parameters:
        dist (jnp.array): altered distribution.
        dist_base (jnp.array): original distribution.
        cov_stat_base (jnp.array): statistical covariance matrix of the original distribution.
        cov_sys (jnp.array): systematic part of the original covariance matrix (remains unchanged).
        eps_psd (float): eigenvalue clipping. Default 1e-10.
        
    Returns:
        cov (jnp.array): altered covariance matrix.
    """
    
    # ------------------------------------------------------------
    # Active bins
    # A bin with zero SNe contains no SN information.
    # ------------------------------------------------------------

    active = dist > 0.0

    active_float = active.astype(
        jnp.float64
    )

    # This is just to avoid division by zero for inactive bins.
    # The value of scale will not matter anyway since the
    # corresponding rows/columns will be zeroed out.
    dist_safe = jnp.maximum(
        dist,
        1.0
    )

    scale = jnp.sqrt(
        dist_base / dist_safe
    )

    # Statistical covariance for active bins only.
    cov_stat = (
        scale[:, None]
        * cov_stat_base
        * scale[None, :]
    )

    # Remove inactive rows/columns.
    cov_stat = (
        active_float[:, None]
        * cov_stat
        * active_float[None, :]
    )

    # Remove systematic covariance involving inactive bins.
    cov_sys_active = (
        active_float[:, None]
        * cov_sys
        * active_float[None, :]
    )

    cov = (
        cov_stat
        + cov_sys_active
    )

    # Add a finite diagonal placeholder (1) for zero-population bins;
    # their Jacobian rows are masked later, so they contribute no Fisher information.
    cov = (
        cov
        + jnp.diag(
            1.0 - active_float
        )
    )

    return cov


# ============================================================
# ONE OPTIMIZATION ITERATION
# ============================================================

def make_optimizer_step(
    nbins,
    ss,
    dist_base,
    Cov_stat_base,
    Cov_sys,
    II,
    lsst_SNIa,
    z_roman,
    tt,
    H0,
    fiducial_cosmo,
    NIa_tot,
    param_tuple,
    param_names,
    perturbation,
    use_marginalization,
):
    """
    Construct the JIT-compiled function corresponding to ONE
    iteration of the original NumPy optimizer.

    The outer iteration loop remains Python-controlled so that
    tqdm can report actual progress.
    """

    # ========================================================
    # COSMOLOGICAL JACOBIAN
    # ========================================================

    J = jax.jacrev(
        lambda p: theory_mb_jc(
            p,
            z_roman,
            H0,
        )
    )(fiducial_cosmo)

    # ========================================================
    # FISHER
    # ========================================================

    def fisher_from_cinv(
        Cinv,
        dist
    ):
        active = (
            dist > 0.0
        )

        active_float = active.astype(
            jnp.float64
        )

        J_active = (
            active_float[:, None]
            * J
        )

        F = (
            J_active.T
            @ Cinv
            @ J_active
        )

        F = 0.5 * (
            F + F.T
        )

        return F

    # ========================================================
    # FOM
    # ========================================================

    def compute_fom(F):

        if use_marginalization:

            C = marginalize_fisher_matrix(
                F,
                param_tuple,
                param_names
            )

        else:

            C = jnp.linalg.solve(
                F,
                jnp.eye(
                    F.shape[0]
                )
            )

        ellipse = ellipse_parameters(
            C,
            0.32
        )

        fom = 1.0 / jnp.sqrt(
            jnp.linalg.det(C)
        )

        return fom, ellipse

    # ========================================================
    # SINGLE-BIN dFOM
    # ========================================================

    def compute_single_bin_gradient(
        i,
        dist_reference,
        FOM_reference
    ):
        """
        Direct JAX equivalent of the NumPy loop
        "for i in range(len(z_roman)):"
        including the distinction between dFOM and
        dFOM_triche.
        """

        # Reference population without LSST
        dist_reference_without_lsst = (
            dist_reference
            - lsst_SNIa
        )

        valid = (
            dist_reference_without_lsst[i]
            >= perturbation
        )

        # Common perturbed distribution
        # (add one SNIa in bin i)
        dist_perturbed = (
            dist_reference
            .at[i]
            .add(perturbation)
        )

        # ----------------------------------------------------
        # Calculate perturbed FOM
        # ----------------------------------------------------

        Cov_perturbed = build_covariance(
            dist_perturbed,
            dist_base,
            Cov_stat_base,
            Cov_sys,
        )

        Cov_perturbed = 0.5 * (
            Cov_perturbed
            + Cov_perturbed.T
        )

        Cinv_perturbed = jnp.linalg.solve(
            Cov_perturbed,
            II
        )

        FF_perturbed = fisher_from_cinv(
            Cinv_perturbed,
            dist_perturbed
        )

        FOM_perturbed, _ = compute_fom(
            FF_perturbed
        )

        dFOM_i = (
            FOM_perturbed
            - FOM_reference
        ) / tt[i]

        # ----------------------------------------------------
        # dFOM_triche
        #
        # Only bins satisfying the population constraint
        # are eligible for removal.
        # ----------------------------------------------------

        dFOM_triche_i = jnp.where(
            valid,
            dFOM_i,
            jnp.nan,
        )

        return (
            dFOM_i,
            dFOM_triche_i
        )

    # ========================================================
    # VMAP OVER BINS
    # ========================================================

    vmapped_gradient = jax.vmap(
        compute_single_bin_gradient,
        in_axes=(0, None, None),
    )

    # ========================================================
    # REDISTRIBUTION
    # ========================================================

    def find_valid_redistribution(
        dist_reference,
        dFOM,
        kk,
    ):
        """
        JAX equivalent of:
        "while valid == False:"
        from the NumPy implementation.

        Negative dFOM bins are allowed to be selected for
        removal, but cannot receive redistributed SNe.
        """

        # ----------------------------------------------------
        # Initial ignore
        #
        # Negative dFOM bins cannot receive redistributed SNe.
        # kk is also excluded from receiving its own SNe back.
        # ----------------------------------------------------

        negative_mask = (
            dFOM < 0.0
        )

        ignore_initial = (
            negative_mask
            .at[kk]
            .set(True)
        )

        # Initial state
        initial_delta_n = jnp.zeros(
            nbins,
            dtype=jnp.float64
        )

        initial_state = (
            dist_reference,
            initial_delta_n,
            ignore_initial,
            jnp.array(False),  # valid
            jnp.array(False),  # stalled
            jnp.array(
                0,
                dtype=jnp.int32
            ),
        )

        # ----------------------------------------------------
        # Maximum number of attempts
        # ----------------------------------------------------

        max_steps = nbins + 1

        # ----------------------------------------------------
        # CONDITION
        # ----------------------------------------------------

        def condition(state):

            (
                dist_new,
                delta_n,
                ignore,
                valid,
                stalled,
                step,
            ) = state

            return (
                ~valid
                & ~stalled
                & (step < max_steps)
            )

        # ----------------------------------------------------
        # BODY
        # ----------------------------------------------------

        def body(state):

            (
                _dist_new,
                _delta_n,
                ignore,
                _valid,
                _stalled,
                step,
            ) = state

            # =================================================
            # Distribution without LSST
            # =================================================

            dist_new_test_without_lsst = (
                dist_reference
                - lsst_SNIa
            )

            # =================================================
            # Safety check on kk
            # =================================================

            kk_population = (
                dist_new_test_without_lsst[kk]
            )

            kk_safe = (
                kk_population
                >= perturbation
            )

            # =================================================
            # Remove perturbation from kk
            # =================================================

            dist_after_removal = (
                dist_new_test_without_lsst
                .at[kk]
                .add(-perturbation)
            )

            # =================================================
            # Time saved
            # =================================================

            time_saved = (
                tt[kk]
                * perturbation
            )

            # =================================================
            # Reallocation mask
            #
            # Only positive dFOM bins can receive SNe.
            # Negative dFOM bins remain excluded.
            # =================================================

            reallocate_mask = (
                (~ignore)
                & (dFOM > 0.0)
            )

            # =================================================
            # Denominator
            # =================================================

            denom = jnp.sum(
                jnp.where(
                    reallocate_mask,
                    dFOM,
                    0.0,
                )
            )

            denominator_valid = (
                jnp.isfinite(denom)
                & (
                    jnp.abs(denom)
                    > 1e-14
                )
            )

            safe_denom = jnp.where(
                denominator_valid,
                denom,
                1.0,
            )

            # =================================================
            # delta_n
            # =================================================

            delta_n_test = (
                perturbation
                * tt[kk]
                / tt
                * dFOM
                / safe_denom
            )

            delta_n_test = jnp.where(
                reallocate_mask,
                delta_n_test,
                0.0,
            )

            # =================================================
            # New distribution without LSST
            # =================================================

            dist_after_redistribution = (
                dist_after_removal
                + delta_n_test
            )

            # =================================================
            # NIa_tot constraint
            # =================================================

            comparison = (
                (dist_after_redistribution >= 0.0)
                & (
                    dist_after_redistribution
                    <= NIa_tot
                )
            )

            all_valid = jnp.all(
                comparison
            )

            problematic_bins = (
                dist_after_redistribution
                > NIa_tot
            )

            # =================================================
            # Update ignore
            # =================================================

            new_ignore = (
                ignore
                | problematic_bins
            )

            # =================================================
            # Did ignore actually change?
            # =================================================

            mask_changed = jnp.any(
                new_ignore != ignore
            )

            # =================================================
            # Successful redistribution
            # =================================================

            new_valid = (
                kk_safe
                & denominator_valid
                & all_valid
            )

            # =================================================
            # Stalling
            # =================================================

            all_ignored = jnp.all(
                new_ignore
            )

            no_progress = (
                ~new_valid
                & ~mask_changed
            )

            new_stalled = (
                ~new_valid
                & (
                    all_ignored
                    | no_progress
                )
            )

            # =================================================
            # Add LSST back
            # =================================================

            dist_candidate = (
                dist_after_redistribution
                + lsst_SNIa
            )

            # =================================================
            # Store candidate only if valid
            # =================================================

            output_dist = jnp.where(
                new_valid,
                dist_candidate,
                _dist_new,
            )

            output_delta = jnp.where(
                new_valid,
                delta_n_test,
                _delta_n,
            )

            return (
                output_dist,
                output_delta,
                new_ignore,
                new_valid,
                new_stalled,
                step + 1,
            )

        # ----------------------------------------------------
        # Execute redistribution loop
        # ----------------------------------------------------

        (
            dist_new,
            delta_n,
            ignore,
            valid,
            stalled,
            step,
        ) = lax.while_loop(
            condition,
            body,
            initial_state,
        )

        # ----------------------------------------------------
        # Hard timeout
        # ----------------------------------------------------

        stalled = (
            stalled
            | (
                (~valid)
                & (step >= max_steps)
            )
        )

        return (
            dist_new,
            delta_n,
            ignore,
            valid,
            stalled,
        )

    # ========================================================
    # ONE OPTIMIZATION STEP
    # ========================================================

    def optimizer_step(
        dist_reference,
        FOM_reference,
        rng_key,
    ):

        # ====================================================
        # Calculate dFOM for every bin
        # ====================================================

        dFOM, dFOM_triche = (
            vmapped_gradient(
                jnp.arange(nbins),
                dist_reference,
                FOM_reference,
            )
        )

        # ====================================================
        # Select kk probabilistically
        #
        # A bin is eligible for REMOVAL if:
        #
        #   1. dFOM_triche is finite
        #
        # Negative dFOM is allowed.
        #
        # Probability:
        #
        #   P_i ∝ exp(-beta * dFOM_i)
        #
        # Therefore:
        #   smaller dFOM -> larger probability
        #   negative dFOM -> particularly favored
        # ====================================================

        eligible = jnp.isfinite(
            dFOM_triche
        )

        has_eligible = jnp.any(
            eligible
        )

        beta = 1.0

        # ----------------------------------------------------
        # Log weights
        # ----------------------------------------------------

        log_weights = (
            -beta * dFOM
        )

        # Only eligible bins contribute to the maximum.
        max_log_weight = jnp.max(
            jnp.where(
                eligible,
                log_weights,
                -jnp.inf,
            )
        )

        # ----------------------------------------------------
        # Exponentiate after subtracting the maximum.
        #
        # This prevents overflow/underflow for large dFOM.
        # ----------------------------------------------------

        weights = jnp.where(
            eligible,
            jnp.exp(
                log_weights
                - max_log_weight
            ),
            0.0,
        )

        weight_sum = jnp.sum(
            weights
        )

        probabilities = jnp.where(
            weight_sum > 0.0,
            weights / weight_sum,
            jnp.zeros_like(weights),
        )

        # ----------------------------------------------------
        # Randomly select kk
        # ----------------------------------------------------

        rng_key, subkey = jax.random.split(
            rng_key
        )

        kk = jax.random.choice(
            subkey,
            nbins,
            shape=(),
            p=probabilities,
        )

        # ====================================================
        # Redistribution
        # ====================================================

        def redistribution_branch(_):

            return find_valid_redistribution(
                dist_reference,
                dFOM,
                kk,
            )

        def stalled_branch(_):

            return (
                dist_reference,
                jnp.zeros(
                    nbins,
                    dtype=jnp.float64,
                ),
                jnp.ones(
                    nbins,
                    dtype=jnp.bool_,
                ),
                jnp.array(False),
                jnp.array(True),
            )

        (
            dist_new,
            delta_n,
            ignore,
            redistribution_valid,
            stalled,
        ) = lax.cond(
            has_eligible,
            redistribution_branch,
            stalled_branch,
            operand=None,
        )

        # ====================================================
        # If redistribution stalled, preserve state
        # ====================================================

        dist_new = jnp.where(
            stalled,
            dist_reference,
            dist_new,
        )

        # ====================================================
        # New covariance
        # ====================================================

        Cov_new = build_covariance(
            dist_new,
            dist_base,
            Cov_stat_base,
            Cov_sys,
        )

        Cov_new = 0.5 * (
            Cov_new
            + Cov_new.T
        )

        # ====================================================
        # Fisher
        # ====================================================

        Cinv_new = jnp.linalg.solve(
            Cov_new,
            II,
        )

        FF_new = fisher_from_cinv(
            Cinv_new,
            dist_new
        )

        # ====================================================
        # FOM
        # ====================================================

        FOM_new, ellipse_new = (
            compute_fom(
                FF_new
            )
        )

        # ====================================================
        # Tracking
        # ====================================================

        tracking = {
            "distribution": dist_new,
            "delta_n": delta_n,
            "Cov": Cov_new,
            "dFOM": dFOM,
            "dFOM_triche": dFOM_triche,
            "kk": kk,
            "FF": FF_new,
            "ellipse": ellipse_new,
            "FOM": FOM_new,
            "stalled": stalled,
            "redistribution_valid":
                redistribution_valid,
            "ignore": ignore,
        }

        return (
            dist_new,
            FOM_new,
            tracking,
        )

    # ========================================================
    # JIT ONE ITERATION
    # ========================================================

    return jax.jit(
        optimizer_step
    )


# ============================================================
# MAIN OPTIMIZER
# ============================================================

def optimize_bins_again_jax(
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
    param_tuple,
    param_names,
    NIa_tot,
    use_marginalization=True,
    verbose=True,
):
    """
    JAX implementation of optimize_bins_again.

    The numerical work of each optimization iteration is JIT
    compiled. The outer iteration loop is Python-controlled,
    allowing tqdm to display genuine iteration progress.
    """

    # ========================================================
    # CONVERT INPUTS
    # ========================================================

    dist_roman = jnp.asarray(
        dist_roman,
        dtype=jnp.float64,
    )

    Cov_roman = jnp.asarray(
        Cov_roman,
        dtype=jnp.float64,
    )

    Cov_roman_stat = jnp.asarray(
        Cov_roman_stat,
        dtype=jnp.float64,
    )

    Cov_roman_sys = jnp.asarray(
        Cov_roman_sys,
        dtype=jnp.float64,
    )

    FOM = jnp.asarray(
        FOM,
        dtype=jnp.float64,
    )

    H0 = jnp.asarray(
        H0,
        dtype=jnp.float64,
    )

    z_roman = jnp.asarray(
        z_roman,
        dtype=jnp.float64,
    )

    tt = jnp.asarray(
        tt,
        dtype=jnp.float64,
    )

    fiducial_cosmo = jnp.asarray(
        fiducial_cosmo,
        dtype=jnp.float64,
    )

    NIa_tot = jnp.asarray(
        NIa_tot,
        dtype=jnp.float64,
    )

    # ========================================================
    # DIMENSIONS
    # ========================================================

    nbins = z_roman.shape[0]

    ss = fiducial_cosmo.shape[0]

    # ========================================================
    # LSST CONTRIBUTION
    # ========================================================

    lsst_SNIa = jnp.zeros(
        nbins,
        dtype=jnp.float64,
    )

    lsst_SNIa = lsst_SNIa.at[0].set(
        801.0
    )

    # ========================================================
    # FIXED BASELINE
    # ========================================================

    dist_base = dist_roman

    # Cov_stat_base = nearest_psd(
    #     Cov_roman_stat
    # )

    Cov_stat_base = Cov_roman_stat

    # Cov_sys = nearest_psd(
    #     Cov_roman_sys
    # )

    Cov_sys = Cov_roman_sys

    II = jnp.eye(
        Cov_roman.shape[0],
        dtype=jnp.float64,
    )

    # ========================================================
    # BUILD JIT ITERATION FUNCTION
    # ========================================================

    optimizer_step_jax = make_optimizer_step(
        nbins=nbins,
        ss=ss,
        dist_base=dist_base,
        Cov_stat_base=Cov_stat_base,
        Cov_sys=Cov_sys,
        II=II,
        lsst_SNIa=lsst_SNIa,
        z_roman=z_roman,
        tt=tt,
        H0=H0,
        fiducial_cosmo=fiducial_cosmo,
        NIa_tot=NIa_tot,
        param_tuple=param_tuple,
        param_names=param_names,
        perturbation=perturbation,
        use_marginalization=use_marginalization,
    )

    # ========================================================
    # TRACKERS
    # ========================================================

    track_FOM = []
    track_FF = []
    track_distribution = []
    track_delta_n = []

    # track_Cov = []

    track_dFOM = []

    # track_dFOM_triche = []

    track_kk = []

    # track_stalled = []
    # track_redistribution_valid = []
    # track_ignore = []

    # ========================================================
    # CURRENT STATE
    # ========================================================

    dist_reference = dist_roman

    FOM_reference = FOM

    # ========================================================
    # RANDOM NUMBER GENERATOR
    # ========================================================

    rng_key = jax.random.PRNGKey(
        42
    )

    # ========================================================
    # PROGRESS BAR
    # ========================================================

    with tqdm(
        range(nb_iter),
        desc="Optimizing bins",
        unit="iter",
    ) as pbar:

        for nn in pbar:

            # =================================================
            # RANDOM KEY FOR THIS ITERATION
            # =================================================

            rng_key, subkey = jax.random.split(
                rng_key
            )

            # =================================================
            # ONE JIT-COMPILED ITERATION
            # =================================================

            (
                dist_reference,
                FOM_reference,
                tracking,
            ) = optimizer_step_jax(
                dist_reference,
                FOM_reference,
                subkey,
            )

            # =================================================
            # FORCE SYNCHRONIZATION
            # =================================================

            jax.block_until_ready(
                dist_reference
            )

            # =================================================
            # STORE TRACKING
            # =================================================

            save_every = 100

            if nn % save_every == 0:

                track_distribution.append(
                    tracking["distribution"]
                )

                track_delta_n.append(
                    tracking["delta_n"]
                )

                # track_Cov.append(
                #     tracking["Cov"]
                # )

                track_dFOM.append(
                    tracking["dFOM"]
                )

                # track_dFOM_triche.append(
                #     tracking["dFOM_triche"]
                # )

                track_kk.append(
                    tracking["kk"]
                )

                track_FF.append(
                    tracking["FF"]
                )

                track_FOM.append(
                    tracking["FOM"]
                )

                # track_stalled.append(
                #     tracking["stalled"]
                # )

                # track_redistribution_valid.append(
                #     tracking["redistribution_valid"]
                # )

                # track_ignore.append(
                #     tracking["ignore"]
                # )

            # =================================================
            # PROGRESS BAR INFO
            # =================================================

            if verbose:

                kk_python = int(
                    tracking["kk"]
                )

                fom_python = float(
                    FOM_reference
                )

                pbar.set_postfix(
                    kk=kk_python,
                    FOM=f"{fom_python:.6g}",
                )

    # ========================================================
    # STACK TRACKING
    # ========================================================

    track_distribution = jnp.stack(
        track_distribution,
        axis=0,
    )

    track_delta_n = jnp.stack(
        track_delta_n,
        axis=0,
    )

    # track_Cov = jnp.stack(
    #     track_Cov,
    #     axis=0,
    # )

    track_dFOM = jnp.stack(
        track_dFOM,
        axis=0,
    )

    # track_dFOM_triche = jnp.stack(
    #     track_dFOM_triche,
    #     axis=0,
    # )

    track_kk = jnp.stack(
        track_kk,
        axis=0,
    )

    track_FF = jnp.stack(
        track_FF,
        axis=0,
    )

    track_FOM = jnp.stack(
        track_FOM,
        axis=0,
    )

    # track_stalled = jnp.stack(
    #     track_stalled,
    #     axis=0,
    # )

    # track_redistribution_valid = jnp.stack(
    #     track_redistribution_valid,
    #     axis=0,
    # )

    # track_ignore = jnp.stack(
    #     track_ignore,
    #     axis=0,
    # )

    # ========================================================
    # FINAL DISTRIBUTION
    # ========================================================

    optimized_dist = (
        jnp.round(
            dist_reference
        )
        .astype(
            jnp.int64
        )
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "optimized_dist":
            optimized_dist,

        "track_distribution":
            track_distribution,

        "track_delta_n":
            track_delta_n,

        # "track_Cov":
        #     track_Cov,

        "track_dFOM":
            track_dFOM,

        # "track_dFOM_triche":
        #     track_dFOM_triche,

        "track_kk":
            track_kk,

        "track_FF":
            track_FF,

        "track_FOM":
            track_FOM,

        # "track_stalled":
        #     track_stalled,

        # "track_redistribution_valid":
        #     track_redistribution_valid,

        # "track_ignore":
        #     track_ignore,
    }