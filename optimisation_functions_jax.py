import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np
import time

import fisher_matrix_analysis_jax as fma


# ============================================================
# PSD UTILITIES
# ============================================================

@jax.jit
def nearest_psd(A, eps=1e-10):

    A = 0.5 * (A + A.T)

    eigvals, eigvecs = jnp.linalg.eigh(A)

    eigvals = jnp.clip(eigvals, a_min=eps)

    A_psd = eigvecs @ jnp.diag(eigvals) @ eigvecs.T

    return 0.5 * (A_psd + A_psd.T)


@jax.jit
def build_covariance(
    dist,
    dist_base,
    Cov_stat_base,
    Cov_sys,
    eps_psd=1e-10,
):

    scale = jnp.sqrt(dist_base / dist)

    Cov_stat = (
        scale[:, None]
        * Cov_stat_base
        * scale[None, :]
    )

    Cov = Cov_stat + Cov_sys

    Cov = nearest_psd(Cov, eps_psd)

    return Cov


# ============================================================
# MAIN OPTIMIZER
# ============================================================

@partial(
    jax.jit,
    static_argnames=(
        "nb_iter",
        "param_tuple",
        "param_names",
        "use_marginalization",
        "verbose",
    ),
)
def optimize_bins_gen_jax(
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
    use_marginalization=True,
    verbose=False,
):

    ss = fiducial_cosmo.shape[0]

    nbins = z_roman.shape[0]

    # ========================================================
    # FIXED BASELINE OBJECTS
    # ========================================================

    dist_base = dist_roman.astype(jnp.float64)

    Cov_stat_base = nearest_psd(
        Cov_roman_stat.astype(jnp.float64)
    )

    Cov_sys = nearest_psd(
        Cov_roman_sys.astype(jnp.float64)
    )

    II = jnp.eye(nbins)

    # ========================================================
    # SINGLE BIN GRADIENT
    # ========================================================

    def compute_single_bin_gradient(
        i,
        dist_reference,
        FOM_reference,
    ):

        min_bin_population = 2.0 * perturbation

        current_bin = dist_reference[i]

        valid_main = (
            current_bin > (min_bin_population + 1.0)
        )

        valid_edge = (
            (current_bin >= min_bin_population)
            & (current_bin <= min_bin_population + 1.0)
        )

        valid = valid_main | valid_edge

        def valid_branch(_):

            time_freed = tt[i] * perturbation

            # ------------------------------------------------
            # remove systems
            # ------------------------------------------------

            dist_perturbed = (
                dist_reference
                .at[i]
                .add(-perturbation)
            )

            # ------------------------------------------------
            # redistribute uniformly
            # ------------------------------------------------

            weights = jnp.ones(nbins)

            weights = weights.at[i].set(0.0)

            weights = weights / weights.sum()

            delta = (
                time_freed
                * weights
            ) / tt

            dist_perturbed = (
                dist_perturbed + delta
            )

            dist_perturbed = jnp.maximum(
                dist_perturbed,
                1e-12,
            )

            # ------------------------------------------------
            # covariance
            # ------------------------------------------------

            Cov_perturbed = build_covariance(
                dist_perturbed,
                dist_base,
                Cov_stat_base,
                Cov_sys,
            )

            Cov_perturbed = (
                0.5
                * (Cov_perturbed + Cov_perturbed.T)
            )

            Cinv_perturbed = jnp.linalg.solve(
                Cov_perturbed,
                II,
            )

            # ------------------------------------------------
            # Fisher matrix
            # ------------------------------------------------

            FF_perturbed = (
                fma.fisher_matrix_observable(
                    fiducial_cosmo,
                    Cinv_perturbed,
                    z_roman,
                    H0,
                )
            )

            FF_perturbed = nearest_psd(
                FF_perturbed
            )

            # ------------------------------------------------
            # parameter covariance
            # ------------------------------------------------

            if use_marginalization:

                CC_perturbed = (
                    fma.marginalize_fisher_matrix(
                        FF_perturbed,
                        param_tuple,
                        param_names,
                    )
                )

            else:

                CC_perturbed = jnp.linalg.inv(
                    FF_perturbed
                )

            CC_perturbed = nearest_psd(
                CC_perturbed
            )

            ellipse = fma.ellipse_parameters(
                CC_perturbed,
                0.32,
            )

            FOM_perturbed = (
                1.0 / ellipse[3]
            )

            dFOM = -(
                FOM_perturbed
                - FOM_reference
            ) / tt[i]

            dFOM_triche = lax.cond(
                valid_main,
                lambda _: dFOM,
                lambda _: jnp.nan,
                operand=None,
            )

            return (
                dFOM,
                dFOM_triche,
                FF_perturbed,
                ellipse,
            )

        def invalid_branch(_):

            return (
                jnp.nan,
                jnp.nan,
                jnp.zeros((ss, ss)),
                jnp.zeros((4,)),
            )

        return lax.cond(
            valid,
            valid_branch,
            invalid_branch,
            operand=None,
        )

    # ========================================================
    # VMAP
    # ========================================================

    vmapped_gradient = jax.vmap(
        compute_single_bin_gradient,
        in_axes=(0, None, None),
    )

    # ========================================================
    # OPTIMIZER STEP
    # ========================================================

    def optimizer_step(carry, _):

        (
            dist_reference,
            Cov_reference,
            FOM_reference,
        ) = carry

        (
            dFOM,
            dFOM_triche,
            FF_all,
            ellipse_all,
        ) = vmapped_gradient(
            jnp.arange(nbins),
            dist_reference,
            FOM_reference,
        )

        # ----------------------------------------------------
        # choose kk
        # ----------------------------------------------------

        masked = jnp.where(
            jnp.isnan(dFOM_triche),
            jnp.inf,
            dFOM_triche,
        )

        kk = jnp.argmin(masked)

        # ----------------------------------------------------
        # ignore bins
        # ----------------------------------------------------

        negative_mask = dFOM < 0.0

        ignore_mask = (
            negative_mask
            .at[kk]
            .set(True)
        )

        # ----------------------------------------------------
        # redistribution
        # ----------------------------------------------------

        dist_new = (
            dist_reference
            .at[kk]
            .add(-perturbation)
        )

        denom = jnp.sum(
            jnp.where(
                ~ignore_mask,
                dFOM,
                0.0,
            )
        )

        safe_denom = jnp.where(
            jnp.abs(denom) < 1e-14,
            1.0,
            denom,
        )

        delta_n = jnp.where(
            ~ignore_mask,
            (
                perturbation
                * tt[kk]
                / tt
                * dFOM
                / safe_denom
            ),
            0.0,
        )

        dist_new = dist_new + delta_n

        dist_new = jnp.maximum(
            dist_new,
            1e-12,
        )

        # ----------------------------------------------------
        # covariance update
        # ----------------------------------------------------

        Cov_new = build_covariance(
            dist_new,
            dist_base,
            Cov_stat_base,
            Cov_sys,
        )

        Cinv_new = jnp.linalg.solve(
            Cov_new,
            II,
        )

        # ----------------------------------------------------
        # Fisher matrix
        # ----------------------------------------------------

        FF_new = (
            fma.fisher_matrix_observable(
                fiducial_cosmo,
                Cinv_new,
                z_roman,
                H0,
            )
        )

        FF_new = nearest_psd(
            FF_new
        )

        # ----------------------------------------------------
        # parameter covariance
        # ----------------------------------------------------

        if use_marginalization:

            CC_new = (
                fma.marginalize_fisher_matrix(
                    FF_new,
                    param_tuple,
                    param_names,
                )
            )

        else:

            CC_new = jnp.linalg.inv(
                FF_new
            )

        CC_new = nearest_psd(CC_new)

        ellipse_new = fma.ellipse_parameters(
            CC_new,
            0.32,
        )

        FOM_new = 1.0 / ellipse_new[3]

        # ----------------------------------------------------
        # tracking
        # ----------------------------------------------------

        tracking = {
            "distribution": dist_new,
            "delta_n": delta_n,
            "Cov": Cov_new,
            "dFOM": dFOM,
            "dFOM_triche": dFOM_triche,
            "kk": kk,
            "FF": FF_new,
            "ellipse": ellipse_new,
        }

        new_carry = (
            dist_new,
            Cov_new,
            FOM_new,
        )

        return new_carry, tracking

    # ========================================================
    # MAIN SCAN
    # ========================================================

    initial_carry = (
        dist_roman.astype(jnp.float64),
        Cov_roman.astype(jnp.float64),
        FOM,
    )

    final_state, history = lax.scan(
        optimizer_step,
        initial_carry,
        xs=None,
        length=nb_iter,
    )

    optimized_dist = jnp.round(
        final_state[0]
    ).astype(jnp.int64)

    return {
        "optimized_dist": optimized_dist,
        "track_distribution": history["distribution"],
        "track_delta_n": history["delta_n"],
        "track_Cov": history["Cov"],
        "track_dFOM": history["dFOM"],
        "track_dFOM_triche": history["dFOM_triche"],
        "track_kk": history["kk"],
        "track_FF": history["FF"],
        "track_ellipse": history["ellipse"],
    }




