# ============================================================
# ================= OPTIMIZATION — JAX =======================
# ============================================================

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax
from functools import partial
from tqdm.auto import tqdm

import fisher_matrix_analysis_jax as fma


# ============================================================
# PSD UTILITIES
# ============================================================

@jax.jit
def nearest_psd(A, eps=1e-10):

    A = 0.5 * (A + A.T)

    eigvals, eigvecs = jnp.linalg.eigh(A)

    eigvals = jnp.maximum(eigvals, eps)

    A_psd = (
        eigvecs
        @ jnp.diag(eigvals)
        @ eigvecs.T
    )

    return 0.5 * (A_psd + A_psd.T)


# ============================================================
# COVARIANCE
# ============================================================

@jax.jit
def build_covariance(
    dist,
    dist_base,
    Cov_stat_base,
    Cov_sys,
    eps_psd=1e-10,
):

    scale = jnp.sqrt(
        dist_base / dist
    )

    Cov_stat = (
        scale[:, None]
        * Cov_stat_base
        * scale[None, :]
    )

    Cov = Cov_stat + Cov_sys

    Cov = nearest_psd(
        Cov,
        eps_psd,
    )

    return Cov


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
    #
    # This is independent of the covariance/distribution, so
    # calculate it once.
    #
    # This replaces the repeated calls to
    # fma.fisher_matrix_observable() in the NumPy implementation.
    # ========================================================

    J = jax.jacrev(
        lambda p: fma.theory_mb_jc(
            p,
            z_roman,
            H0,
        )
    )(fiducial_cosmo)

    # ========================================================
    # FISHER
    # ========================================================

    def fisher_from_cinv(Cinv):

        F = (
            J.T
            @ Cinv
            @ J
        )

        F = 0.5 * (
            F + F.T
        )

        F = nearest_psd(F)

        return F

    # ========================================================
    # FOM
    # ========================================================

    def compute_fom(F):

        if use_marginalization:

            C = fma.marginalize_fisher_matrix(
                F,
                param_tuple,
                param_names,
            )

        else:

            C = jnp.linalg.inv(F)

        C = nearest_psd(C)

        ellipse = fma.ellipse_parameters(
            C,
            0.32,
        )

        fom = 1.0 / ellipse[3]

        return fom, C, ellipse

    # ========================================================
    # SINGLE-BIN dFOM
    # ========================================================

    def compute_single_bin_gradient(
        i,
        dist_reference,
        FOM_reference,
    ):
        """
        Direct JAX equivalent of the NumPy loop

            for i in range(len(z_roman)):

        including the distinction between dFOM and
        dFOM_triche.
        """

        # ----------------------------------------------------
        # Reference population without LSST
        # ----------------------------------------------------

        dist_reference_without_lsst = (
            dist_reference - lsst_SNIa
        )

        min_bin_population = perturbation

        valid = (
            dist_reference_without_lsst[i]
            > min_bin_population + 1.0
        )

        # ----------------------------------------------------
        # Common perturbed distribution
        # ----------------------------------------------------

        dist_perturbed = (
            dist_reference
            .at[i]
            .add(perturbation)
        )

        # ----------------------------------------------------
        # Calculate perturbed FOM
        #
        # IMPORTANT:
        #
        # The NumPy implementation calculates dFOM even for
        # bins which are subsequently excluded from kk.
        #
        # Therefore we retain that behavior.
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
            II,
        )

        FF_perturbed = fisher_from_cinv(
            Cinv_perturbed
        )

        FOM_perturbed, _, _ = compute_fom(
            FF_perturbed
        )

        dFOM_i = (
            FOM_perturbed
            - FOM_reference
        ) / tt[i]

        # ----------------------------------------------------
        # dFOM_triche
        #
        # NumPy:
        #
        # if population > perturbation + 1:
        #     dFOM_triche[i] = dFOM[i]
        # else:
        #     dFOM_triche[i] = np.nan
        # ----------------------------------------------------

        dFOM_triche_i = jnp.where(
            valid,
            dFOM_i,
            jnp.nan,
        )

        return (
            dFOM_i,
            dFOM_triche_i,
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

            while valid == False:

        from the NumPy implementation.

        `ignore` is represented as a boolean mask.
        """

        # ----------------------------------------------------
        # Initial ignore
        #
        # NumPy:
        #
        # gg = np.where(dFOM < 0)[0]
        # ignore = np.unique(np.sort(np.append(gg, kk)))
        # ----------------------------------------------------

        negative_mask = (
            dFOM < 0.0
        )

        ignore_initial = (
            negative_mask
            .at[kk]
            .set(True)
        )

        # ----------------------------------------------------
        # Initial state
        # ----------------------------------------------------

        initial_delta_n = jnp.zeros(
            nbins,
            dtype=jnp.float64,
        )

        initial_state = (
            dist_reference,
            initial_delta_n,
            ignore_initial,
            jnp.array(False),  # valid
            jnp.array(False),  # stalled
            jnp.array(0, dtype=jnp.int32),
        )

        # ----------------------------------------------------
        # Maximum number of attempts
        #
        # Every unsuccessful iteration must either:
        #
        # 1. add at least one problematic bin to ignore, or
        # 2. stall.
        #
        # Therefore nbins + 1 is sufficient.
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
            # Equivalent to:
            #
            # dist_new_test = np.copy(dist_reference)
            # dist_new_test_without_lsst =
            #       dist_new_test - lsst_SNIa
            # =================================================

            dist_new_test_without_lsst = (
                dist_reference - lsst_SNIa
            )

            # =================================================
            # Safety check on kk
            # =================================================

            kk_population = (
                dist_new_test_without_lsst[kk]
            )

            kk_safe = (
                kk_population > perturbation
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
                tt[kk] * perturbation
            )

            # =================================================
            # Reallocation mask
            #
            # NumPy:
            #
            # reallocate = [
            #     j for j in range(len(z_roman))
            #     if j not in ignore
            # ]
            # =================================================

            reallocate_mask = ~ignore

            # =================================================
            # Denominator
            #
            # NumPy:
            #
            # denom = np.sum(dFOM[reallocate])
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
            #
            # NumPy:
            #
            # delta_n[j] =
            #   perturbation * tt[kk] / tt[j]
            #   * dFOM[j] / denom
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
            #
            # NumPy:
            #
            # comparison = np.less_equal(
            #     dist_new_test_without_lsst,
            #     NIa_tot
            # )
            # =================================================

            comparison = (
                dist_after_redistribution
                <= NIa_tot
            )

            all_valid = jnp.all(
                comparison
            )

            problematic_bins = ~comparison

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
        # Find kk
        #
        # NumPy:
        #
        # kk = np.nanargmin(dFOM_triche)
        # ====================================================

        eligible = jnp.isfinite(
            dFOM_triche
        )

        masked_dFOM = jnp.where(
            eligible,
            dFOM_triche,
            jnp.inf,
        )

        kk = jnp.argmin(
            masked_dFOM
        )

        has_eligible = jnp.any(
            eligible
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
        #
        # This is the JAX equivalent of returning/stopping
        # rather than accepting an invalid distribution.
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
            Cinv_new
        )

        # ====================================================
        # FOM
        # ====================================================

        FOM_new, CC_new, ellipse_new = (
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

    return jax.jit(optimizer_step)


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
    cosmo_name,
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

    Cov_stat_base = nearest_psd(
        Cov_roman_stat
    )

    Cov_sys = nearest_psd(
        Cov_roman_sys
    )

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

    track_ellipse = []
    track_FF = []
    track_distribution = []
    track_delta_n = []
    track_Cov = []
    track_dFOM = []
    track_dFOM_triche = []
    track_kk = []
    track_stalled = []
    track_redistribution_valid = []
    track_ignore = []

    # ========================================================
    # CURRENT STATE
    # ========================================================

    dist_reference = dist_roman
    FOM_reference = FOM

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
            # ONE JIT-COMPILED ITERATION
            # =================================================

            (
                dist_reference,
                FOM_reference,
                tracking,
            ) = optimizer_step_jax(
                dist_reference,
                FOM_reference,
            )

            # =================================================
            # FORCE SYNCHRONIZATION
            #
            # This is intentional.
            #
            # It means tqdm represents completed iterations,
            # rather than merely dispatched asynchronous work.
            # =================================================

            jax.block_until_ready(
                dist_reference
            )

            # =================================================
            # STORE TRACKING
            # =================================================

            track_distribution.append(
                tracking["distribution"]
            )

            track_delta_n.append(
                tracking["delta_n"]
            )

            track_Cov.append(
                tracking["Cov"]
            )

            track_dFOM.append(
                tracking["dFOM"]
            )

            track_dFOM_triche.append(
                tracking["dFOM_triche"]
            )

            track_kk.append(
                tracking["kk"]
            )

            track_FF.append(
                tracking["FF"]
            )

            track_ellipse.append(
                tracking["ellipse"]
            )

            track_stalled.append(
                tracking["stalled"]
            )

            track_redistribution_valid.append(
                tracking["redistribution_valid"]
            )

            track_ignore.append(
                tracking["ignore"]
            )

            # =================================================
            # PROGRESS BAR INFO
            #
            # These are ordinary Python operations, not
            # jax.debug.print callbacks.
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

    track_Cov = jnp.stack(
        track_Cov,
        axis=0,
    )

    track_dFOM = jnp.stack(
        track_dFOM,
        axis=0,
    )

    track_dFOM_triche = jnp.stack(
        track_dFOM_triche,
        axis=0,
    )

    track_kk = jnp.stack(
        track_kk,
        axis=0,
    )

    track_FF = jnp.stack(
        track_FF,
        axis=0,
    )

    track_ellipse = jnp.stack(
        track_ellipse,
        axis=0,
    )

    track_stalled = jnp.stack(
        track_stalled,
        axis=0,
    )

    track_redistribution_valid = jnp.stack(
        track_redistribution_valid,
        axis=0,
    )

    track_ignore = jnp.stack(
        track_ignore,
        axis=0,
    )

    # ========================================================
    # FINAL DISTRIBUTION
    # ========================================================

    optimized_dist = (
        jnp.round(
            dist_reference
        )
        .astype(jnp.int64)
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

        "track_Cov":
            track_Cov,

        "track_dFOM":
            track_dFOM,

        "track_dFOM_triche":
            track_dFOM_triche,

        "track_kk":
            track_kk,

        "track_FF":
            track_FF,

        "track_ellipse":
            track_ellipse,

        "track_stalled":
            track_stalled,

        "track_redistribution_valid":
            track_redistribution_valid,

        "track_ignore":
            track_ignore,
    }
