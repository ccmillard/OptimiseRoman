import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optimisation_functions as opf
import optimisation_functions_jax as opfj
import time


args = np.load(
    "arguments.npz",
    allow_pickle=True,
)


result = opf.optimize_bins_gen(
    nb_iter=iter,
    perturbation=args["perturbation"],
    dist_roman=args["dist_roman"],
    Cov_roman=args["Cov_roman"],
    Cov_roman_stat=args["Cov_roman_stat"],
    Cov_roman_sys=args["Cov_roman_sys"],
    FOM=args["FOM"],
    H0=args["H0"],
    z_roman=args["z_roman"],
    tt=args["tt"],
    fiducial_cosmo=args["fiducial_cosmo"],
    cosmo_name=args["cosmo_name"],
    param_tuple=tuple(args["param_tuple"]),
    param_names=args["param_names"],
    use_marginalization=args["use_marginalization"],
    verbose=args["verbose"],
)





result_jax = opfj.optimize_bins_gen_jax(
    nb_iter=iter,
    perturbation=float(args["perturbation"]),
    dist_roman=jnp.array(args["dist_roman"]),
    Cov_roman=jnp.array(args["Cov_roman"]),
    Cov_roman_stat=jnp.array(args["Cov_roman_stat"]),
    Cov_roman_sys=jnp.array(args["Cov_roman_sys"]),
    FOM=float(args["FOM"]),
    H0=float(args["H0"]),
    z_roman=jnp.array(args["z_roman"]),
    tt=jnp.array(args["tt"]),
    fiducial_cosmo=jnp.array(args["fiducial_cosmo"]),
    param_tuple=tuple(
        str(x)
        for x in args["param_tuple"]
    ),
    param_names=tuple(
        str(x)
        for x in args["param_names"]
    ),
    use_marginalization=bool(
        args["use_marginalization"]
    ),
)













area1 = result['track_ellipse'][:, -1]
area2 = result_jax['track_ellipse'][:, -1]



# FoM = inverse area
fom1 = 1.0 / area1
fom2 = 1.0 / area2
n = len(fom1)



# ----------------------------------------
# Plot FoM evolution
# ----------------------------------------

plt.figure(figsize=(9,6))

#plt.plot(fom1, label='r1 FoM', lw=2)
#plt.plot(fom2, label='r2 FoM', lw=2, linestyle='--')
plt.plot((fom1-fom2[:n])/fom1)

plt.xlabel("Iteration")
plt.ylabel("FoM = 1 / area")
plt.title("FoM Evolution Comparison")

plt.legend()
plt.grid(True)

plt.show()