import numpy as np
import matplotlib
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


iters = [5, 10, 25, 50, 100, 250, 500, 750, 1000]
#iters = [100]
times = []
times_jax = []


for iter in iters:

    start = time.perf_counter()


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

    end = time.perf_counter()
    times.append(end-start)
    print(f"OPTIMIZE BINS NUMPY Execution time: {end - start:.4f} seconds")
    np.savez( f"result_numpy_iter_{iter}.npz", result=np.array(result), ) 
    print( f"Saved NumPy result for iter={iter}" )


    start = time.perf_counter()

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
    
    jax.block_until_ready(result_jax)

    end = time.perf_counter()
    times_jax.append(end-start)
    print(
        f"OPTIMIZE BINS JAX Execution time: "
        f"{end - start:.4f} seconds"
    )
     # Move from device -> NumPy 
    result_jax_np = jax.device_get(result_jax) 
    np.savez( f"result_jax_gpu_iter_{iter}.npz", result=result_jax_np, ) 
    print( f"Saved JAX result for iter={iter}")

np.savez("timing_results_gpu.npz", nb_iter=iters, time_numpy=times, time_jax=times_jax)


#PLOT 

plt.figure(figsize=(8, 5))

plt.plot(
    iters,
    times,
    'o--',
    label='NumPy / Standard',
    linewidth=2,
    markersize=6,
)

plt.plot(
    iters,
    times_jax,
    's--',
    label='JAX',
    linewidth=2,
    markersize=6,
)

plt.xlabel("Number of Iterations", fontsize=12)
plt.ylabel("Execution Time (s)", fontsize=12)

plt.title(
    "Execution Time Comparison: Standard vs JAX Optimization",
    fontsize=14,
)

plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Optional:
# plt.yscale('log')

plt.tight_layout()
plt.savefig("timing_comparison_gpu.png", dpi=300)


#JAX_PLATFORMS=cpu python
