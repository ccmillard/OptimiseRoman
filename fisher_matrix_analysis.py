import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy as sc
import scipy.stats as stats
import astropy.cosmology as ac
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
import numdifftools as nd
from scipy.stats import norm


################################ Hessian Fisher formula ################################

def chi2(params,z,data_obs,cov,cosmology,H0):
    """Compute the chi square
    
    Parameters :
        z (np.ndarray): redshift of data 
        data_obs (np.ndarray): observed dataset, here: apparent magnitude of SNIa (mb)
        cov (np.ndarray): covariance matrix of observation
        params (tuple): Model parameters, depending on the model_type. Respect the order: Om0, Ode0, w0, wa, H0, Mb
        cosmology (str): 
            - 'fLCDM': Flat LambdaCDM
            - 'kLCDM': LambdaCDM with curvature
            - 'fwCDM': Flat wCDM
            - 'kwCDM': wCDM with curvature
            - 'fCPL': Flat CPL
            - 'kCPL': CPL with curvature

    Returns:
        chi2 (float): chi square
    """
    # Define cosmology
    if cosmology == 'fLCDM':
        Om0,  Mb = params
        cosmo = ac.FlatLambdaCDM(H0=H0, Om0=Om0)

    elif cosmology == 'kLCDM':
        Om0, Ode0, Mb = params
        cosmo = ac.LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)

    elif cosmology == 'fwCDM':
        Om0, w0, Mb = params
        cosmo = ac.FlatwCDM(H0=H0, Om0=Om0, w0=w0)

    elif cosmology == 'kwCDM':
        Om0, Ode0, w0, Mb = params
        cosmo = ac.wCDM(H0=H0, Om0=Om0, Ode0=Ode0, w0=w0)

    elif cosmology == 'fCPL':
        Om0, w0, wa, Mb = params
        cosmo = ac.Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa)

    elif cosmology == 'kCPL':
        Om0, Ode0, w0, wa, Mb = params
        cosmo = ac.w0waCDM(H0=H0, Om0=Om0, Ode0=Ode0, w0=w0, wa=wa)
    else:
        raise ValueError(f"Unsupported cosmology: {cosmology}") 

    # expected data
    data_exp = cosmo.distmod(z).value

    #compute chi2
    delta_data = data_obs - (data_exp + Mb) # add Mb to the expected data, since mb = mu + Mb
    
    return delta_data.T @ np.linalg.inv(cov) @ delta_data

chi2_vec = np.vectorize(chi2, excluded=[1,2,3,4,5])


def loglikelihood(params,z,data_obs,cov,cosmology,H0):
    """Compute the loglikelihood
    
    Parameters :
        data_obs (np.ndarray): observed dataset
        dat_exp (np.ndarray): expected dataset
        cov (np.ndarray): covariance matrix of observation

    Returns:
        L (float): Loglikelihood
    """

    return -chi2(params,z,data_obs,cov,cosmology,H0)/2


def fisher_matrix(chi2, param_values, cosmology, H0, z, data_obs, cov, h=None):
    """
    Compute the Fisher matrix for a given cosmological model.

    Parameters:
    - chi2 : The chi2 function
    - param_values : List of parameter values at the truth (e.g., [Om0, w0, wa, Mb])
    - cosmology : Cosmological model ('fLCDM', 'kLCDM', etc.)
    - H0 : Hubble constant
    - z, data_obs, cov : Observational data and covariance matrix
    - h : Step sizes for finite differences (default: 1e-4 for each parameter)

    Returns:
    - Fisher matrix (numpy array)
    """

    num_params = len(param_values)

    if h is None:
        h = np.full(num_params, 1e-4)  # Default step size for all parameters
        
    # Create a meshgrid around the truth values
    param_grids = []
    for i in range(num_params):
        # param_vals = np.array([param_values[i]]) * (1 + np.linspace(-2, 2, 5) * h[i])
        param_vals = param_values[i] + np.linspace(-2, 2, 5) * h[i]
        param_grids.append(param_vals)

    meshgrid = np.meshgrid(*param_grids, indexing='ij')

    # Compute chi2 values at each grid point
    chi2_values = np.zeros(meshgrid[0].shape)
    
    # Iterate over all grid points and evaluate chi2
    for idx in np.ndindex(meshgrid[0].shape):
        point_params = [grid[idx] for grid in meshgrid]  # Extract parameter values at this grid point
        chi2_values[idx] = chi2(tuple(point_params), z, data_obs, cov, cosmology, H0)

    # Compute first derivatives
    first_derivatives = [np.gradient(chi2_values, h[i], axis=i)
                        for i in range(num_params)]

    # Compute second derivatives
    fisher_matrix = np.zeros((num_params, num_params))

    for i in range(num_params):
        fisher_matrix[i, i] = np.gradient(first_derivatives[i],
                                  h[i],
                                  axis=i)[(2,) * num_params]

    for i in range(num_params):
        for j in range(i + 1, num_params):
            fisher_matrix[i, j] = np.gradient(first_derivatives[i],
                                  h[j],
                                  axis=j)[(2,) * num_params]
            fisher_matrix[j, i] = fisher_matrix[i, j]  # Symmetric matrix

    return 0.5 * fisher_matrix  # Scaling factor of 1/2 as per Fisher matrix definition

def fisher_matrix_fd(chi2, param_values, cosmology, H0, z, data_obs, cov, h=None):
    """
    Compute Fisher matrix using central finite differences (no grid).

    Much faster and more stable than grid-based version.
    """

    params = np.array(param_values, dtype=float)
    n = len(params)

    if h is None:
        h = 1e-3 * np.abs(params)

    F = np.zeros((n, n))

    chi2_0 = chi2(tuple(params), z, data_obs, cov, cosmology, H0)

    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal element: second derivative w.r.t. param i
                p_plus = params.copy()
                p_minus = params.copy()

                p_plus[i] += h[i]
                p_minus[i] -= h[i]

                chi_plus = chi2(tuple(p_plus), z, data_obs, cov, cosmology, H0)
                chi_minus = chi2(tuple(p_minus), z, data_obs, cov, cosmology, H0)

                F_ij = (chi_plus - 2 * chi2_0 + chi_minus) / (h[i] ** 2)
                F[i, j] = 0.5 * F_ij


            # Create parameter shifts
            p_pp = params.copy()
            p_pm = params.copy()
            p_mp = params.copy()
            p_mm = params.copy()

            p_pp[i] += h[i]; p_pp[j] += h[j]
            p_pm[i] += h[i]; p_pm[j] -= h[j]
            p_mp[i] -= h[i]; p_mp[j] += h[j]
            p_mm[i] -= h[i]; p_mm[j] -= h[j]

            # Evaluate chi2 at 4 points
            chi_pp = chi2(tuple(p_pp), z, data_obs, cov, cosmology, H0)
            chi_pm = chi2(tuple(p_pm), z, data_obs, cov, cosmology, H0)
            chi_mp = chi2(tuple(p_mp), z, data_obs, cov, cosmology, H0)
            chi_mm = chi2(tuple(p_mm), z, data_obs, cov, cosmology, H0)

            # Second derivative
            F_ij = (chi_pp - chi_pm - chi_mp + chi_mm) / (4 * h[i] * h[j])

            F[i, j] = 0.5 * F_ij
            F[j, i] = F[i, j]  # symmetry

    return F

def marginalize_fisher_matrix(fisher_matrix, param_tuple, param_names):
    """
    Extracts the marginalized Fisher submatrix for the given parameters.

    Parameters:
        fisher_matrix (np.array): The full Fisher matrix (NxN).
        param_tuple (tuple of str): The parameters to keep, e.g., ('Om0', 'w0').
        param_names (list of str): The ordered list of parameter names corresponding to the Fisher matrix.

    Returns:
        cov_submatrix (np.array): The inverted Fisher submatrix (covariance matrix) for the selected parameters.
    """
    #Invert NxN fisher matrix
    fisher_matrix_inv = np.linalg.inv(fisher_matrix)

    # Extract the rows and columns of interest
    param_index = {name: i for i, name in enumerate(param_names)} # Create dictionary to map parameter names to indices
    selected_indices = [param_index[param] for param in param_tuple] # Extract indices for the parameters of interest
    cov_submatrix = fisher_matrix_inv[np.ix_(selected_indices, selected_indices)] # Extract the submatrix
    
    return cov_submatrix  # This can be used as input for ellipse_parameter()

def is_positive_definite(C):
    """
    Check if a matrix is positive definite using Cholesky decomposition.
    
    Parameters:
        Cov (np.ndarray): covariance matrix
    
    Returns:
        (bool): True if the matrix is positive definite, False otherwise.
    """
    try:
        # Attempt Cholesky decomposition
        np.linalg.cholesky(C)
        return True  # If it succeeds, the matrix is positive definite
    except np.linalg.LinAlgError:
        return False  # If decomposition fails, the matrix is not positive definite

def eigenvalues_clipping(C,epsilon=1e-12):
    '''Clip eigenvalues at a minimum value min_eigvals
    
    Parameters:
        C (np.ndarray): covariance matrix
        min_eigvals (float): minimum value below which eigenvalues will be clipped at.
        
    Returns:
        C_clipped (np.ndarray): covariance matrix with clipped eigenvalues
    '''
    
    # symmetrize
    C_sym = 1/2 * (C + C.transpose())

    # diagonalize
    eigvals, eigvecs = np.linalg.eigh(C_sym)

    # clip eigenvalues
    eigvals_clipped = np.maximum(eigvals, epsilon)

    # reconstruct clipped covariance matrix
    C_clipped = eigvecs @ np.diag(eigvals_clipped) @ eigvecs 

    return C_clipped

def diagonal_perturbation(C, epsilon):
    '''
    Add a small perturbation to the diagonal of an almost PSD matrix, tpo make it PSD
    
    Parameters:
        C (np.ndarray): covariance matrix
        eps (float): perturbation

    Returns:
        Cprime (np.ndarray): Perturbed covariance matrix
    '''

    I = np.identity(len(C))

    return C + epsilon*I

def ellipse_parameters(C, alpha, epsilon=1e-12):
    ''''
    Compute the semi-axis, the angle and the area of an ellipse
    
    Parameter:
        Cov (np.array): 2x2 Covariance matrix
        alpha (float): Confidence level of the ellipse
        epsilon (float): minimum value below which eigenvalues will be clipped at, or diagonal perturbation

    Returns:
        a (float): Major semi axis
        b (float): Minor semi_axis
        theta (float): counterclockwise angle in degree
        A (float): area
    '''

    # Test positive definiteness
    test = is_positive_definite(C)
    if test==True: # proceed
        pass
    else: 
        print('Not positive semi-definite')
        return None

    # Diagonalise the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Sort indices in descending order of eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1] 
    lambda1, lambda2 = eigenvalues[sorted_indices]
    v = eigenvectors[:, sorted_indices]  # Columns correspond to eigenvectors
  
    # Compute ellipse's parameters :
    width,height = np.sqrt(sc.stats.chi2.ppf(1-alpha,2)*lambda1),np.sqrt(sc.stats.chi2.ppf(1-alpha,2)*lambda2)
    theta = np.degrees(np.arctan2(v[:,0][1],v[:,0][0]))
    A = np.pi*width*height

    return width, height, theta, A 

def create_fisher_plot(param_names, param_to_plot=None, param_labels=None, figsize=8):
    """
    Initializes a grid of subplots for Fisher matrix visualization.

    Parameters:
        param_names (list of str): List of all parameter names.
        param_to_plot (list of str, optional): Names of parameters to include in the plot.
            If None, all parameters are used.
        param_labels (list of str, optional): Labels to display instead of names.
            If None, `param_names` are used.
        figsize (float, optional): Base size of the square plot (in inches).

    Returns:
        fig (matplotlib.figure.Figure): The created figure object.
        gs (matplotlib.gridspec.GridSpec): The gridspec layout for subplot alignment.
        axs_2d (dict): Dictionary of 2D axis handles with keys (i, j).
        axs_1d (dict): Dictionary of 1D axis handles with key i.
        param_indices (list of int): Indices of selected parameters in `param_names`.
    """
    if param_to_plot is None:
        param_indices = list(range(len(param_names)))
    else:
        param_indices = [param_names.index(p) for p in param_to_plot]

    if param_labels is None:
        param_labels = [param_names[i] for i in param_indices]

    num_param = len(param_indices)
    fig = plt.figure(figsize=(figsize, figsize), constrained_layout=True)
    gs = GridSpec(num_param, num_param, figure=fig)
    axs_2d = {}
    axs_1d = {}

    for i in range(num_param):
        for j in range(num_param):
            if i == j:
                axs_1d[i] = fig.add_subplot(gs[i, j])
            elif i < j:
                axs_2d[(i, j)] = fig.add_subplot(gs[j, i])

    for i, ax in axs_1d.items():
        ax.set_xlabel(param_labels[i])
        ax.set_ylabel('Density')

    for (i, j), ax in axs_2d.items():
        ax.set_xlabel(param_labels[i])
        ax.set_ylabel(param_labels[j])
        ax.set_aspect('equal', 'box')

    return fig, gs, axs_2d, axs_1d, param_indices

def add_fisher_to_plot(
    fisher_matrix,
    fiducial_param,
    axs_2d,
    axs_1d,
    param_indices,
    alpha_values=[0.32, 0.05, 0.01],
    color="red",
    filled=True,
    fontsize=14,
    param_labels=None
):
    """
    Add Fisher ellipses and 1D Gaussians to subplots, using only a selected subspace of parameters.

    fisher_matrix (ndarray): Full (n x n) Fisher matrix.
    fiducial_param (list): Full fiducial parameters, length n.
    param_indices (list): Indices of the parameters to plot.
    """

    # Slice to selected subspace
    F_sub = fisher_matrix[np.ix_(param_indices, param_indices)]
    fid_sub = np.array(fiducial_param)[param_indices]
    if is_positive_definite(np.linalg.inv(F_sub)):
        inv_fisher = np.linalg.inv(F_sub)
    else:
        inv_fisher = eigenvalues_clipping(np.linalg.inv(F_sub))
    constraints = np.sqrt(np.diag(inv_fisher))

    if param_labels is None:
        param_labels = [f"param_{i}" for i in param_indices]

    n_param = len(param_indices)

    for i in range(n_param):
        for j in range(i + 1, n_param):
            center = [fid_sub[i], fid_sub[j]]
            sub_cov = inv_fisher[np.ix_([i, j], [i, j])]

            ellipses = [ellipse_parameters(sub_cov, alpha) for alpha in alpha_values]

            ax = axs_2d.get((i, j))
            if ax is not None:
                max_w = max(e[0] for e in ellipses)
                max_h = max(e[1] for e in ellipses)
                lim = max(max_w, max_h)

                ax.set_xlim(center[0] - 1.1 * lim, center[0] + 1.1 * lim)
                ax.set_ylim(center[1] - 1.1 * lim, center[1] + 1.1 * lim)

                for (w, h, a, _) in ellipses:
                    ell = Ellipse(
                        xy=center, width=2*w, height=2*h, angle=a,
                        facecolor=color if filled else "none",
                        edgecolor=color, alpha=0.3 if filled else 1.0,
                        linewidth=1.5
                    )
                    ax.add_patch(ell)

    for i in range(n_param):
        mu, sigma = fid_sub[i], constraints[i]
        x = np.linspace(mu - 5*sigma, mu + 5*sigma, 1000)
        ax = axs_1d.get(i)
        if ax is not None:
            ax.plot(x, norm.pdf(x, mu, sigma), color=color)
            ax.set_xlabel(param_labels[i], fontsize=fontsize)


################################ derivative-of-observables Fisher formula ################################
def theory_vector(
    params,
    z,
    cosmology,
    H0
):
    """
    Returns theory prediction vector.

    """

    # define the cosmology
    if cosmology == 'fLCDM':
        Om0,  Mb = params
        cosmo = ac.FlatLambdaCDM(H0=H0, Om0=Om0)

    elif cosmology == 'kLCDM':
        Om0, Ode0, Mb = params
        cosmo = ac.LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)

    elif cosmology == 'fwCDM':
        Om0, w0, Mb = params
        cosmo = ac.FlatwCDM(H0=H0, Om0=Om0, w0=w0)

    elif cosmology == 'kwCDM':
        Om0, Ode0, w0, Mb = params
        cosmo = ac.wCDM(H0=H0, Om0=Om0, Ode0=Ode0, w0=w0)

    elif cosmology == 'fCPL':
        Om0, w0, wa, Mb = params
        cosmo = ac.Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa)

    elif cosmology == 'kCPL':
        Om0, Ode0, w0, wa, Mb = params
        cosmo = ac.w0waCDM(H0=H0, Om0=Om0, Ode0=Ode0, w0=w0, wa=wa)
    else:
        raise ValueError(f"Unsupported cosmology: {cosmology}") 
    
    # compute µ and mb
    mu_model = cosmo.distmod(z).value
    mb_model = mu_model + Mb

    return np.asarray(mb_model)


def derivative_vector(
    params,
    i,
    h,
    z,
    cosmology,
    H0
):

    # define parameters vectors for finite difference
    p_plus = np.array(params, dtype=float)
    p_minus = np.array(params, dtype=float)

    # add or remove step h to the i-th parameter
    p_plus[i] += h
    p_minus[i] -= h

    # compute theory vector at p_plus and p_minus
    mu_plus = theory_vector(
        p_plus,
        z,
        cosmology,
        H0
    )

    mu_minus = theory_vector(
        p_minus,
        z,
        cosmology,
        H0
    )

    dmu = (mu_plus - mu_minus) / (2*h)

    return dmu


def fisher_matrix_observable(
    params,
    Cinv,
    z,
    cosmology,
    H0,
    h=None
):

    params = np.array(params, dtype=float)

    npar = len(params)

    # -----------------------------
    # Step sizes
    # -----------------------------
    if h is None:
        h = np.full(npar, 1e-4)
        # à revoir : peut-être adapter h à chaque paramètre, en fonction de son échelle ? (ex: h = 1e-4 * params)

    # -----------------------------
    # Derivative vectors
    # -----------------------------
    derivs = np.zeros((npar, len(z)))  #  derivs initialized to hold derivative vectors

    for i in range(npar):

        dmu = derivative_vector(
            params,
            i,
            h[i],
            z,
            cosmology,
            H0
        )

        derivs[i] = dmu

    # -----------------------------
    # Fisher matrix
    # -----------------------------
    F = np.zeros((npar, npar))

    for i in range(npar):

        for j in range(i, npar):

            Fij = derivs[i] @ Cinv @ derivs[j]

            F[i, j] = Fij
            F[j, i] = Fij

    # explicit symmetrization
    F = 0.5 * (F + F.T)

    return F