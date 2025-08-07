import numpy as np
from scipy.stats import norm
import argparse

# Reuse helper functions and constants from compute_A_eta
from compute_A_eta import (
    sample_lens_population,
    compute_magnifications,
    ms_distribution,
    selection_function,
    OBS_SCATTER_MAG,
)


def compute_single_A_eta(mu_DM, sigma_DM, beta_DM, n_samples=100000,
                          ms_points=15, m_lim=26.5, n_jobs=None):
    """High precision Monte Carlo estimate of A for a single (mu, sigma, beta).

    Parameters
    ----------
    mu_DM, sigma_DM, beta_DM : float
        Parameters describing the dark matter halo distribution.
    n_samples : int, optional
        Number of Monte Carlo lens samples to draw.  Larger values give
        higher precision at the cost of longer runtime.
    ms_points : int, optional
        Number of points to use when integrating over the source magnitude
        distribution.
    m_lim : float, optional
        Limiting magnitude of the survey.
    n_jobs : int, optional
        Number of processes to use when solving lens equations.  Defaults to
        ``None`` which lets ``ProcessPoolExecutor`` pick.
    """

    # Sample lenses and compute magnifications
    samples = sample_lens_population(n_samples)
    mu1, mu2 = compute_magnifications(
        samples["logM_star"],
        samples["logRe"],
        samples["logMh"],
        samples["beta"],
        samples["zl"],
        samples["zs"],
        n_jobs=n_jobs,
    )

    # Detection probability weights
    ms_grid = np.linspace(20.0, 30.0, ms_points)
    pdf_ms = ms_distribution(ms_grid)
    sel1 = selection_function(mu1[:, None], m_lim, ms_grid[None, :], OBS_SCATTER_MAG)
    sel2 = selection_function(mu2[:, None], m_lim, ms_grid[None, :], OBS_SCATTER_MAG)
    p_det = sel1 * sel2
    w_ms = np.trapezoid(p_det * pdf_ms[None, :], ms_grid, axis=1)
    w_static = w_ms  # Beta sampling correction already included in sampling

    # Probability of sampled halo mass under the assumed DM distribution
    mean = mu_DM + beta_DM * (samples["logM_star"] - 11.4)
    p_Mh = norm.pdf(samples["logMh"], loc=mean, scale=sigma_DM)

    Mh_range = samples.get("logMh_max", 15.0) - samples.get("logMh_min", 11.0)
    A = Mh_range * np.sum(w_static * p_Mh) / n_samples
    return A


def main():
    parser = argparse.ArgumentParser(
        description="Compute high-precision A for a single eta (mu,sigma,beta)")
    parser.add_argument("--mu_DM", type=float, required=True)
    parser.add_argument("--sigma_DM", type=float, required=True)
    parser.add_argument("--beta_DM", type=float, required=True)
    parser.add_argument("--n_samples", type=int, default=100000)
    parser.add_argument("--ms_points", type=int, default=15)
    parser.add_argument("--m_lim", type=float, default=26.5)
    parser.add_argument("--n_jobs", type=int, default=None)
    args = parser.parse_args()

    A = compute_single_A_eta(
        args.mu_DM,
        args.sigma_DM,
        args.beta_DM,
        n_samples=args.n_samples,
        ms_points=args.ms_points,
        m_lim=args.m_lim,
        n_jobs=args.n_jobs,
    )
    print(A)


if __name__ == "__main__":
    main()
