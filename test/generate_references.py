import os
import json
import numpy as np
import sys

# Add jax-loglike to path
sys.path.append("/home/marcobonici/Desktop/work/CosmologicalLikelihoods/jax-loglike")

from jax_loglike.hillipop import HillipopPR4
from jax_loglike.module import array

def generate_cases():
    # Setup HillipopPR4
    # Note: we need to provide the compressed covariance info to avoid loading the huge FITS if possible,
    # but the Julia side is already using the compressed version.
    # Let's see if we can just initialize it.
    
    # We need to simulate the additional_args used in Julia
    # binning_matrix and binned_invkll were loaded from .npy files
    data_dir = "/home/marcobonici/Desktop/work/CosmologicalLikelihoods/Hillipop.jl/data" # Wait, I deleted this!
    # I should use the artifact path if I can find it, or just use the jax-loglike/data if it has it.
    
    jax_data_dir = "/home/marcobonici/Desktop/work/CosmologicalLikelihoods/jax-loglike/data/planck_pr4_hillipop"
    
    # Load compressed covariance info to match Julia setup and avoid missing FITS
    binning_matrix = np.load(os.path.join(jax_data_dir, "binning_matrix.npy"), allow_pickle=True)
    binned_invkll = np.load(os.path.join(jax_data_dir, "binned_invkll.npy"), allow_pickle=True)
    
    print(f"binning_matrix type: {type(binning_matrix)}, shape: {binning_matrix.shape}")
    if binning_matrix.ndim == 0:
        print("binning_matrix is 0-dim, extracting item...")
        binning_matrix = binning_matrix.item()
        print(f"new binning_matrix type: {type(binning_matrix)}, shape: {getattr(binning_matrix, 'shape', 'N/A')}")

    print(f"binned_invkll type: {type(binned_invkll)}, shape: {binned_invkll.shape}")
    if binned_invkll.ndim == 0:
        print("binned_invkll is 0-dim, extracting item...")
        binned_invkll = binned_invkll.item()
        print(f"new binned_invkll type: {type(binned_invkll)}, shape: {getattr(binned_invkll, 'shape', 'N/A')}")

    
    additional_args = {
        "binning_matrix": binning_matrix,
        "binned_invkll": binned_invkll
    }
    
    # Actually, I'll just use the fiducial initialization.
    hp = HillipopPR4(additional_args=additional_args)
    
    lmax = 2500
    ells = np.arange(2, lmax + 1)
    
    def get_logL(cltt, clte, clee, pars):
        # HillipopPR4 expects batched parameters if Cls are batched
        batched_pars = {k: np.array([v]) for k, v in pars.items()}
        return hp.compute_like(cltt[None, :], clte[None, :], clee[None, :], batched_pars)[0]

    cases = []
    
    # Case 1: Fiducial-ish
    pars_fid = {
        "cal100A": 1.0, "cal100B": 1.0, "cal143A": 1.0, "cal143B": 1.0, "cal217A": 1.0, "cal217B": 1.0,
        "pe100A": 1.0, "pe100B": 1.0, "pe143A": 1.0, "pe143B": 1.0, "pe217A": 1.0, "pe217B": 1.0,
        "A_planck": 1.0, "AdustT": 1.0, "AdustP": 1.0, "beta_dustT": 1.5, "beta_dustP": 1.5,
        "Atsz": 1.0, "Aksz": 1.0, "Acib": 1.0, "beta_cib": 1.75, "xi": 0.1,
        "Aradio": 0.0, "beta_radio": -0.7, "Adusty": 0.0
    }
    cltt_fid = 6000.0 / (ells * (ells + 1)) * 1e-12
    clte_fid = np.zeros_like(ells)
    clee_fid = cltt_fid * 0.01
    
    logL_fid = get_logL(cltt_fid, clte_fid, clee_fid, pars_fid)
    cases.append({
        "name": "fiducial",
        "params": pars_fid,
        "cltt": cltt_fid.tolist(),
        "clte": clte_fid.tolist(),
        "clee": clee_fid.tolist(),
        "logL": float(logL_fid)
    })
    
    # Case 2: Zero Cls (Foreground only)
    cl_zero = np.zeros_like(ells)
    logL_fg = get_logL(cl_zero, cl_zero, cl_zero, pars_fid)
    cases.append({
        "name": "foreground_only",
        "params": pars_fid,
        "cltt": cl_zero.tolist(),
        "clte": cl_zero.tolist(),
        "clee": cl_zero.tolist(),
        "logL": float(logL_fg)
    })
    
    # Case 3: High Foreground
    pars_high_fg = pars_fid.copy()
    pars_high_fg.update({"AdustT": 10.0, "Atsz": 5.0, "Acib": 2.0})
    logL_high_fg = get_logL(cltt_fid, clte_fid, clee_fid, pars_high_fg)
    cases.append({
        "name": "high_foreground",
        "params": pars_high_fg,
        "cltt": cltt_fid.tolist(),
        "clte": clte_fid.tolist(),
        "clee": clee_fid.tolist(),
        "logL": float(logL_high_fg)
    })
    
    # Case 4: Random
    np.random.seed(42)
    cltt_rand = np.random.rand(len(ells)) * 1e-10
    clte_rand = np.random.rand(len(ells)) * 1e-11
    clee_rand = np.random.rand(len(ells)) * 1e-11
    pars_rand = {k: v * (0.9 + 0.2 * np.random.rand()) for k, v in pars_fid.items()}
    # ensure A_planck is reasonable
    pars_rand["A_planck"] = 0.99 + 0.02 * np.random.rand()
    
    logL_rand = get_logL(cltt_rand, clte_rand, clee_rand, pars_rand)
    cases.append({
        "name": "random",
        "params": pars_rand,
        "cltt": cltt_rand.tolist(),
        "clte": clte_rand.tolist(),
        "clee": clee_rand.tolist(),
        "logL": float(logL_rand)
    })

    with open("/home/marcobonici/Desktop/work/CosmologicalLikelihoods/Hillipop.jl/test/variational_references.json", "w") as f:
        json.dump(cases, f)

if __name__ == "__main__":
    generate_cases()
