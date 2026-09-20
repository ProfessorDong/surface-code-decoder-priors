"""
Quantum Error Correction Simulation Package
=============================================

Adaptive GNN-based decoder with contractive latent dynamics, deep ensemble
uncertainty quantification, and risk-limited fallback for circuit-level noise
on rotated surface codes.

Modules:
    data        -- Stim circuit construction, noise models, graph conversion
    models      -- GNN, GRU, latent state, output head, ensemble decoder
    baselines   -- MWPM, Union-Find, static ML, oracle MWPM decoders
    training    -- Training loop, losses, calibration
    evaluation  -- Metrics, evaluator, risk-limited policy
    plotting    -- IEEE-style figures and tables
    scripts     -- End-to-end pipeline scripts
"""

__version__ = "0.1.0"
