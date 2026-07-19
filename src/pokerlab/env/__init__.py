"""M6 vectorized env + §6.1 perturbation bot + exact-BR exploiter.

Approved deviation: numpy batching over the Slice-A engine instead of Rust; the
port seam is the API contract (VecEnv/encoding) plus the shared M0 differential.

    from pokerlab.env import (
        VecEnv, EnvConfig, SingleEnvAdapter, single_env_action,   # env surface
        PerturbationBot,                                          # §6.1 bot
        encode_obs, decode_action, legal_action_mask, OBS_DIM,    # encodings
    )

Exact best response vs the (queryable) bot is computed with `cfr.best_response`
on a toy tree; see tests/test_env_exploit.py.
"""

from pokerlab.env.bot import AGGRESSIVE_LABELS, PerturbationBot, perturb_dist
from pokerlab.env.encoding import (
    DEFAULT_FRACTIONS,
    OBS_DIM,
    action_names,
    decode_action,
    encode_obs,
    legal_action_mask,
    legal_engine_actions,
    num_actions,
)
from pokerlab.env.vec import (
    EnvConfig,
    SingleEnvAdapter,
    VecEnv,
    single_env_action,
)

__all__ = [
    "VecEnv",
    "EnvConfig",
    "SingleEnvAdapter",
    "single_env_action",
    "PerturbationBot",
    "perturb_dist",
    "AGGRESSIVE_LABELS",
    "encode_obs",
    "decode_action",
    "legal_action_mask",
    "legal_engine_actions",
    "action_names",
    "num_actions",
    "OBS_DIM",
    "DEFAULT_FRACTIONS",
]
