"""Excavator adapters for https://arxiv.org/abs/2606.22836v2.

Use ``python -m policy.training.cloak --help`` for preparation, training and serving.
The attention implementation comes from the pinned Stanford CLOAK release.
"""

UPSTREAM_URL = 'https://github.com/Stanford-TML/cloak.git'
UPSTREAM_REVISION = 'f3d6ab37bb850bfe63430adf9f769e6eda4e10e3'
WRIST = 'left_wrist_0_rgb'
ASSET = 'excavator_cloak'
