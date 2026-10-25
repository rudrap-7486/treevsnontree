"""Tree vs non-tree classification of urban LiDAR point clouds.

CEC201 term project starter kit. Pipeline:
    tiles -> CSF ground filter -> height above ground
          -> eigenvalue (covariance) features at one or more radii
          -> Random Forest -> per-point tree / non-tree label
          -> accuracy assessment on held-out tiles
"""

__version__ = "0.1.0"
