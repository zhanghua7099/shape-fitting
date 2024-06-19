# Cylinder Fitting Using Hough Transform

## Installation

- Clone this repository
- `pip install ./cy_fit_hough`

## Test call

```python
import cy_fit_hough
import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("pcd.pcd")

cy_params = cy_fit_hough.cylinder_fitting_Figueiredo(np.array(pcd.points))

```
