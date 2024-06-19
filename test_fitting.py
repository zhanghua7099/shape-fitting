import cy_fit_hough
import numpy as np
import open3d as o3d


pcd = o3d.io.read_point_cloud("./sample/cylinder.pcd")


cy_params = cy_fit_hough.cylinder_fitting_Figueiredo(np.array(pcd.points))
print(cy_params)
