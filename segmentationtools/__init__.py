import open3d as o3d
import matplotlib
import numpy as np

def show_geometries(geometries : 'List[o3d.geometry]', color : bool = False):
    """Displays different types of geometry in a scene

    Args:
        geometries (List[open3d.geometry]): The list of geometries
        color (bool, optional): recolor the objects to have a unique color. Defaults to False.
    """

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    viewer.add_geometry(frame)
    for i, geometry in enumerate(geometries):
        if color:
            geometry.paint_uniform_color(matplotlib.colors.hsv_to_rgb([float(i)/len(geometries),0.8,0.8]))
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.background_color = np.asarray([1,1,1])
    opt.light_on = True
    viewer.run()