import open3d as o3d
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from typing import List
import cv2
import math


def show_img(img, switchChannels: bool = False, title:str = None):
    if(switchChannels):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if(title): plt.title = str(title)
    plt.imshow(img)
    plt.show()

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

def show_detected_lines(imgPath):
    img = cv2.imread(imgPath)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dst = cv2.Canny(img, 100, 300, None, 3)
    lines = cv2.HoughLines(dst, 1.5, np.pi / 120, 150, None, 0, 0)
    fig.suptitle(str('image: ' + imgPath + '\n Lines Detected: ' + str(len(lines))))
    for line in lines:
        points = get_edge_points(line, img.shape[1], img.shape[0])
        ax2.plot(*zip(*points),color='orangered', linewidth=2)
    plt.show()


def detect_edges(img: np.array) -> np.array:
    """Finds straight lines in the images and converts them to parametric polar lines

    Args:
        img (cv2.image): The input image.

    Returns:
        np.array: an array of parametric polar lines
    """
 
    # First detect the edges using Canny detection to convert the image to a bw edge image.
    dst = cv2.Canny(img, 50, 200, None, 3)

    # Use Standard HoughLine transform to detect parametric lines explressed in polar coordinates (rho, theta)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    return lines



def get_point_on_polar_line(line: np.array, x:float = None, y:float = None) -> float:
    """Returns the corresponding value of a single axis value on a polar line

    Args:
        line (np.array): The polar line [[rho,theta]]
        x (float, optional): The X coordinate, leave empty if you want to use the y value. Defaults to None.
        y (float, optional): The Y coordinate. Defaults to None.

    Returns:
        float: the corresponding x or y value
        None: if no value was provided
    """

    rho = line[0][0]
    theta = line[0][1]
    cost = math.cos(theta)
    sint = math.sin(theta)
    a = -cost/sint # the slope
    b = rho/sint # the constant

    if(x is not None):
        return a * x + b
    elif(y is not None):
        return (y - b) / a
    else:
        return None

def get_edge_points(line: np.array, xMax: float, yMax: float) -> np.array:
    """Returns intersections between a line and a bounds area

    Args:
        line (np.array): The polar line to use for the intersection [[rho,theta]]
        xMax (float): the max x value of the bounds
        yMax (float): the max y value of the bounds

    Returns:
        np.array: the array of intersectiong points along the edges
    """

    points = []

    # check the min x value
    y0 = get_point_on_polar_line(line, x = 0)
    if(y0 > 0 and y0 < yMax):
        points.append(np.array([0,y0]))

    # check the max x value
    y1 = get_point_on_polar_line(line, x = xMax)
    if(y1 > 0 and y1 < yMax):
        points.append(np.array([xMax,y1]))

    # check the min y value
    x0 = get_point_on_polar_line(line, y = 0)
    if(x0 > 0 and x0 < xMax):
        points.append(np.array([x0,0]))

    # check the max y value
    x1 = get_point_on_polar_line(line, y = yMax)
    if(x1 > 0 and x1 < xMax):
        points.append(np.array([x1,yMax]))
        
    if(len(points) == 0): 
        return None
    return np.array(points)

def line_intersection(line1: np.array, line2: np.array) -> np.array:
    """Calculates an instesection point between 2 lines defined by a pair of points

    Args:
        line1 (np.array): The first line as a couple of points [[xy],[xy]]
        line2 (np.array): The second line as a couple of points [[xy],[xy]]

    Returns:
        np.array: The resulting intersection point
        None: if the lines are parallel 
    """

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array((x, y))

def check_bounds(point:np.array, bounds:np.array) -> bool:
    """Checks if a point lies withing a bounding area defined by a couple of line points

    Args:
        point (np.array): The point to test
        bounds (np.array): The points defined as a array of 2 points [[xy],[xy]]

    Returns:
        bool: if it's inside the bounds or not
    """

    bounds = np.round(np.array(bounds),1)
    minx = np.min(bounds[:,0])
    maxx = np.max(bounds[:,0])
    miny = np.min(bounds[:,1])
    maxy = np.max(bounds[:,1])

    roundPoint = np.round(point,1)
    if(roundPoint[0] >= minx and roundPoint[0] <= maxx and roundPoint[1] >= miny and roundPoint[1] <= maxy):
        return True
    return False

def line_segment_intersection(line1: np.array,line2:np.array) -> np.array:
    """Calculates an instesection point between 2 line segments defined by a pair of points

    Args:
        line1 (np.array): The first line segment as a couple of points [[xy],[xy]]
        line2 (np.array): The second line segment as a couple of points [[xy],[xy]]

    Returns:
        np.array: The resulting intersection point
        None: if the lines are parallel or not instersecting
    """

    point = line_intersection(line1,line2)
    if(check_bounds(point, line1) and check_bounds(point, line2)):
        return point
    return None

def find_triangle_intersection(edgePoints: np.array, line: np.array) -> np.array:
    """Finds intersecting points between the edges of a triangle and a line

    Args:
        edgePoints (np.array): The 3 points of the triangle [[xy],[xy],[xy]]
        line (np.array): The intersection line as a couple of points [[xy],[xy]]

    Returns:
        np.array: An array of the intersection points
        None: If there are no intersections
    """
    
    tLine1 = ((edgePoints[0][0], edgePoints[0][1]), (edgePoints[1][0], edgePoints[1][1]))
    tLine2 = ((edgePoints[1][0], edgePoints[1][1]), (edgePoints[2][0], edgePoints[2][1]))
    tLine3 = ((edgePoints[2][0], edgePoints[2][1]), (edgePoints[0][0], edgePoints[0][1]))

    points = []
    point1 = line_segment_intersection(tLine1, line)
    if(point1 is not None): points.append(point1)
    point2 = line_segment_intersection(tLine2, line)
    if(point2 is not None): points.append(point2)
    point3 = line_segment_intersection(tLine3, line)
    if(point3 is not None): points.append(point3)

    if(len(points) == 0): 
        return None
    return np.array(points)
