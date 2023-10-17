Search.setIndex({"docnames": ["examples/edgedetection", "examples/faceslicing", "examples/texturedetection", "examples/workflow", "index", "information/overview", "segmentationtools/modules", "segmentationtools/segmentationtools"], "filenames": ["examples/edgedetection.ipynb", "examples/faceslicing.ipynb", "examples/texturedetection.ipynb", "examples/workflow.ipynb", "index.rst", "information/overview.md", "segmentationtools/modules.rst", "segmentationtools/segmentationtools.rst"], "titles": ["Edge detection", "Face Slicing", "&lt;no title&gt;", "Texture-based separation to refine building meshes", "Welcome to the SegmentationTools\u2019s documentation!", "Texture-based separation to refine building meshes", "segmentationtools", "segmentationtools"], "terms": {"overview": 0, "possibl": 0, "algorithm": [0, 1, 3], "sharp": [0, 2, 3], "pixel": [0, 3], "from": [0, 1, 2, 3, 5], "context": [0, 1, 2, 3], "segmentationtool": [0, 1, 2, 3], "cv2": [0, 2, 3, 7], "math": [0, 2, 3], "numpi": [0, 1, 2, 3, 7], "np": [0, 1, 2, 3, 7], "st": [0, 1, 2, 3], "matplotlib": [0, 1, 2, 3], "pyplot": [0, 1, 2, 3], "plt": [0, 1, 2, 3], "jupyt": [0, 1, 2, 3], "environ": [0, 1, 2, 3], "enabl": [0, 1, 2, 3], "open3d": [0, 1, 2, 3, 7], "webvisu": [0, 1, 2, 3], "info": [0, 1, 2, 3], "webrtc": [0, 1, 2, 3], "gui": [0, 1, 2, 3], "backend": [0, 1, 2, 3], "webrtcwindowsystem": [0, 1, 2, 3], "http": [0, 1, 2, 3], "handshak": [0, 1, 2, 3], "server": [0, 1, 2, 3], "disabl": [0, 1, 2, 3], "show_detected_lin": [0, 6, 7], "localfil": [0, 1, 2, 3], "2colortextur": 0, "jpg": [0, 2], "3colortextur": 0, "colorpiramidstextur": 0, "colorshadowtextur": 0, "sours": 0, "img": [0, 1, 2, 7], "imread": [0, 2, 3], "show_img": [0, 1, 2, 3, 6, 7], "true": [0, 1, 2, 3], "perform": [0, 1, 2, 3, 5], "tedect": [0, 3], "find": [0, 2, 5, 7], "gradient": [0, 2, 3], "dst": [0, 2, 3], "100": [0, 3], "300": [0, 3], "none": [0, 1, 2, 3, 7], "3": [0, 1, 2, 3, 7], "dstp": [0, 2, 3], "cvtcolor": [0, 2, 3], "color_gray2bgr": [0, 2, 3], "let": [0, 3], "appli": [0, 2, 3], "standard": [0, 3], "transform": [0, 3], "line": [0, 2, 3, 5, 7], "dstp2": [0, 2, 3], "1": [0, 1, 2, 3], "5": [0, 1, 2, 3], "pi": [0, 2, 3], "120": [0, 3], "150": [0, 2, 3], "0": [0, 1, 2, 3], "print": [0, 1, 2, 3], "below": [0, 3], "we": [0, 1, 3, 5], "displai": [0, 3, 7], "result": [0, 1, 2, 3, 5, 7], "draw": [0, 3], "i": [0, 1, 2, 3, 5], "rang": [0, 1, 2, 3], "len": [0, 1, 2, 3], "rho": [0, 3, 7], "theta": [0, 3, 7], "co": [0, 3], "b": [0, 3], "sin": [0, 3], "x0": [0, 3], "y0": [0, 3], "pt1": [0, 3], "int": [0, 3], "1000": [0, 3], "pt2": [0, 3], "255": [0, 2, 3], "line_aa": [0, 2, 3], "297": 0, "75": 0, "5235988": 0, "ar": [0, 1, 3, 5, 7], "look": [0, 3], "intersect": [0, 3, 7], "between": [0, 5, 7], "parametr": [0, 1, 5, 7], "do": 0, "thi": [0, 5], "need": [0, 1, 5], "evalu": [0, 5], "valu": [0, 1, 3, 7], "imshow": [0, 1, 2, 3], "color_bgr2rgb": 0, "get_edge_point": [0, 1, 3, 6, 7], "shape": [0, 1, 2, 3, 5], "n": [0, 1, 3, 7], "plot": [0, 3], "zip": [0, 1, 3], "color": [0, 1, 2, 3, 5, 7], "orang": [0, 1, 3], "linewidth": [0, 1, 3], "2": [0, 1, 2, 3, 7], "show": [0, 1, 2], "343": 0, "81208819": 0, "148": [0, 1], "09034035": 0, "339": [0, 1], "explor": 1, "how": [1, 5], "can": [1, 3, 5], "cut": [1, 3, 5, 7], "along": [1, 7], "preserv": 1, "map": [1, 5], "foreach": 1, "check": [1, 3, 5, 7], "point": [1, 3, 5, 7], "interpol": [1, 7], "all": [1, 3], "make": [1, 3, 5, 7], "segment": [1, 2, 7], "2d": 1, "arrai": [1, 3, 7], "x": [1, 7], "y": [1, 7], "fave": 1, "base": [1, 4], "its": 1, "coordin": [1, 3, 5, 7], "o3d": [1, 3], "meshpath": [1, 3], "obj": [1, 3], "io": [1, 3], "read_triangle_mesh": [1, 3], "detect": [1, 2, 4], "show_geometri": [1, 6, 7], "store": [1, 5], "currenti": 1, "onli": [1, 3], "work": 1, "one": [1, 2, 3, 5], "asarrai": [1, 3], "us": [1, 2, 3, 5, 7], "opencv": 1, "an": [1, 2, 5, 7], "houghlin": 1, "estim": [1, 2, 3], "straight": [1, 3, 5, 7], "function": 1, "terurn": 1, "tupl": 1, "defin": [1, 2, 3, 5, 7], "detect_edg": [1, 6, 7], "95637344": 1, "344": 1, "67813565": 1, "after": 1, "got": 1, "scale": [1, 2, 3], "dimens": [1, 3], "uvpoint": [1, 3], "triangle_uv": [1, 3], "fill": [1, 3], "facecolor": [1, 3], "edgecolor": [1, 3], "titl": [1, 7], "my": 1, "pictur": 1, "486": 1, "21696067": 1, "61": 1, "16271722": 1, "254": 1, "296": 1, "51041192": 1, "21": 1, "78303933": 1, "each": [1, 3, 5], "intersectionpoint": 1, "within": 1, "bound": [1, 3, 7], "find_triangle_intersect": [1, 3, 6, 7], "scatter": [1, 3], "295": 1, "72943893": 1, "2183794": 1, "184": 1, "26868847": 1, "exist": [1, 5], "ha": 1, "connect": [1, 3, 5], "opposit": 1, "uncut": 1, "second": [1, 2, 7], "link": 1, "last": 1, "other": [1, 7], "def": [1, 3], "cut_triangl": [1, 3, 6, 7], "edgepoint": [1, 3, 7], "index": [1, 3, 4, 5], "correspond": [1, 7], "oposit": 1, "tline0": 1, "tline1": 1, "tline2": 1, "tline": 1, "loop": 1, "over": 1, "case": [1, 3], "line_segment_intersect": [1, 6, 7], "return": [1, 7], "gener": 1, "alwai": 1, "exacli": 1, "start": 1, "wa": [1, 7], "otherp1": 1, "otherp2": 1, "newtriangle0": 1, "vstack": 1, "newtriangle1": 1, "newtriangle2": 1, "There": 1, "some": [1, 5], "where": 1, "These": 1, "still": 1, "have": [1, 3, 7], "adress": 1, "newtriangl": [1, 3], "green": 1, "blue": 1, "locat": 1, "origin": [1, 2, 3], "t": [1, 5], "relat": [1, 7], "meshpoints3d": 1, "vertic": [1, 3], "posit": 1, "59346402": 1, "925264": 1, "ax": [1, 2], "project": [1, 3, 5], "plot_trisurf": 1, "alpha": 1, "shade": 1, "fals": [1, 2, 3, 7], "mpl_toolkit": 1, "mplot3d": 1, "art3d": 1, "poly3dcollect": 1, "0x11a7b2056a0": 1, "interpolate_coordin": 1, "new2dpoint": 1, "og2dpoint1": 1, "og2dpoint2": 1, "og3dpoint1": 1, "og3dpoint2": 1, "val1": 1, "interp": [1, 3], "val2": 1, "interpolate_coordinate3d": 1, "new3dpoint": 1, "3498574106559067": 1, "3002851786881866": 1, "input": [1, 7], "coupl": [1, 7], "p1": [1, 7], "p2": [1, 7], "p1_3d": 1, "p2_3d": 1, "4": [1, 2, 3], "p3": 1, "px": [1, 7], "output": 1, "step": 1, "interp_2d": 1, "p_int_x": 1, "p_int_i": 1, "averag": [1, 7], "interp_point": [1, 6, 7], "p": 1, "append": [1, 3], "interp_valu": [1, 6, 7], "val": [1, 7], "p_int": 1, "interp_nd": 1, "interv": 1, "p_3d": 1, "import": [2, 3], "fseg": [2, 3], "time": 2, "time0": 2, "exampl": 2, "read": 2, "imag": [2, 3, 5, 7], "matterport": [2, 5], "livingroom": 2, "detailzoom": 2, "resiz": [2, 3], "512": [2, 3], "windows": [2, 3], "10": 2, "omegav": [2, 3], "filter": [2, 3], "bank": [2, 3], "convert": [2, 3, 5, 7], "rgb": [2, 3, 5], "grei": [2, 3], "filter_list": [2, 3], "log": [2, 3], "gabor": [2, 3], "filter_out": [2, 3], "image_filt": [2, 3], "includ": [2, 3], "band": [2, 3], "ig": [2, 3], "concaten": [2, 3], "float32": [2, 3], "reshap": [2, 3], "axi": [2, 3, 7], "run": [2, 3], "try": [2, 3], "differ": [2, 3, 5, 7], "window": [2, 3], "size": [2, 3], "without": [2, 3], "nonneg": [2, 3], "constraint": [2, 3], "seg_out": 2, "w": [2, 3], "segn": [2, 3], "omega": [2, 3], "nonneg_constraint": [2, 3], "2f": 2, "fig": 2, "subplot": 2, "ncol": 2, "sharex": 2, "sharei": 2, "figsiz": 2, "12": 2, "6": 2, "cmap": [2, 3], "grai": [2, 3], "tight_layout": [2, 3], "number": [2, 3], "32": 2, "canni": [2, 5], "edg": [2, 4, 7], "50": 2, "linesp": [2, 3], "houghlinesp": [2, 3], "360": [2, 3], "30": [2, 3], "l": [2, 3], "materi": [3, 5], "hough": [3, 5], "onto": [3, 5], "new": [3, 5, 7], "boundari": [3, 5], "zone": [3, 5], "region": 3, "grow": [3, 5], "packag": 3, "paramet": [3, 5, 7], "load_ext": 3, "autoreload": 3, "3d": 3, "testwal": 3, "visual": 3, "draw_geometri": 3, "mesh_show_wirefram": 3, "bw": [3, 7], "want": [3, 5, 7], "group": [3, 5], "reduc": [3, 5], "detail": [3, 5], "better": [3, 5], "factoris": [3, 5], "20": 3, "graytextur": 3, "color_bgr2grai": 3, "segmentedtextur": 3, "remap": 3, "greyscal": 3, "segmentedmappedtextur": 3, "round": 3, "min": 3, "max": [3, 7], "astyp": 3, "uint8": 3, "The": [3, 5, 7], "patch": [3, 5], "pefect": [3, 5], "hed": [3, 5], "get": [3, 5], "dtect": 3, "5707964": 3, "path": [3, 5], "continu": 3, "off": 3, "clip_on": 3, "linenr": 3, "newpoint": 3, "nx2": 3, "301": 3, "88468707": 3, "0000132": 3, "02310181": 3, "00001115": 3, "167": 3, "85111918": 3, "00000734": 3, "98969269": 3, "00000529": 3, "389": 3, "05703735": 3, "00001701": 3, "seperatli": 3, "uv_coord": 3, "vert": [3, 7], "new_vertic": 3, "new_uv": 3, "new_triangl": 3, "nrofvert": 3, "nrofofvert": 3, "points_uv": 3, "points_3d": 3, "newtriangles_3d": 3, "newtex": 3, "bwtex": 3, "png": 3, "newmesh": 3, "create_mesh": [3, 6, 7], "pcd": 3, "pointcloud": 3, "util": 3, "vector3dvector": 3, "blob": [3, 5], "color_gray2rgb": 3, "split": [3, 5], "list": [3, 7], "neighbour": [3, 5], "vertec": 3, "prepar": 3, "triangle_adjac": 3, "triangleindex": 3, "full": 3, "currentindex": 3, "nroftri": 3, "mesh_vert": 3, "mesh_tri": 3, "mesh_uv": 3, "mesh_tris_posit": 3, "zero": 3, "mesh_tris_valu": 3, "j": 3, "get_tri_pixel_valu": [3, 6, 7], "28": 3, "mark_adjac": 3, "global": 3, "compar": 3, "idx": 3, "ignor": 3, "alreadi": 3, "seglent": 3, "both": 3, "tri": [3, 7], "same": [3, 5], "newadjac": 3, "find_adjacent_triangl": [3, 6, 7], "everi": [3, 5], "give": 3, "itter": 3, "nr": 3, "long": 3, "foundal": 3, "found": 3, "trianglemesh": 3, "np_vertic": 3, "np_triangl": 3, "vector3ivector": 3, "copi": 3, "partial": 3, "first": [3, 7], "half": 3, "mesh1": 3, "deepcopi": 3, "triangle_norm": 3, "std": 3, "vector": 3, "eigen": 3, "vector3i": 3, "14": 3, "element": 3, "access": 3, "data": 3, "textur": [4, 7], "separ": 4, "refin": 4, "build": 4, "mesh": 4, "face": 4, "slice": 4, "modul": 4, "search": 4, "page": 4, "problem": 5, "effici": 5, "wai": 5, "cloud": 5, "repres": 5, "surfac": 5, "isol": 5, "indoor": 5, "scene": [5, 7], "spars": 5, "densiti": 5, "mean": 5, "part": 5, "multipl": 5, "goal": 5, "creat": 5, "model": 5, "cleanli": 5, "triangl": [5, 7], "sure": [5, 7], "belong": 5, "singl": [5, 7], "main": 5, "contribut": 5, "novel": 5, "structur": 5, "A": 5, "lot": 5, "network": 5, "normal": 5, "thei": 5, "don": 5, "directli": 5, "geometri": [5, 7], "abstract": 5, "In": 5, "most": 5, "inform": 5, "mostli": 5, "extra": 5, "difficult": 5, "repeat": 5, "distinct": 5, "localis": 5, "uv": [5, 7], "rere": 5, "becaus": 5, "lack": 5, "taxtur": 5, "like": 5, "well": 5, "clear": 5, "geometrix": 5, "bt": 5, "overdetect": 5, "rough": 5, "brick": 5, "wall": 5, "sampl": 5, "similar": 5, "them": [5, 7], "parametris": 5, "plane": 5, "explain": 5, "also": 5, "phoughlin": 5, "common": 5, "adjac": 5, "anoth": 5, "s3di": 5, "our": 5, "unwrap": 5, "good": 5, "v": 5, "ground": 5, "truth": 5, "overseg": 5, "complex": 5, "geometryless": 5, "poster": 5, "floormat": 5, "workflow": 5, "add": 5, "more": 5, "discuntinu": 5, "great": 5, "geometr": 5, "high": 5, "optimis": 5, "photograpmmetri": 5, "scan": 5, "handheld": 5, "show_detected_lines_img": [6, 7], "get_point_on_polar_lin": [6, 7], "line_intersect": [6, 7], "check_bound": [6, 7], "switchchannel": 7, "bool": 7, "str": 7, "type": 7, "arg": 7, "option": 7, "recolor": 7, "object": 7, "uniqu": 7, "default": 7, "imgpath": 7, "polar": 7, "float": 7, "leav": 7, "empti": 7, "you": 7, "provid": 7, "xmax": 7, "ymax": 7, "area": 7, "intersectiong": 7, "line1": 7, "line2": 7, "calcul": 7, "pair": 7, "xy": 7, "parallel": 7, "li": 7, "withing": 7, "test": 7, "": 7, "insid": 7, "If": 7, "diment": 7, "fraction": 7, "match": 7, "refer": 7, "given": 7, "1st": 7, "2nd": 7, "edgepoints3d": 7, "x1": 7, "y1": 7, "x2": 7, "y2": 7, "x3": 7, "y3": 7, "form": 7, "referencetri": 7, "othertri": 7}, "objects": {"": [[7, 0, 0, "-", "segmentationtools"]], "segmentationtools": [[7, 1, 1, "", "check_bounds"], [7, 1, 1, "", "create_mesh"], [7, 1, 1, "", "cut_triangle"], [7, 1, 1, "", "detect_edges"], [7, 1, 1, "", "find_adjacent_triangles"], [7, 1, 1, "", "find_triangle_intersection"], [7, 1, 1, "", "get_edge_points"], [7, 1, 1, "", "get_point_on_polar_line"], [7, 1, 1, "", "get_tri_pixel_value"], [7, 1, 1, "", "interp_point"], [7, 1, 1, "", "interp_value"], [7, 1, 1, "", "line_intersection"], [7, 1, 1, "", "line_segment_intersection"], [7, 1, 1, "", "show_detected_lines"], [7, 1, 1, "", "show_detected_lines_img"], [7, 1, 1, "", "show_geometries"], [7, 1, 1, "", "show_img"]]}, "objtypes": {"0": "py:module", "1": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "function", "Python function"]}, "titleterms": {"edg": [0, 1, 3, 5], "detect": [0, 3, 5], "complet": 0, "function": 0, "imag": [0, 1], "import": [0, 1], "canni": [0, 3], "houghlin": [0, 3], "get": [0, 1], "point": 0, "face": [1, 3, 5], "slice": [1, 3, 5], "workflow": [1, 3], "data": [1, 5], "format": 1, "mesh": [1, 3, 5], "The": 1, "textur": [1, 3, 5], "find": [1, 3], "plot": 1, "triangl": [1, 3], "uv": [1, 3], "": [1, 4], "intersect": 1, "line": 1, "creat": [1, 3], "new": 1, "3d": 1, "base": [3, 5], "separ": [3, 5], "refin": [3, 5], "build": [3, 5], "setup": 3, "segment": [3, 5], "p": 3, "plane": 3, "map": 3, "intersectionpoint": 3, "option": 3, "updat": 3, "show": 3, "geometri": 3, "object": [3, 5], "adjac": 3, "welcom": 4, "segmentationtool": [4, 6, 7], "document": 4, "inform": 4, "api": 4, "exampl": 4, "indic": 4, "tabl": 4, "introduct": 5, "background": 5, "relat": 5, "work": 5, "methodologi": 5, "experi": 5, "region": 5, "compar": 5, "method": 5, "discuss": 5, "conclus": 5}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 57}, "alltitles": {"Edge detection": [[0, "edge-detection"]], "Complete function": [[0, "complete-function"]], "Image Import": [[0, "image-import"]], "Canny Edge Detection": [[0, "canny-edge-detection"]], "HoughLines": [[0, "houghlines"]], "Get the points at the edges": [[0, "get-the-points-at-the-edges"]], "Face Slicing": [[1, "face-slicing"], [3, "face-slicing"], [5, "face-slicing"]], "Workflow": [[1, "workflow"], [3, "workflow"]], "Data Format": [[1, "data-format"]], "Import the mesh": [[1, "import-the-mesh"]], "Get The Mesh Texture": [[1, "get-the-mesh-texture"]], "Find the edges in the image": [[1, "find-the-edges-in-the-image"]], "Plot the triangle UV\u2019s": [[1, "plot-the-triangle-uv-s"]], "Finding the intersections of the triangle and the edge line": [[1, "finding-the-intersections-of-the-triangle-and-the-edge-line"]], "Creating new triangles": [[1, "creating-new-triangles"]], "Creating 3D Triangles": [[1, "creating-3d-triangles"]], "Texture-based separation to refine building meshes": [[3, "texture-based-separation-to-refine-building-meshes"], [5, "texture-based-separation-to-refine-building-meshes"]], "Setup": [[3, "setup"]], "Texture Segmentation": [[3, "texture-segmentation"], [5, "texture-segmentation"]], "Edge Detection": [[3, "edge-detection"], [5, "edge-detection"]], "Canny": [[3, "canny"]], "Houghlines P": [[3, "houghlines-p"]], "Houghlines": [[3, "houghlines"]], "UV Plane Mapping": [[3, "uv-plane-mapping"]], "Finding the intersectionPoints (Optional)": [[3, "finding-the-intersectionpoints-optional"]], "Triangle Slicing": [[3, "triangle-slicing"]], "Creating the Updated Mesh": [[3, "creating-the-updated-mesh"]], "Show the Geometry": [[3, "show-the-geometry"]], "Object Segmentation": [[3, "object-segmentation"], [5, "object-segmentation"]], "Face adjacency": [[3, "face-adjacency"]], "Welcome to the SegmentationTools\u2019s documentation!": [[4, "welcome-to-the-segmentationtools-s-documentation"]], "information": [[4, null]], "API": [[4, null]], "Examples": [[4, null]], "Indices and tables": [[4, "indices-and-tables"]], "Introduction": [[5, "introduction"]], "Background and related work": [[5, "background-and-related-work"]], "Methodology": [[5, "methodology"]], "Experiments": [[5, "experiments"]], "Data": [[5, "data"]], "Region detection and separation": [[5, "region-detection-and-separation"]], "Comparing Segmentation methods": [[5, "comparing-segmentation-methods"]], "Discussion": [[5, "discussion"]], "Conclusion": [[5, "conclusion"]], "segmentationtools": [[6, "segmentationtools"], [7, "module-segmentationtools"]]}, "indexentries": {"check_bounds() (in module segmentationtools)": [[7, "segmentationtools.check_bounds"]], "create_mesh() (in module segmentationtools)": [[7, "segmentationtools.create_mesh"]], "cut_triangle() (in module segmentationtools)": [[7, "segmentationtools.cut_triangle"]], "detect_edges() (in module segmentationtools)": [[7, "segmentationtools.detect_edges"]], "find_adjacent_triangles() (in module segmentationtools)": [[7, "segmentationtools.find_adjacent_triangles"]], "find_triangle_intersection() (in module segmentationtools)": [[7, "segmentationtools.find_triangle_intersection"]], "get_edge_points() (in module segmentationtools)": [[7, "segmentationtools.get_edge_points"]], "get_point_on_polar_line() (in module segmentationtools)": [[7, "segmentationtools.get_point_on_polar_line"]], "get_tri_pixel_value() (in module segmentationtools)": [[7, "segmentationtools.get_tri_pixel_value"]], "interp_point() (in module segmentationtools)": [[7, "segmentationtools.interp_point"]], "interp_value() (in module segmentationtools)": [[7, "segmentationtools.interp_value"]], "line_intersection() (in module segmentationtools)": [[7, "segmentationtools.line_intersection"]], "line_segment_intersection() (in module segmentationtools)": [[7, "segmentationtools.line_segment_intersection"]], "module": [[7, "module-segmentationtools"]], "segmentationtools": [[7, "module-segmentationtools"]], "show_detected_lines() (in module segmentationtools)": [[7, "segmentationtools.show_detected_lines"]], "show_detected_lines_img() (in module segmentationtools)": [[7, "segmentationtools.show_detected_lines_img"]], "show_geometries() (in module segmentationtools)": [[7, "segmentationtools.show_geometries"]], "show_img() (in module segmentationtools)": [[7, "segmentationtools.show_img"]]}})