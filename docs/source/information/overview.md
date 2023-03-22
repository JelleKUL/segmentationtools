# Texture-based separation to refine building meshes 

## Introduction
- Meshes need to be segmented, but the sparse density of the faces means some triangles can be part of multiple segments
- How can we split the triangle on the correct place so it's not segmented in the wrong part?
- Can we use edge or boundary detection to segment the different zones of the image and create new triangles at the edges



## Background and related work

- A lot of point based segmentation
- rgb is mostly used as a extra paramter in geometry based segmentayion, meaning 
- use existing edge detection
- check out boundary detection to reduce texture detail

## Methodology

- Use Texture segmentation to seprate the different materials in the meshes
- Detect the edges on the texture
- Convert the detected edges to hough lines
- project the uv coordinates of the mesh onto the plane
- Cut the triangles with the detected lines
- Define the new faces with the edge boundaries

### Texture Segentation

- We want to grup textures and reduce their detail for better edge detection
- use Factorisation based Texture segmentation

### Edge Detection

- The patched image is pefect to detect the boundaries of the textures
- Use canny or HED edge detection
- Find the hough lines to get straight lines in the dtected edges

### Face Slicing

- slice the triangles that are in the path of the new lines
- check for continuity in adjacent faces to continue the line

### Object Segmentation

- use the texture blobs and adjacent faces region growing to segment the different objects

## Experiments

### Region detection and separation

- check how good the method is at detecting the different materials 

### Comparing Segmentation methods

- evaluate the texture based segmentation vs other mesh based segmentations

## Discussion

- did the method performed well

## Conclusion
- we created a new workflow to create more dense 