---
marp: true
headingDivider: 4
---

# Segmentation Development
Exploring the possible paths of the segmentation model


## End Goal
*Dynamic Reality Modeling*

Create interactable objects from a static mesh
- Completion of occluded parts
- Full coverage of the scene
- Generic primitive interaction

![bg right w:500](localfiles/MeshCompletion%20Icon.png)

## Possible Approaches
- Global
  - Voxel clustering
  - Region growing
  - Primitive fitting
- Local
  - hierarchical face clustering
  - Normal seperation / fold detection
  - 3d feature based clustering

### Voxel Clustering
Subsampling the space into voxels and combine them into joined shapes depending on their proximity and convex joinability. Aimed at creating rectangular shape cages

### Primitive fitting
Primitives can form a great basis for the generic shape segmentation, however, most of the soa is aimed at planes, spheres and cylinders, we also need prisms -> rectangles. Also helpful for the object completion

### Hierarchical Clustering
*Hierarchical mesh segmentation based on fitting primitives*

- Genrates a binary tree of clusters
- allows dynamic selection of the number of clusters
- edge cutting
- decimation of the mesh 
- converting to implicit geometry as end goal for interaction

## Texture integration
Textures can provide extra detail and clarify seperations in places of small geometric changes. 
> Problem: object edges are not always on polygon edges
- Use vertex colors as extra features in the descision making
- Create face cuts on texture edges to create texture based polygon edges.
- compare cosine distance


## Gameplan
Set up a modular workflow with interchangable methods.
The first step is defining relations between the neighbouring faces/points
The next step is determining the treshold for clusters

- Face/vertex relations
  - feature based
  - curvature based
  - region based
- Face/vertex grouping
  - Feature treshold
  - Primitive fitting

## Key Points
- Focus on hierachical segmentation to allow interaction on multiple scales
- get greater geometric detail from the texture cuts
- Use the segmentation method as a stepping stone to the completion

## todo
- meshnetwerk
- clustering van de faces, via locale en buren eigenschappen
- weinig kans op intersectie tussen objecten
- decimation + subdivision
- spelen met primitivenet
- partoon detectie (vb. baksteen)
- uv mapping chunk seperation