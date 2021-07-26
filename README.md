# Volume Exploration Using Multidimensional Bhattacharyya Flow
---
## Project Description
We present a novel approach for volume exploration that is versatile yet effective in isolating semantic structures in both noisyand clean data. Specifically, we have implemented a hierarchical active contours approach based on Bhattacharyya gradient flow which is easier to control, robust to noise, and can incorporate various types of statistical information to drive an edge-agnostic exploration process.

---
## Dependencies to Build
Build using CMake 3.14 or higher, and Visual Studio Community 2017 or higher. Tested on Windows 10 64-bit.  

[VTK 9.0.1](https://gitlab.kitware.com/vtk/vtk/-/tree/v9.0.3)  
[ITK 5.0.1](https://github.com/InsightSoftwareConsortium/ITK/tree/v5.0.1)  
[Qt 5.15](https://www.qt.io/blog/qt-5.15-released)  
CUDA 10.2  
[libglm 0.9.8.4](https://github.com/g-truc/glm)  
libglew  
[LIBICS C++ library](https://svi-opensource.github.io/libics)  

---
## Citation
Shreeraj Jadhav, Mahsa Torkaman, Allen Tannenbaum, Saad Nadeem, Arie E Kaufman. "Volume Exploration Using MultidimensionalBhattacharyya Flow", IEEE Transactions on Visualization and Computer Graphics, 2021.

O. Michailovich, Y. Rathi, and A. Tannenbaum, “Image segmentationusing active contours driven by the Bhattacharyya gradient flow,” IEEE Transactions on Image Processing, vol. 16, pp. 2787–2801, 2007.
