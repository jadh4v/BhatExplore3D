#pragma once
#include <vector>

class vtkDataSet;
class vtkPoints;
class vtkImageData;
class vtkPolyData;
template <class T> class vpVolume;
namespace sjDS {
    class Grid;
    class Image;
}


class Utils {
public:
    /*
    struct Grid
    {
    public:
        int m_dim[3] = { 0,0,0 };
        double m_spacing[3] = { 0,0,0 };
        double m_origin[3] = { 0,0,0 };
        Grid(const int d[3])
        {
            m_dim[0] = d[0]; m_dim[1] = d[1]; m_dim[2] = d[2];
        }
        Grid(const int d[3], const double s[3], const double o[3])
        {
            for (int i = 0; i < 3; ++i)
            {
                m_dim[i] = d[i];
                m_spacing[i] = s[i];
                m_origin[i] = o[i];
            }
        }
    };*/

    static void GetDimensions(vtkImageData* image, size_t dim[3]);
    static sjDS::Grid GetGrid(vtkImageData* image);


    // Curve functions
    static void SmoothCurve(vtkSmartPointer<vtkPoints> input_curve, double stepSize, int iterations);

    // Print functions
    template<typename T>
    static void PrintPoint(T* p, size_t sz, const char* name = nullptr);
    static void PrintArrayNames(vtkDataSet* data, const char* msg);
    static void PrintOrigin(vtkImageData * data, const char* name);
    static void PrintBounds(vtkDataSet* data, const char* name);
    static void PrintDimensions(vtkImageData* img, const char* name);
    static void PrintBounds(const sjDS::Image& data, const char* name);
    static void PrintExtents(vtkImageData * data, const char* name);
    static void PrintSpacing(vtkImageData* img, const char * name);
    static void PrintScalarRange(vtkImageData* volume, const char* volume_name);
    static void PrintScalarRange(const sjDS::Image& volume, const char* volume_name);
    static void WriteVolume(vtkImageData* volume, const char* filename, bool compress=false);
    static void WriteVolume(const sjDS::Image& volume, const char* filename);
    static void WriteVolume(const std::vector<float>& data, const sjDS::Grid& g, const std::string& filename, bool compress = false);
    static void Write(vtkPolyData* polydata, const char* filename, bool compress=false);
    static vtkSmartPointer<vtkPolyData> ReadPolyData(const char* filename);
    static vtkSmartPointer<vtkImageData> ReadVolume(const char* filename);
    static vtkSmartPointer<vtkImageData> ConstructImage(const int dim[3], const double spacing[3], const double origin[3], int vtk_scalar_type);
    static vtkSmartPointer<vtkImageData> ConstructImage(vtkImageData* ref, int vtk_scalar_type);

    template <typename T> 
    static void ImageToVP(const sjDS::Image& image, vpVolume<T>& vp_volume, T offset);

    template<typename T>
    static void CopyVec3(const T from[3], T to[3], size_t size);
};
