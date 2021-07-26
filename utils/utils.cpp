#include <iostream>
#include <vtkDataSet.h>
#include <vtkCellData.h>
#include <vtkMath.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkPolyDataReader.h>
#include <vtkImageData.h>
#include <vtkMetaImageWriter.h>
#include <vtkNIFTIImageWriter.h>

#include <QString>

#include "vpcommon/vpVolume.h"
#include "ds/AlgoImageToVtk.h"
#include "ds/Grid.h"
#include "ds/Image.h"
#include "io/VolumeReader.h"
#include "utils.h"


void Utils::GetDimensions(vtkImageData * image, size_t dim[3])
{
    int d[3];
    image->GetDimensions(d);
    dim[0] = (size_t)d[0]; 
    dim[1] = (size_t)d[1]; 
    dim[2] = (size_t)d[2];
}

sjDS::Grid Utils::GetGrid(vtkImageData * image)
{
    size_t dim[3];
    Utils::GetDimensions(image, dim);
    return sjDS::Grid(dim);
}

void Utils::PrintArrayNames(vtkDataSet* data, const char* msg)
{
    std::cout << msg << std::endl;
    std::cout << "PointData: " << std::endl;
    if( data && data->GetPointData() )
    {
        int num = data->GetPointData()->GetNumberOfArrays();
        for( int i=0; i < num; ++i)
            std::cout << "array[" << i << "] = " << data->GetPointData()->GetArrayName(i) << std::endl;
    }

    std::cout << "CellData: " << std::endl;
    if( data && data->GetCellData() )
    {
        int num = data->GetCellData()->GetNumberOfArrays();
        for( int i=0; i < num; ++i)
            std::cout << "array[" << i << "] = " << data->GetCellData()->GetArrayName(i) << std::endl;
    }
}

void Utils::PrintOrigin(vtkImageData * data, const char* name)
{
    double origin[3];
    data->GetOrigin(origin);
    std::string str = std::string(name) + " Origin";
    Utils::PrintPoint(origin, 3, str.c_str());
}

void Utils::PrintBounds(vtkDataSet * data, const char* name)
{
    std::cout << name << " Bounds = ";
    double bounds[6];
    data->GetBounds(bounds);
    std::cout << "x = [" << bounds[0] << ", " << bounds[1] << "], ymin = [" << bounds[2]
            << ", " << bounds[3] << "], zmin = [" << bounds[4] << "," << bounds[5] << "]" << std::endl;
}

void Utils::PrintExtents(vtkImageData * data, const char* name)
{
    std::cout << name << " Extents = ";
    int extent[6];
    data->GetExtent(extent);
    std::cout << "x = [" << extent[0] << ", " << extent[1] << "], ymin = [" << extent[2]
            << ", " << extent[3] << "], zmin = [" << extent[4] << "," << extent[5] << "]" << std::endl;
}

void Utils::PrintDimensions(vtkImageData* img, const char * name)
{
    size_t dim[3] = { 0,0,0 };
    Utils::GetDimensions(img, dim);
    std::cout << name << ": dim[3] = {" << dim[0] << ", " << dim[1] << ", " << dim[2] << "}" << std::endl;
}

void Utils::PrintSpacing(vtkImageData* img, const char * name)
{
    double spacing[3] = { 0,0,0 };
    img->GetSpacing(spacing);
    std::cout << name << ": spacing[3] = {" << spacing[0] << ", " << spacing[1] << ", " << spacing[2] << "}" << std::endl;
}

void Utils::PrintBounds(const sjDS::Image& data, const char* name)
{
    std::cout << name << " Bounds = ";
    double bounds[6];
    data.GetBounds(bounds);
    std::cout << "x = [" << bounds[0] << ", " << bounds[1] << "], ymin = [" << bounds[2]
            << ", " << bounds[3] << "], zmin = [" << bounds[4] << "," << bounds[5] << "]" << std::endl;
}

template<typename T>
void Utils::ImageToVP(const sjDS::Image & image, vpVolume<T>& vp_volume, T offset)
{
    size_t array_sz = image.GetArraySize();
    std::vector<T> cprVoxelData(array_sz);
    const uint* cpr_ptr = image.GetDataPointer();

    for (size_t i = 0; i < array_sz; ++i)
        cprVoxelData[i] = (T)cpr_ptr[i]+offset;

    size_t dim[3] = { 0,0,0 };
    image.GetDimensions(dim);
    vp_volume.CopyData(cprVoxelData.data(), array_sz);
    vp_volume.SetDimensions(dim);
    double spacing[3] = { 0,0,0 };
    image.GetGrid()->GetSpacing(spacing);
    vp_volume.SetSpacing(spacing);
    vp_volume.ComputeRange();
    dim[0] = 0; dim[1] = 0; dim[2] = 0;
    vp_volume.GetDimensions(dim);
    spacing[0] = 0; spacing[1] = 0; spacing[2] = 0;
    vp_volume.GetSpacing(spacing);
}
template void Utils::ImageToVP<short>(const sjDS::Image & image, vpVolume<short>& vp_volume, short offset);


void Utils::SmoothCurve(vtkSmartPointer<vtkPoints> input_curve, double stepSize, int iterations)
{
    double p0[3], p1[3], p2[3], x[3];
    vtkNew<vtkPoints> smoothed;
    vtkNew<vtkPoints> input;
    input->DeepCopy(input_curve);
    smoothed->DeepCopy(input);
    for (int i = 0; i < iterations; ++i)
    {
        for (vtkIdType id = 1; id < input->GetNumberOfPoints() - 1; ++id)
        {
            input->GetPoint(id - 1, p0);
            input->GetPoint(id, p1);
            input->GetPoint(id + 1, p2);
            vtkMath::Add(p0, p2, x);
            vtkMath::MultiplyScalar(x, 0.5);
            vtkMath::Subtract(x, p1, x);
            vtkMath::MultiplyScalar(x, stepSize);
            vtkMath::Add(p1, x, p1);
            smoothed->SetPoint(id, p1);
        }
        input->DeepCopy(smoothed);
    }

    input_curve->DeepCopy(smoothed);
}

void Utils::WriteVolume(vtkImageData* volume, const char* filename, bool compress)
{
    QString fname(filename);
    if (fname.endsWith(".mhd"))
    {
        vtkNew<vtkMetaImageWriter> writer;
        writer->SetInputData(volume);
        writer->SetFileName(filename);
        writer->SetCompression(compress);
        writer->Write();
        if (writer->GetErrorCode() != 0)
            MacroWarning("Failed to write volume image: " << filename);
    }
    else if (fname.endsWith(".nii"))
    {
        vtkNew<vtkNIFTIImageWriter> writer;
        writer->SetInputData(volume);
        writer->SetFileName(filename);
        writer->Write();
        if (writer->GetErrorCode() != 0)
            MacroWarning("Failed to write volume image: " << filename);
    }
    else
        MacroWarning("Invalid extension. Unable to write volume data file.");
}

void Utils::WriteVolume(const sjDS::Image & volume, const char * filename)
{
    vtkSmartPointer<vtkImageData> vtk_volume = hseg::AlgoImageToVtk::Convert(volume, VTK_UNSIGNED_INT);
    WriteVolume(vtk_volume, filename);
}

void Utils::WriteVolume(const std::vector<float>& data, const sjDS::Grid& g, const std::string& filename, bool compress)
{
    int d[3];
    g.GetDimensions(d);
    auto volume = ConstructImage(d, g.Spacing(), g.Origin(), VTK_FLOAT);
    std::copy(data.begin(), data.end(), (float*)volume->GetScalarPointer());
    volume->Modified();
    WriteVolume(volume, filename.c_str(), compress);
}

void Utils::PrintScalarRange(vtkImageData* volume, const char* volume_name)
{
    double r[2] = { 0,0 };
    volume->GetScalarRange(r);
    std::cout << volume_name << " scalar_range: [" << r[0] << ", " << r[1] << "]" << std::endl;
}

void Utils::PrintScalarRange(const sjDS::Image& volume, const char * volume_name)
{
    uint r[2] = { 0,0 };
    volume.GetScalarRange(r);
    std::cout << volume_name << " scalar_range: [" << r[0] << ", " << r[1] << "]" << std::endl;
}

void Utils::Write(vtkPolyData* polydata, const char* filename, bool compress)
{
    if (polydata == nullptr || filename == nullptr)
        return;

    vtkNew<vtkPolyDataWriter> writer;
    writer->SetInputData(polydata);
    writer->SetFileName(filename);
    //writer->SetFileTypeToBinary();
    writer->SetFileTypeToASCII();
    writer->Write();
    if (writer->GetErrorCode() != 0)
    {
        MacroWarning("Failed to write polydata : " << filename);
    }
}

vtkSmartPointer<vtkPolyData> Utils::ReadPolyData(const char* filename)
{
    vtkSmartPointer<vtkPolyData> ret = nullptr;

    vtkNew<vtkPolyDataReader> reader;
    reader->SetFileName(filename);
    reader->Update();
    if (reader->GetErrorCode() != 0)
    {
        MacroWarning("Failed to read polydata : " << filename);
    }

    ret = reader->GetOutput();
    return ret;
}

vtkSmartPointer<vtkImageData> Utils::ReadVolume(const char* filename)
{
    std::string f(filename);
    VolumeReader reader(f);
    if (reader.Read())
        return reader.GetVtkOutput();
    else
        return nullptr;
}

template<typename T>
void Utils::PrintPoint(T* p, size_t sz, const char* name)
{
    std::cout << name << " = { ";
    for(size_t i=0; i < sz; ++i)
    {
        std::cout << p[i] << ", ";
    }
    std::cout << " }" << std::endl;
}

template void Utils::PrintPoint<double>(double* p, size_t sz, const char* name);
template void Utils::PrintPoint<float>(float* p, size_t sz, const char* name);
template void Utils::PrintPoint<int>(int* p, size_t sz, const char* name);

template<typename T>
static void CopyVec3(const T from[3], T to[3], size_t size)
{
    for (size_t i = 0; i < size; ++i)
        to[i] = from[i];
}

vtkSmartPointer<vtkImageData> Utils::ConstructImage(const int dim[3], const double spacing[3], const double origin[3], int vtk_scalar_type)
{
    vtkNew<vtkImageData> img;
    img->SetDimensions(dim);
    double o[] = { origin[0], origin[1], origin[2] };
    img->SetOrigin(o);
    double s[] = { spacing[0], spacing[1],  spacing[2] };
    img->SetSpacing(s);
    img->AllocateScalars(vtk_scalar_type, 1);
    return img;
}

vtkSmartPointer<vtkImageData> Utils::ConstructImage(vtkImageData* ref, int vtk_scalar_type)
{
    vtkNew<vtkImageData> img;
    img->SetDimensions(ref->GetDimensions());
    img->SetOrigin(ref->GetOrigin());
    img->SetSpacing(ref->GetSpacing());
    img->AllocateScalars(vtk_scalar_type, 1);
    return img;
}

