#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkMath.h>
#include <vtkImageGradient.h>
#include <vtkImageGradientMagnitude.h>
#include <vtkImageMedian3D.h>
#include "core/macros.h"
#include "bhat/AbstractRegion.h"
#include "AttributeGenerator.h"

#define TEMPLATE_SIGN template<typename T>
#define CLASS_SIGN Bhat::AttributeGenerator<T>

typedef unsigned short ushort;
typedef unsigned char uchar;

TEMPLATE_SIGN
CLASS_SIGN::AttributeGenerator(ImagePtr inputImage, const Bhat::AbstractRegion& baseRegion, std::vector<T>& output)
    : m_InputImage(inputImage), m_BaseRegion(baseRegion), m_Output(output)
{
}

TEMPLATE_SIGN
void CLASS_SIGN::SetAttributes(Attribute x, Attribute y, Attribute z)
{
    m_Attribs[0] = x;
    m_Attribs[1] = y;
    m_Attribs[2] = z;
}

TEMPLATE_SIGN
bool CLASS_SIGN::validate_input() const
{
    bool ret = true;

    MacroAssert(m_InputImage.Get());
    if (!m_InputImage)
        ret = false;

    if (m_InputImage->GetNumberOfScalarComponents() == 1)
    {
        MacroAssert(m_InputImage->GetScalarType() == VTK_UNSIGNED_SHORT);
        if (m_InputImage->GetScalarType() != VTK_UNSIGNED_SHORT)
            ret = false;
    }
    else if (m_InputImage->GetNumberOfScalarComponents() == 3)
    {
        MacroAssert(m_InputImage->GetScalarType() == VTK_UNSIGNED_CHAR);
        if (m_InputImage->GetScalarType() != VTK_UNSIGNED_CHAR)
            ret = false;
    }
    else
        ret = false;

    ret = ret && validate_attribs();

    return ret;
}

TEMPLATE_SIGN
bool CLASS_SIGN::validate_attribs() const
{
    MacroWarning("Not implemented.");
    return false;
}

template<> bool Bhat::AttributeGenerator<float>::validate_attribs() const
{
    MacroAssert(m_Attribs[0] != AttribNone && m_Attribs[1] == AttribNone && m_Attribs[2] == AttribNone);
    return (m_Attribs[0] != AttribNone && m_Attribs[1] == AttribNone && m_Attribs[2] == AttribNone);
}
template<> bool Bhat::AttributeGenerator<float2>::validate_attribs() const
{
    MacroAssert(m_Attribs[0] != AttribNone && m_Attribs[1] != AttribNone && m_Attribs[2] == AttribNone);
    return (m_Attribs[0] != AttribNone && m_Attribs[1] != AttribNone && m_Attribs[2] == AttribNone);
}
template<> bool Bhat::AttributeGenerator<float3>::validate_attribs() const
{
    MacroAssert(m_Attribs[0] != AttribNone && m_Attribs[1] != AttribNone && m_Attribs[2] != AttribNone);
    return (m_Attribs[0] != AttribNone && m_Attribs[1] != AttribNone && m_Attribs[2] != AttribNone);
}

TEMPLATE_SIGN
void CLASS_SIGN::make_zero(T& a)
{
    MacroWarning("Not implemented.");
}

template<> void Bhat::AttributeGenerator<float>::make_zero(float& a)
{
    a = 0.0f;
}
template<> void Bhat::AttributeGenerator<float2>::make_zero(float2& a)
{
    a.x = 0.0f;
    a.y = 0.0f;
}
template<> void Bhat::AttributeGenerator<float3>::make_zero(float3& a)
{
    a.x = 0.0f;
    a.y = 0.0f;
    a.z = 0.0f;
}


TEMPLATE_SIGN
void CLASS_SIGN::Generate()
{
    m_Output.clear();
    if (!validate_input())
    {
        MacroWarning("Invalid Inputs.");
        return;
    }

    vtkIdType numOfPoints = m_InputImage->GetNumberOfPoints();
    m_Output.resize(size_t(numOfPoints));
    T zero_value;
    make_zero(zero_value);
    std::fill(m_Output.begin(), m_Output.end(), zero_value);

    std::vector<float> values;
    for (int i = 0; i < 3; ++i)
    {
        if (m_Attribs[i] == AttribNone)
            break;

        compute_values(m_Attribs[i], values);

        _ReScale(values, i);

        assign_output(m_Output, values, i);
    }
}

TEMPLATE_SIGN
void CLASS_SIGN::assign_output(std::vector<float>& output, std::vector<float>& values, int comp)
{
    output = values;
}

TEMPLATE_SIGN
void CLASS_SIGN::assign_output(std::vector<float2>& output, std::vector<float>& values, int comp)
{
    for (size_t k = 0; k < output.size(); ++k)
    {
        if (comp == 0)
            output[k].x = values[k];
        else if(comp == 1)
            output[k].y = values[k];
    }
}

TEMPLATE_SIGN
void CLASS_SIGN::assign_output(std::vector<float3>& output, std::vector<float>& values, int comp)
{
    for (size_t k = 0; k < output.size(); ++k)
    {
        if (comp == 0)
            output[k].x = values[k];
        else if(comp == 1)
            output[k].y = values[k];
        else if(comp == 2)
            output[k].z = values[k];
    }
}

TEMPLATE_SIGN
void CLASS_SIGN::compute_values(Attribute attrib, std::vector<float>& values)
{
    vtkIdType numOfPoints = m_InputImage->GetNumberOfPoints();
    values.resize(size_t(numOfPoints));
    std::fill(values.begin(), values.end(), 0.0f);

    switch (attrib)
    {
    case AttribRed:
    case AttribGreen:
    case AttribBlue:
            rgb(values, attrib);
            break;
    case AttribGrey:
            grey(values);
            break;
    case AttribCompL2:
            L2(values);
            break;
    case AttribGradMag:
            grad_mag(values);
            break;
    case AttribGradAngle:
            grad_angle(values);
            break;
    case AttribMedian3:
            median(values, 3);
    case AttribMedian5:
            median(values, 5);
            break;
    case AttribBlueByRed:
            BlueByRed(values);
            break;
    case AttribRedByBlue:
            RedByBlue(values);
            break;
    default:
            MacroWarning("Invalid Attribute type.");
            return;
    }

    /*
    // TEMP remove re-scaling of attributes since it won't work well with multi-GPU / multi-block approach.
    auto e = std::minmax_element(values.begin(), values.end());
    float values_range[] = { *e.first, *e.second };
    float req_range[] = { m_GlobalRange[0], m_GlobalRange[1] };

    std::transform(
        values.begin(),
        values.end(),
        values.begin(),
        [values_range, req_range](const auto& v)
        {
            return (v - values_range[0]) / (values_range[1]-values_range[0]) * (req_range[1] - req_range[0]) + req_range[0];
        }
    );
    */
}

TEMPLATE_SIGN
void CLASS_SIGN::rgb(std::vector<float>& values, Attribute attrib)
{
    vtkIdType numOfPoints = m_InputImage->GetNumberOfPoints();
    int stride = m_InputImage->GetNumberOfScalarComponents();
    const ushort* ushort_scalars = (const ushort*)m_InputImage->GetScalarPointer();
    const uchar* uchar_scalars = (const uchar*)m_InputImage->GetScalarPointer();

    int comp = 0;
    switch (attrib)
    {
    case AttribRed:
        comp = 0;
        break;
    case AttribGreen:
        comp = 1;
        MacroConfirm(stride >= 2);
        break;
    case AttribBlue:
        comp = 2;
        MacroConfirm(stride >= 3);
        break;
    }

    for (vtkIdType i = 0; i < numOfPoints; ++i)
    {
        if (m_InputImage->GetScalarType() == VTK_UNSIGNED_SHORT)
            values[i] = (float)ushort_scalars[i*stride + comp];
        else
            values[i] = (float)uchar_scalars[i*stride + comp];
    }
}

TEMPLATE_SIGN
void CLASS_SIGN::grey(std::vector<float>& values)
{
    vtkIdType numOfPoints = m_InputImage->GetNumberOfPoints();
    int stride = m_InputImage->GetNumberOfScalarComponents();
    if (stride == 3)
    {
        const uchar* scalars = (const uchar*)m_InputImage->GetScalarPointer();
        for (vtkIdType i = 0; i < numOfPoints; ++i)
        {
            float r = float(scalars[i*stride + 0]);
            float g = float(scalars[i*stride + 1]);
            float b = float(scalars[i*stride + 2]);
            int grey = int((11.0f*r + 16.0f*g + 5.0f*b) / 32.0f);
            vtkMath::ClampValue(grey, 0, 255);
            //values[i] = float(grey > 1? grey : 1);
            values[i] = float(grey);
        }
    }
    else
    {
        rgb(values, AttribRed);
    }
}

TEMPLATE_SIGN
void CLASS_SIGN::L2(std::vector<float>& values)
{
    vtkIdType numOfPoints = m_InputImage->GetNumberOfPoints();
    int stride = m_InputImage->GetNumberOfScalarComponents();
    const ushort* ushort_scalars = (const ushort*)m_InputImage->GetScalarPointer();
    const uchar* uchar_scalars = (const uchar*)m_InputImage->GetScalarPointer();
    const int type = m_InputImage->GetScalarType();
    for (vtkIdType i = 0; i < numOfPoints; ++i)
    {
        float sum = 0;
        for (vtkIdType comp = 0; comp < stride; ++comp)
        {
            float value = 0.0f;
            if(type == VTK_UNSIGNED_SHORT)
                value = (float)ushort_scalars[i*stride + comp];
            else
                value = (float)uchar_scalars[i*stride + comp];

            sum += value * value;
        }
        values[i] = sqrt(sum);
    }
}

TEMPLATE_SIGN
void CLASS_SIGN::grad_mag(std::vector<float>& values)
{
    vtkNew<vtkImageGradientMagnitude> grads_filter;
    grads_filter->SetInputData(m_InputImage);
    grads_filter->SetNumberOfThreads(16);
    grads_filter->Update();
    //MacroPrint(grads_filter->GetOutput()->GetScalarTypeAsString());

    auto grads = grads_filter->GetOutput();
    //ReScale(grads, m_Param.ScalarRange());
    auto gradient_scalars = grads->GetPointData()->GetScalars();
    size_t baseSize = m_BaseRegion.Size();
    for (size_t i = 0; i < baseSize; ++i)
    {
        double voxelValue = gradient_scalars->GetVariantValue(m_BaseRegion[i]).ToDouble();
        values[i] = (float)voxelValue;
    }
}

TEMPLATE_SIGN
void CLASS_SIGN::grad_angle(std::vector<float>& values)
{
    vtkNew<vtkImageGradient> grads_filter;
    grads_filter->SetInputData(m_InputImage);
    grads_filter->SetNumberOfThreads(16);
    grads_filter->Update();
    //MacroPrint(grads_filter->GetOutput()->GetScalarTypeAsString());

    auto grads = grads_filter->GetOutput();
    //ReScale(grads, m_Param.ScalarRange());
    auto gradient_scalars = grads->GetPointData()->GetScalars();
    size_t baseSize = m_BaseRegion.Size();
    for (size_t i = 0; i < baseSize; ++i)
    {
        double x = gradient_scalars->GetComponent(m_BaseRegion[i], 0);
        double y = gradient_scalars->GetComponent(m_BaseRegion[i], 1);
        double angle = atan(y / (x + 1e-15));
        angle = std::isnan(angle) ? 0 : angle;
        values[i] = (float)angle;
    }
}

TEMPLATE_SIGN
void CLASS_SIGN::median(std::vector<float>& values, int kernelSize)
{
    vtkNew<vtkImageMedian3D> filter;
    filter->SetInputData(m_InputImage);
    filter->SetNumberOfThreads(16);
    filter->Update();
    auto out = filter->GetOutput();
    //ReScale(grads, m_Param.ScalarRange());
    auto scalars = out->GetPointData()->GetScalars();
    size_t baseSize = m_BaseRegion.Size();
    for (size_t i = 0; i < baseSize; ++i)
    {
        double m = scalars->GetComponent(m_BaseRegion[i], 0);
        values[i] = (float)m;
    }
}

TEMPLATE_SIGN
void CLASS_SIGN::BlueByRed(std::vector<float>& values)
{
    vtkIdType numOfPoints = m_InputImage->GetNumberOfPoints();
    int stride = m_InputImage->GetNumberOfScalarComponents();
    const ushort* ushort_scalars = (const ushort*)m_InputImage->GetScalarPointer();
    const uchar*  uchar_scalars  = (const uchar*)m_InputImage->GetScalarPointer();
    const int type = m_InputImage->GetScalarType();
    for (vtkIdType i = 0; i < numOfPoints; ++i)
    {
        float red = 0.0f, blue = 0.0f;
        if (type == VTK_UNSIGNED_SHORT)
        {
            red = (float)ushort_scalars[i*stride + 0];
            blue = (float)ushort_scalars[i*stride + 2];
        }
        else
        {
            red = (float)uchar_scalars[i*stride + 0];
            blue = (float)uchar_scalars[i*stride + 2];
        }
        values[i] = blue / (red + 1.0f); // avoid divide by zero
    }
}

TEMPLATE_SIGN
void CLASS_SIGN::RedByBlue(std::vector<float>& values)
{
    vtkIdType numOfPoints = m_InputImage->GetNumberOfPoints();
    int stride = m_InputImage->GetNumberOfScalarComponents();
    const ushort* ushort_scalars = (const ushort*)m_InputImage->GetScalarPointer();
    const uchar*  uchar_scalars  = (const uchar*)m_InputImage->GetScalarPointer();
    const int type = m_InputImage->GetScalarType();
    for (vtkIdType i = 0; i < numOfPoints; ++i)
    {
        float red = 0.0f, blue = 0.0f;
        if (type == VTK_UNSIGNED_SHORT)
        {
            red = (float)ushort_scalars[i*stride + 0];
            blue = (float)ushort_scalars[i*stride + 2];
        }
        else
        {
            red = (float)uchar_scalars[i*stride + 0];
            blue = (float)uchar_scalars[i*stride + 2];
        }
        values[i] = red / (blue + 1.0f); // avoid divide by zero
    }
}

TEMPLATE_SIGN
void CLASS_SIGN::_ReScale(std::vector<float>& values, const int component)
{
    MacroAssert(component >= 0 && component < 3);
    float values_range[] = { m_GlobalRange[0], m_GlobalRange[1] };
    float req_range[] = { 0.0f, float(m_HistDimensions[component] - 1.0f) };

    auto extremes = std::minmax_element(values.begin(), values.end());
    std::cout << "min = " << *(extremes.first) << ", max = " << *(extremes.second) << std::endl;
    std::cout << "values_range = " << values_range[0] << " , " << values_range[1] << std::endl;
    std::cout << "req_range = " << req_range[0] << " , " << req_range[1] << std::endl;

    std::transform(
        values.begin(),
        values.end(),
        values.begin(),
        [values_range, req_range](const float v)
        {
            float ret = (v - values_range[0]) / (values_range[1]-values_range[0]) * (req_range[1] - req_range[0]) + req_range[0];
            ret = vtkMath::ClampValue(ret, req_range[0], req_range[1]);
            return ret;
        }
    );

    extremes = std::minmax_element(values.begin(), values.end());
    std::cout << "min = " << *(extremes.first) << ", max = " << *(extremes.second) << std::endl;
}

template class Bhat::AttributeGenerator<float>;
template class Bhat::AttributeGenerator<float2>;
template class Bhat::AttributeGenerator<float3>;

