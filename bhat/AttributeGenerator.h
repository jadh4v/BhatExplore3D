#pragma once
#include <vector>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>

namespace Bhat {
    class AbstractRegion;
}

typedef vtkSmartPointer<vtkImageData> ImagePtr;
typedef unsigned int uint;

namespace Bhat {

    template<typename T>
    class AttributeGenerator
    {
    public:
        enum Attribute{ AttribNone = 0, AttribGrey, AttribGradMag, AttribRed, AttribGreen, AttribBlue, AttribGradAngle, AttribCompL2, AttribMedian3, AttribMedian5, AttribBlueByRed, AttribRedByBlue };
        AttributeGenerator(ImagePtr inputImage, const Bhat::AbstractRegion& baseRegion, std::vector<T>& output);
        void SetAttributes(Attribute x, Attribute y=AttribNone, Attribute z=AttribNone);
        void SetGlobalRange(double minValue, double maxValue) { m_GlobalRange[0] = minValue; m_GlobalRange[1] = maxValue; }
        void SetHistDimensions(uint x, uint y, uint z) { m_HistDimensions[0] = x; m_HistDimensions[1] = y; m_HistDimensions[2] = z; }
        void Generate();
        bool OutputTypeFloat()  const { return sizeof(T) == sizeof(float);  }
        bool OutputTypeFloat2() const { return sizeof(T) == sizeof(float2); }
        bool OutputTypeFloat3() const { return sizeof(T) == sizeof(float3); }

    private:
        bool validate_input() const;
        bool validate_attribs() const;
        static void assign_output(std::vector<float>& output, std::vector<float>& values, int comp);
        static void assign_output(std::vector<float2>& output, std::vector<float>& values, int comp);
        static void assign_output(std::vector<float3>& output, std::vector<float>& values, int comp);
        static void make_zero(T& a);
        void compute_values(Attribute attrib, std::vector<float>& values);
        void rgb(std::vector<float>& values, Attribute attrib);
        void grey(std::vector<float>& values);
        void L2(std::vector<float>& values);
        void grad_mag(std::vector<float>& values);
        void grad_angle(std::vector<float>& values);
        void median(std::vector<float>& values, int kernelSize);
        void BlueByRed(std::vector<float>& values);
        void RedByBlue(std::vector<float>& values);
        void _ReScale(std::vector<float>& values, const int component);

        Attribute m_Attribs[3] = { AttribNone, AttribNone, AttribNone };
        ImagePtr m_InputImage = nullptr;
        const Bhat::AbstractRegion& m_BaseRegion;
        std::vector<T>& m_Output;
        float m_GlobalRange[2] = { 0.0f, 255.0f };
        uint m_HistDimensions[3] = { 32, 32, 32 };
    };

}
