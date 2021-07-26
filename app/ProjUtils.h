#pragma once
#include <string>

class vtkImageData;

class ProjUtils
{
public:
    class Print {
    public:
        static void ComponentCount(vtkImageData* img, const std::string& id="");
        static void Dimensions(vtkImageData* img, const std::string& id="");
        static void Range(vtkImageData* img, const std::string& id="");
    };

};