#pragma once
// ONLY DECLARATIONS OF GLOBALS
// please define them in vpapp_globals.cpp

#include <iostream>
#include <map>
#include <QString>
#include <vtkColor.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkImageData.h>
//#include <vpStructures.h>

extern QString openedPath;
extern std::fstream gLog;

struct globalParameters
{
    int SliderLength = 100;
    int thread_cnt = 12;
    int Dimensions[3] = { 0,0,0 };
    double Origin[3] = { 0,0,0 };
    double Bounds[6] = { 0,0,0,0,0,0 };
    QString openedPath;

    vtkColor4d pancreasColor = vtkColor4d(170.0/255.0, 226.0/255.0, 170.0/255.0, 0.3);
    vtkColor4d cystColor     = vtkColor4d(255.0/255.0, 112.0/255.0, 112.0/255.0, 0.3);
    vtkColor4d ductColor     = vtkColor4d(0.5, 0.5, 1.0, 1.0);
    vtkColor4d clineColor    = vtkColor4d(1.0, 0.3, 0.3, 1.0);

    struct
    {
        vtkColor4d Color = vtkColor4d(0.1, 0.1, 0.1, 1.0);
        double Width = 3.0;
    }
    ExtentLines;

    struct
    {
        vtkColor4d XColor = vtkColor4d(0.1, 0.1, 0.1, 1.0);
        vtkColor4d YColor = vtkColor4d(0.1, 0.1, 0.1, 1.0);
        vtkColor4d ZColor = vtkColor4d(0.1, 0.1, 0.1, 1.0);
        double Length = 0.1;
        double LineWidth = 3.0;
    }SelectionPoint;

    struct {
        int Gender = 0;
        int Age = 35;
    }Patient;

};// gParam;

extern struct globalParameters gParam;

struct globalObjects{
    std::map<std::string, std::pair<int, int>> patientInfo;
    vtkSmartPointer<vtkImageData> CT_Data = nullptr;
    struct {
        vtkSmartPointer<vtkImageData> Volume = nullptr;
        vtkSmartPointer<vtkImageData> Features = nullptr;
        vtkSmartPointer<vtkImageData> Smooth = nullptr;
        vtkSmartPointer<vtkPolyData> Surface = nullptr;
        unsigned int Features_tex = 0;
        float Box[6] = { 0,0,0,0,0,0 };
    }Cyst;
    vtkSmartPointer<vtkPolyData> PancreasSurface = nullptr;
    vtkSmartPointer<vtkPolyData> DuctSurface = nullptr;
    vtkSmartPointer<vtkPolyData> PancreasCenterline = nullptr;
    vtkSmartPointer<vtkPolyData> CPRSurface = nullptr;
};

extern struct globalObjects gObjects;

