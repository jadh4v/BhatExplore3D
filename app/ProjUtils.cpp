#include <vtkImageData.h>
#include "core/macros.h"
#include "ProjUtils.h"

void ProjUtils::Print::ComponentCount(vtkImageData * img, const std::string& id)
{
    NullCheckVoid(img);
    int c = img->GetNumberOfScalarComponents();
    std::cout << id << ":" << " Num of components = " << c << std::endl;
}

void ProjUtils::Print::Dimensions(vtkImageData * img, const std::string& id)
{
    NullCheckVoid(img);
    int dim[] = { 0,0,0 };
    img->GetDimensions(dim);
    std::cout << id << ": dim[] = {" << dim[0] << ", " << dim[1] << ", " << dim[2] << "}" << std::endl;
}

void ProjUtils::Print::Range(vtkImageData* img, const std::string& id)
{
    NullCheckVoid(img);
    double r[2];
    img->GetScalarRange(r);
    std::cout << id << ":"<< " range2[] = (" << r[0] << ", " << r[1] << ")" << std::endl;
}

