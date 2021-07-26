// VTK
#include "vtkMetaImageWriter.h"
#include "vtkImageData.h"
#include "vtkSmartPointer.h"

// Proj
#include "VolumeWriter.h"
#include "ds/Image.h"
#include "ds/Grid.h"

using sjDS::Image;
using sjDS::Grid;

VolumeWriter::VolumeWriter(const Image * img)
{
    m_input = img;
}

void VolumeWriter::SetFileName(const QString filename)
{
    m_filename = filename;
}

int VolumeWriter::Write()
{
    if (!m_input)
    { 
        MacroWarning("Input volume not set.");
        return 0;
    }

    if(m_filename.isEmpty())
    { 
        MacroWarning("Input path/filename not specified.");
        return 0;
    }

    MacroNewVtkObject(vtkImageData, vtk_img);

    double s[3] = { 0,0,0 };
    const double* spacing = m_input->GetGrid()->Spacing();
    for (int i = 0; i < 3; ++i)
        s[i] = spacing[i];

    vtk_img->SetSpacing(s);

    size_t dim1[3];
    m_input->GetDimensions(dim1);
    vtk_img->SetDimensions((int)dim1[0], (int)dim1[1], (int)dim1[2]);
    vtk_img->SetExtent(0, (int)dim1[0] - 1, 0, (int)dim1[1] - 1, 0, (int)dim1[2] - 1);
    vtk_img->AllocateScalars(VTK_UNSIGNED_INT, 1);
    uint* ptr = (uint*)vtk_img->GetScalarPointer();
    memcpy(ptr, m_input->GetDataPointer(), sizeof(uint)*m_input->GetArraySize());

    MacroNewVtkObject( vtkMetaImageWriter, writer );
    QString outfilename = m_filename;
    outfilename.truncate(outfilename.lastIndexOf('.'));
    outfilename = outfilename + ".mhd";
    writer->SetFileName(outfilename.toLatin1().constData());
    writer->SetFileDimensionality(3);
    writer->SetInputData( vtk_img );
    writer->SetCompression(false);
    writer->Update();
    writer->Write();
    MacroAssert((writer->GetErrorCode() == 0));
    return 1;
}
