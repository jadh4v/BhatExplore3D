#include <QDataStream>
//#include "ds/ddsbase.h"
#include "PVMReader.h"
#include "ds/Grid.h"
#include "ds/Image.h"
#include "core/CoreTypedefs.h"

using sjDS::Grid;
using sjDS::Image;
using hseg::PVMReader;
using std::cout;
using std::endl;

void PVMReader::init()
{
}

PVMReader::~PVMReader()
{
}

PVMReader::PVMReader()
{
}

PVMReader::PVMReader(const QString& filename)
{
    m_filename = filename;
    m_file.setFileName(filename);
}

void PVMReader::SetFileName(const QString& filename)
{
    m_filename = filename;
    m_file.setFileName(filename);
}

int PVMReader::input_validation() const
{
    if( m_file.fileName().isEmpty() )
        return 0;

    if( m_filename.isEmpty() )
        return 0;

    return 1;
}

int PVMReader::primary_process()
{
    int retValue = 1;

    unsigned int dim[3] = {0,0,0};
    unsigned int comp = 0;
    float scale[3] = {0,0,0};
    unsigned char* data = nullptr;
    // data = readPVM();
    cout << "dim   = { " << dim[0] << ", " << dim[1] << ", " << dim[2] << " }" << endl;
    cout << "scale = { " << scale[0] << ", " << scale[1] << ", " << scale[2] << " }" << endl;
    cout << "comp  = " << comp << endl;
    if( retValue == 1 && data != nullptr )
    {

        Grid grid;
        size_t d[3] = { (size_t)dim[0], (size_t)dim[1], size_t(dim[2]) };
        grid.SetDimensions(d);
        double spacing[3] = { (double)scale[0], (double)scale[1], (double)scale[2] };
        grid.SetSpacing( spacing );
        double origin[3] = {0,0,0};
        grid.SetOrigin( origin );

        m_img = Image( &grid );
        size_t arraySize = grid.GetArraySize();

        unsigned short* uShortArray = nullptr;
        unsigned char*  uCharArray  = nullptr;

        if(comp == 1)
            uCharArray  = (unsigned char*) data;
        else if(comp == 2)
            uShortArray = (unsigned short*) data;
        else
        {
            MacroWarning("Unable to handle data-type.");
            retValue = 0;
        }

        for( size_t i=0; i < arraySize; ++i )
        {
            type_uint value = 0;
            if( comp == 2)
                value = (type_uint)uShortArray[i];
            else
                value = (type_uint)uCharArray[i];

            m_img.SetVoxel( type_uid(i), value );
        }
    }
    else
        retValue = 0;

    free(data);
    return retValue;
}

int PVMReader::post_processing()
{
    return 1;
}
