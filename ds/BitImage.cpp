#include "BitImage.h"

namespace sjDS{

typedef unsigned char uchar;

BitImage::BitImage()
{
}

BitImage::BitImage(const Grid & grid)
{
    Resize(grid);
}

int BitImage::Resize(const Grid & grid)
{
    m_LogicalGrid = grid;
    if (_AllocateMemory() == 0)
    {
        MacroWarning("Unable to allocate memory.");
        SetInvalidConstruction();
    }
    return 0;
}

int BitImage::Set(const size_t gridPoint[3])
{
    NullCheck(gridPoint, 0);
    if( !_ValidateIndex(gridPoint) )
        return 0;

    size_t bitPosition = (size_t) _CalcVoxelID(gridPoint);
    m_BitData.Set( bitPosition );
    return 1;
}

int BitImage::Clear(const size_t gridPoint[3])
{
    NullCheck(gridPoint, 0);
    if( !_ValidateIndex(gridPoint) )
        return 0;

    size_t bitPosition = (size_t) _CalcVoxelID(gridPoint);
    m_BitData.Clear( bitPosition );
    return 1;
}

int BitImage::Get(const size_t gridPoint[3]) const
{
    NullCheck(gridPoint, -1);
    if( !_ValidateIndex(gridPoint) )
        return -1;

    size_t bitPosition = (size_t) _CalcVoxelID(gridPoint);
    if (m_BitData.Get(bitPosition))
        return 1;
    else
        return 0;
}

/*
int BitImage::Get(size_t voxelId) const
{
    if( voxelId >= m_LogicalGrid.GetArraySize() || voxelId >= m_BitData.ArraySize() )
        return -1;

    if (m_BitData.Get(voxelId))
        return 1;
    else
        return 0;
}*/

const void * BitImage::GetRawPointer() const
{
    return m_BitData.GetRawPointer();
}

const void* BitImage::GetRawPointerToSlice(size_t zSliceNumber) const
{
    uchar* ptr = (uchar*)m_BitData.GetRawPointer();
    size_t offset = (m_PhysicalGrid.x() / 8) * m_PhysicalGrid.y() * zSliceNumber;
    return (const void*)(&(ptr[offset]));
}

void BitImage::GetDimensions(size_t dim[3]) const
{
    m_LogicalGrid.GetDimensions(dim);
}

void BitImage::GetByteDimensions(size_t dim[3]) const
{
    dim[0] = m_PhysicalGrid.x() / 8;
    dim[1] = m_PhysicalGrid.y();
    dim[2] = m_PhysicalGrid.z();
}

int BitImage::ClearAll()
{
    m_BitData.ClearBits();
    return 1;
}

int BitImage::_AllocateMemory()
{
    if (m_LogicalGrid.GetArraySize() == 0)
    {
        MacroWarning("Invalid grid object.");
        return 0;
    }

    // Calculate the dimensions of the grid in bytes.
    MacroAssert(m_LogicalGrid.Dim() != nullptr);

    size_t phys_dim[3] = { 0,0,0 };
    size_t r = m_LogicalGrid.x() % 8;   // remainder bits beyond divisible by 8
    phys_dim[0] = m_LogicalGrid.x() + 8 - (r == 0 ? 8 : r);
    phys_dim[1] = m_LogicalGrid.y();
    phys_dim[2] = m_LogicalGrid.z();

    // Update the physical Grid object to represent the actual grid in memory (based on padding).
    m_PhysicalGrid.SetDimensions(phys_dim);
    MacroAssert(m_PhysicalGrid.x() % 8 == 0);

    // Calculate total number of bits required for storage in bitvector.
    size_t data_array_size = phys_dim[0] * phys_dim[1] * phys_dim[2];


    if (data_array_size <= 1)
        return 0;

    m_BitData.Resize(data_array_size);

    return 1;
}

bool BitImage::_ValidateIndex(const size_t ijk[3]) const
{
    return m_LogicalGrid.ValidateIndex(ijk);
}

size_t BitImage::_CalcVoxelID(const size_t ijk[3]) const
{
    return m_PhysicalGrid.CalcVoxelID(ijk);
}

}