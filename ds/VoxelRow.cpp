#include "VoxelRow.h"
#include "core/macros.h"

typedef unsigned int uint;

using sjDS::VoxelRow;
using sjDS::voxelNo_t;

VoxelRow::VoxelRow()
{
    m_start = m_end = UINT32_MAX;
}

VoxelRow::VoxelRow(voxelNo_t start, voxelNo_t end)
{
    if(start > end)
        MacroWarning("Invalid specification for construction.");

    m_start = start;
    m_end   = end;
}

bool VoxelRow::ValidConstruction() const
{
    if( m_start == m_end && m_start == UINT32_MAX )
        return false;
    else
        return true;
}

voxelNo_t VoxelRow::Start() const
{
    return m_start;
}

voxelNo_t VoxelRow::End() const
{
    return m_end;
}

bool VoxelRow::atEnd(voxelNo_t pos) const
{
    return (pos > m_end);
}

bool sjDS::VoxelRow::isInside(voxelNo_t v) const
{
    return (m_start <= v && v <= m_end);
}

uint sjDS::VoxelRow::Length() const
{
    return size_t(m_end - m_start + 1);
}
