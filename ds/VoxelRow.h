#pragma once

namespace sjDS{
    class VoxelRow;
    typedef unsigned int voxelNo_t;
}
    

class sjDS::VoxelRow
{
public:
    VoxelRow();
    VoxelRow(voxelNo_t start, voxelNo_t end);
    bool ValidConstruction() const;
    voxelNo_t Start() const;
    voxelNo_t End() const;
    bool atEnd(voxelNo_t) const;
    bool isInside(voxelNo_t) const;
    unsigned int Length() const;

private:

    voxelNo_t m_start, m_end;

};