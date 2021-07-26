#pragma once

#include <map>
#include <list>
#include <vector>
#include "DSObject.h"
#include "VoxelRow.h"
#include "CompressedSegment.h"


namespace sjDS{
    class Grid;
    class BoundingBox;

class CompressedSegmentation : public DSObject
{
    typedef unsigned int uint;
public:
    CompressedSegmentation() {} 
    CompressedSegmentation(const uint* seg, size_t array_size, const Grid& grid);
    int GetRegion(segId_t seg_id, std::vector<VoxelRow>& voxRows ) const;
    bool Difference(segId_t seg_id, std::list<uint>& voxels, const BoundingBox& bb ) const;
    size_t GetSize() const { return m_segs.size(); }
    CompressedSegment& rSeg(size_t i) { return m_segs[i]; }

    // Streaming API
    int Read(QDataStream& in_stream);
    int Write(QDataStream& out_stream) const;

private:
    std::vector<sjDS::CompressedSegment> m_segs;

};

}