
#pragma once

#include <map>
#include <vector>
#include "DSObject.h"
#include "BoundingBox.h"
#include "VoxelRow.h"

class QDataStream;
namespace sjDS{
class BoundingBox;
class Grid;
typedef unsigned int segId_t;

class CompressedSegment
{
public:
    CompressedSegment() {};
    CompressedSegment(segId_t id) : m_segmentID(id) {};
    CompressedSegment(std::vector<VoxelRow>&& rows, const Grid& grid);
    //~CompressedSegment();
    //CompressedSegment( const CompressedSegment& );
    //CompressedSegment( CompressedSegment&& );
    //CompressedSegment& operator=( const CompressedSegment& ); 
    //CompressedSegment& operator=( CompressedSegment&& ); 

    void SetID(segId_t id) { m_segmentID = id;}
    segId_t Id() const { return m_segmentID; }
    void push_back(const VoxelRow& r, const Grid& grid);
    std::vector<VoxelRow>& Rows();
    const std::vector<VoxelRow>& Rows() const;
    int Read(QDataStream& in_stream);
    int Write(QDataStream& out_stream) const;
    const BoundingBox& rBoundingBox() const { return m_bb;}
    void SetBoundingBox(const BoundingBox& bb) { m_bb = bb; }

    /// Calculate the difference set between passed list of voxels and this Segment.
    /// Difference set is returned by erasing common voxels from the passed list.
    /// True is returned if intersection is found.
    /// False is returned if there is no intersection.
    bool Difference(std::list<uint>& voxels, const BoundingBox& bb ) const;

private:
    segId_t m_segmentID = UINT32_MAX;
    BoundingBox m_bb;
    std::vector<VoxelRow> m_rows;
};
}
