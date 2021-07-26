#include <QDataStream>
#include "Grid.h"
#include "CompressedSegmentation.h"
#include "core/macros.h"

using std::map;
using std::vector;

namespace sjDS{

CompressedSegmentation::CompressedSegmentation(const uint* seg, size_t array_size, const Grid& grid)
{
    voxelNo_t start = 0, end = 0; 
    segId_t currSegId = 0;
    std::map<segId_t,CompressedSegment> tmp_segs;
    for(size_t i=0; i < array_size; ++i)
    {
        if( start == i )
            currSegId = (segId_t)seg[i];

        if( currSegId != (segId_t)seg[i] || i == (array_size-1) )
        {
            MacroAssert(i > 0);
            end = i-1;
            VoxelRow r(start, end);

            auto fnd = tmp_segs.find(currSegId);
            if( fnd == tmp_segs.end() )
            {
                auto ib = tmp_segs.insert( std::make_pair(currSegId, CompressedSegment(currSegId)) );
                if( ib.second == false)
                {
                    MacroWarning("Unable to insert CompressedSegment in the temporary map.");
                }
                fnd = ib.first;
            }

            if( fnd != tmp_segs.end() )
                fnd->second.push_back(r, grid);

            start = i;
            currSegId = (segId_t)seg[i];
        }
    }

    m_segs.reserve( tmp_segs.size() );

    for( auto& p : tmp_segs )
        m_segs.push_back( std::move(p.second) );
}

int CompressedSegmentation::GetRegion(segId_t seg_id, vector<VoxelRow>& voxRows) const
{
    CompressedSegment key(seg_id);
    auto fnd = std::lower_bound(m_segs.begin(), m_segs.end(), key, [](auto a, auto b){ return(a.Id() < b.Id()); } );

    if( fnd != m_segs.end() && fnd->Id() == seg_id )
    {
        voxRows.insert( voxRows.end(), fnd->Rows().begin(), fnd->Rows().end() );
        return 1;
    }
    else
    {
        MacroWarning("Cannot find specified segment.");
        return 0;
    }
}

bool CompressedSegmentation::Difference(segId_t seg_id, std::list<uint>& voxels, const BoundingBox & bb) const
{
    CompressedSegment key(seg_id);
    auto fnd = std::lower_bound(m_segs.begin(), m_segs.end(), key, [](auto a, auto b){ return(a.Id() < b.Id()); } );

    if( fnd != m_segs.end() && fnd->Id() == seg_id )
        return fnd->Difference( voxels, bb);
    else
    {
        MacroWarning("Cannot find specified segment.");
        return false;
    }
}

int CompressedSegmentation::Read(QDataStream& in_stream)
{
    if( in_stream.atEnd() || in_stream.status() != QDataStream::Ok )
    {
        MacroWarning("vector<VoxelRow> read failure.");
        return 0;
    }

    size_t numOfSegs = 0;
    in_stream >> numOfSegs;
    m_segs.resize( numOfSegs );

    int retValue = 1;
    for( auto& s : m_segs )
        retValue &= s.Read(in_stream);

    return retValue;
}

int CompressedSegmentation::Write(QDataStream& out_stream) const
{    
    if( out_stream.status() != QDataStream::Ok )
    {
        MacroWarning("vector<VoxelRow> write Failure.");
        return 0;
    }

    size_t numOfSegs = m_segs.size();
    out_stream << numOfSegs;

    int retValue = 1;
    for(auto& s : m_segs)
        retValue &= s.Write(out_stream);

    return retValue;
}

}