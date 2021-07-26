#include <QDataStream>
#include "Grid.h"
#include "CompressedSegment.h"
#include "core/macros.h"

using std::vector;

namespace sjDS{

CompressedSegment::CompressedSegment(std::vector<VoxelRow>&& rows, const Grid& grid)
{
    m_rows = rows;
    for(auto& r : m_rows)
    {
        size_t ijk[3] = {0,0,0};

        grid.CalcGridIndex(r.Start(), ijk);
        m_bb.Expand( ijk );
        grid.CalcGridIndex(r.End(), ijk);
        m_bb.Expand( ijk );
    }
}

void sjDS::CompressedSegment::push_back(const VoxelRow& r, const Grid& grid)
{
    m_rows.push_back(r);

    size_t ijk[3] = {0,0,0};

    grid.CalcGridIndex(r.Start(), ijk);
    m_bb.Expand( ijk );
    grid.CalcGridIndex(r.End(), ijk);
    m_bb.Expand( ijk );
}

vector<VoxelRow>& sjDS::CompressedSegment::Rows()
{
    return m_rows;
}

const vector<VoxelRow>& sjDS::CompressedSegment::Rows() const
{
    return m_rows;
}

int CompressedSegment::Read(QDataStream& in_stream)
{
    if( in_stream.atEnd() || in_stream.status() != QDataStream::Ok )
    {
        MacroWarning("vector<VoxelRow> Read Failure.");
        return 0;
    }

    size_t numOfVoxelRows = 0;
    in_stream >> m_segmentID;

    size_t box[6]={0,0,0,0,0,0};
    for( auto& x : box )
        in_stream >> x;

    m_bb = BoundingBox();
    m_bb.Expand( &box[0] );
    m_bb.Expand( &box[3] );

    in_stream >> numOfVoxelRows;
    m_rows.resize( numOfVoxelRows );
    int bytesRead = in_stream.readRawData((char*)(&m_rows[0]), numOfVoxelRows*sizeof(VoxelRow) );

    if( bytesRead == numOfVoxelRows*sizeof(VoxelRow) )
        return 1;
    else
    {
        MacroWarning("vector<VoxelRow> Read Failure.");
        return 0;
    }
}
int CompressedSegment::Write(QDataStream& out_stream) const
{
    if( out_stream.status() != QDataStream::Ok )
    {
        MacroWarning("vector<VoxelRow> write Failure.");
        return 0;
    }

    out_stream << m_segmentID;

    size_t box[6]={0,0,0,0,0,0};
    m_bb.GetBox(box);
    for( auto x : box )
        out_stream << x;

    out_stream << m_rows.size();
    int writtenBytes = out_stream.writeRawData((const char*)&m_rows[0], m_rows.size()*sizeof(VoxelRow));
    if( writtenBytes != m_rows.size()*sizeof(VoxelRow) )
    {
        MacroWarning("vector<VoxelRow> write Failure.");
        return 0;
    }
    else
        return 1;
}

bool sjDS::CompressedSegment::Difference(std::list<uint>& voxels, const BoundingBox & bb) const
{
    bool intersects = false;
    if( m_bb.Intersects(bb) )
    {
        auto vox_iter = voxels.begin();
        auto row_iter = m_rows.begin();

        while( vox_iter != voxels.end() && row_iter != m_rows.end() )
        {
            // Iterate through the voxels to find the first voxel that is geater than start of first row.
            if( *vox_iter < row_iter->Start() )
            {
                ++vox_iter;
                continue;
            }

            // Iterate through the rows to find the first row whose End() is not less than the current voxel.
            if( row_iter->End() < *vox_iter )
            {
                ++row_iter;
                continue;
            }

            if( row_iter->isInside(*vox_iter) )
            {
                intersects = true;
                voxels.erase(vox_iter++);
                continue;
            }

            ++vox_iter;
        }
    }

    return intersects;
}

}