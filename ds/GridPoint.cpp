#include <climits>
#include <algorithm>
#include "Grid.h"
#include "GridPoint.h"

namespace sjDS{

typedef uint32_t type_uid;
const type_uid GridPoint::cInvalidID = UINT32_MAX;
const type_uid GridPoint::cOrthoDisp[6] = {4,10,12,14,16,22};

//==================================================================================================
void GridPoint::init()
{
    m_grid = NULL;
    m_id = GridPoint::cInvalidID;
    m_dim[0] = m_dim[1] = m_dim[2] = 0;
    m_nei_mode = AllNeighbors;
    m_nei_iteration = false;
    m_std_nei_cnt = 0;
}

//==================================================================================================
GridPoint::~GridPoint()
{
}

//==================================================================================================
GridPoint::GridPoint(type_uid id, const Grid* grid)
{
    init();
    m_id = id;
    m_grid = grid;

    m_dim[0] = (int)m_grid->x();
    m_dim[1] = (int)m_grid->y();
    m_dim[2] = (int)m_grid->z();
    m_std_nei_cnt = m_grid->is3D() ? 27 : 9;

    if( !grid )
        SetInvalidConstruction();
}

//==================================================================================================
// Orthogonal neighbor is identified as the voxel
// whose coordinates are exactly 1 manhattan distance away.
bool GridPoint::is_orthogonal_neighbor(type_uid nei_id) const
{
    // get coordinates of "this" grid point
    size_t this_ijk[3] = {0,0,0};
    m_grid->CalcGridIndex(m_id, this_ijk);

    // get coordinates of neighbor
    size_t nei_ijk[3] = {0,0,0};
    m_grid->CalcGridIndex(nei_id, nei_ijk);

    size_t manhattan_distance = 0;
    for(int i=0; i < 3; ++i)
        manhattan_distance += std::max(this_ijk[i],nei_ijk[i]) - std::min(this_ijk[i], nei_ijk[i]);

    return ( manhattan_distance == 1 );
}

//==================================================================================================
void GridPoint::StartNeighborIteration()
{
    if( !ValidConstruction() )
    {
        MacroWarning("Invalid GridPoint.");
        return;
    }

    // Get the current grid point coordinates (ijk) for current grid point id (m_id)
    size_t ijk[3]={0,0,0};
    m_grid->CalcGridIndex( m_id, ijk );

    // typecast current grid point coordinates to int64_t
    for(int i=0; i < 3; ++i)
        m_nei_start_ijk[i] = (int64_t)ijk[i];

    int numOfDims = m_grid->is3D()? 3 : 2;

    // find the starting neighbor position (without bounds test)
    for(int i=0; i < numOfDims; ++i)
        m_nei_start_ijk[i] = m_nei_start_ijk[i]-1;

    m_nei_disp = 0;

    m_nei_iteration = true;
}

//==================================================================================================
type_uid GridPoint::GetNextNeighborID()
{
    // initialize return value variable
    type_uid ret = GridPoint::cInvalidID;

    if( ! m_nei_iteration )
    {
        MacroWarning("Neighbor iteration not initialized, or expired.");
        return ret;
    }

    // Calculate xy-plane size in voxels.
    const int plane_sz = m_dim[0] * m_dim[1];

    // ensure the displacement variable is within the expected number
    // of neighbors
    if( m_nei_disp >= m_std_nei_cnt )
        return ret;

    // validate current index. return if true, increment if false.
    do
    {
        // calculate displacement values in each dimension
        int d[3] = {0,0,0};
        d[0] = m_nei_disp % 3;
        d[1] = (m_nei_disp % 9) / 3;
        d[2] = m_nei_disp / 9;

        int pos[3]={0,0,0};
        for(int i=0; i<3; ++i)
            pos[i] = m_nei_start_ijk[i] + d[i];

        m_nei_disp++;
        //if( m_grid->ValidateIndex( pos ) )
        if( pos[0] >= 0 && pos[1] >= 0 && pos[2] >= 0 && pos[0] < m_dim[0] && pos[1] < m_dim[1] && pos[2] < m_dim[2] )
        {
            //ret = m_grid->CalcVoxelID( (size_t)pos[0],(size_t)pos[1],(size_t)pos[2] );
            ret = static_cast<type_uid>( plane_sz*pos[2] + m_dim[0]*pos[1] + pos[0] );

            // don't return voxel id that is the same as current node. We want neighbors.
            if(ret == m_id)
            {
                ret = GridPoint::cInvalidID; // in case this is the last neighbor.
                continue;
            }

            // Skip non-orthogonal neighbors if mode == OrthogonalNeighbors
            if( m_nei_mode == OrthogonalNeighbors && !is_orthogonal_neighbor(ret) )
            //if( m_nei_mode == OrthogonalNeighbors && std::find( cOrthoDisp, &cOrthoDisp[6], m_nei_disp ) == &cOrthoDisp[6] )
            {
                ret = GridPoint::cInvalidID; // in case this is the last neighbor.
                continue;
            }

            // break the loop if a valid neighbor is found.
            break;
        }
    }
    while(m_nei_disp < m_std_nei_cnt); // search until next valid neighbor.

    // If iteration over neighbors has completed, set the neighbor iteration flag back to false.
    if( ret == GridPoint::cInvalidID )
        m_nei_iteration = false;

    return ret;
}

//==================================================================================================
/*
type_uid GridPoint::GetLabel() const
{
    if( m_seg )
        m_seg->GetLabel( m_id );
    else
        return GridPoint::cInvalidID;
}
*/

//==================================================================================================
void GridPoint::SetModeToOrthogonalNeighbors()
{
    m_nei_mode = OrthogonalNeighbors;
}

//==================================================================================================
void GridPoint::SetModeToAllNeighbors()
{
    m_nei_mode = AllNeighbors;
}

//==================================================================================================
void GridPoint::SetMode(NeighborhoodMode m)
{
    m_nei_mode = m;
}

//==================================================================================================
int GridPoint::ToIJK(size_t ijk[3]) const
{
    if( !ValidConstruction() )
    {
        MacroWarning("Invalid GridPoint.");
        return 0;
    }

    return m_grid->CalcGridIndex(m_id, ijk);
}

//==================================================================================================
GridPoint GridPoint::forward(uint axis) const
{
    size_t ijk[3] = {0,0,0};
    this->ToIJK(ijk);

    ijk[axis]++;

    //if( ijk[axis] < m_grid->x() )
    if( ijk[axis] < m_grid->Dim()[axis] )
    {
        type_uid xf_id = m_grid->CalcVoxelID(ijk);
        return GridPoint(xf_id, m_grid);
    }
    else
        //return GridPoint(0,0,0);
        return *this;
}

//==================================================================================================
GridPoint GridPoint::backward(uint axis) const
{
    size_t ijk[3] = {0,0,0};
    this->ToIJK(ijk);

    ijk[axis]--;

    //if( ijk[axis] < m_grid->x() )
    if( ijk[axis] < m_grid->Dim()[axis] )
    {
        type_uid xf_id = m_grid->CalcVoxelID(ijk);
        return GridPoint(xf_id, m_grid);
    }
    else
        //return GridPoint(0,0,0);
        return *this;
}

//==================================================================================================
GridPoint GridPoint::x_f() const
{
    return forward(0);
}

GridPoint GridPoint::x_b() const
{
    return backward(0);
}

GridPoint GridPoint::y_f() const
{
    return forward(1);
}

GridPoint GridPoint::y_b() const
{
    return backward(1);
}

GridPoint GridPoint::z_f() const
{
    return forward(2);
}

GridPoint GridPoint::z_b() const
{
    return backward(2);
}

//==================================================================================================
GridPoint& GridPoint::operator++()
{
    if( !ValidConstruction() )
    {
        MacroWarning("Invalid GridPoint.");
        return *this;
    }

    if( m_nei_iteration )
    {
        MacroWarning("Ongoing neighbor iteration terminated. Re-start iteration by calling StartNeighborIteration().");
        m_nei_iteration = false;
    }

    m_id++;
    if( m_id >= m_grid->GetArraySize() )
    {
        m_id = GridPoint::cInvalidID;
        SetInvalidConstruction();
    }

    return *this;
}

GridPoint & GridPoint::operator+(size_t offset)
{
    if( !ValidConstruction() )
    {
        MacroWarning("Invalid GridPoint.");
        return *this;
    }

    if( m_nei_iteration )
    {
        MacroWarning("Ongoing neighbor iteration terminated. Re-start iteration by calling StartNeighborIteration().");
        m_nei_iteration = false;
    }

    m_id += (uint)offset;
    if( m_id >= m_grid->GetArraySize() )
    {
        m_id = GridPoint::cInvalidID;
        SetInvalidConstruction();
    }

    return *this;
}

GridPoint & GridPoint::operator+=(size_t offset)
{
    return (*this + offset);
}

}