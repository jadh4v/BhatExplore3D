#include <map>
#include <set>
#include <queue>
#include <bitset>
#include <cstring>
#include <QFile>
#include <QDataStream>
#include <QTime>
#include <vtkImageData.h>

#include "Image.h"
#include "Histogram.h"
#include "ImageSegmentation.h"
#include "Grid.h"
#include "GridPoint.h"
#include "Bitvector.h"
#include "core/macros.h"

using std::set;
using std::vector;
namespace sjDS{
//using hseg::Bitvector;
//using hseg::Histogram;

const type_uid ImageSegmentation::cInvalidLabel = UINT32_MAX;

void ImageSegmentation::init()
{
    m_histogram  = nullptr;
}

ImageSegmentation::~ImageSegmentation() noexcept
{
}


ImageSegmentation::ImageSegmentation()
{
    init();
}

ImageSegmentation::ImageSegmentation(const Grid& grid )
{
    init();
    if( grid.x() == 0 || grid.y() == 0 )
    {
        MacroWarning("grid must be of dimensions two or three.");
    }
    else
    {
        m_data = Image(&grid);
        InitializeLabels();
    }
}

ImageSegmentation::ImageSegmentation(const type_uint* data, const Grid& grid )
{
    init();
    if( grid.x() == 0 || grid.y() == 0 )
    {
        MacroWarning("grid must be of dimensions two or three.");
    }
    else if (data == nullptr)
    {
        MacroWarning("data is nullptr.");
    }
    else
    {
        m_data = Image(data, &grid);
    }
}
ImageSegmentation::ImageSegmentation(const sjDS::Image& image)
{
    init();
    m_data = image;
}

ImageSegmentation::ImageSegmentation(sjDS::Image&& image)
{
    init();
    m_data = image;
}

ImageSegmentation::ImageSegmentation(QDataStream& in_stream)
{
    init();
    if( !read( in_stream ) )
        DSObject::SetInvalidConstruction();
}

int ImageSegmentation::SetLabel(size_t point_id, type_uid label)
{
    m_data[point_id] = label;
    return 1;
}

type_uid ImageSegmentation::GetLabel(size_t point_id) const
{
    return m_data[point_id];
}

type_uid ImageSegmentation::GetLabel(size_t ijk[3]) const
{
    type_uid voxel_id = m_data.GetGrid()->CalcVoxelID(ijk);
    if(voxel_id == GridPoint::cInvalidID)
        return cInvalidLabel;
    else
        return GetLabel(voxel_id);
}

int ImageSegmentation::InitializeLabels()
{
    if(m_data.GetArraySize() == 0)
        return 0;

    // Don't use label == 0
    for(size_t i=0; i < m_data.GetArraySize(); ++i)
        m_data[i] = (type_uid)(i+1);

    return 1;
}

int ImageSegmentation::ClearLabels(uint value)
{
    // Don't use label == 0
    for(size_t i=0; i < m_data.GetArraySize(); ++i)
        m_data[i] = value;

    return 1;
}

int ImageSegmentation::ReplaceId(type_uid from, type_uid to)
{
    size_t sz = GetArraySize();
    std::replace( &m_data[0], &m_data[sz], from, to );
    return 1;
}

// TODO: This process is slow, as it goes through the entire image
// everytime a merge between regions is performed. Can we make this more efficient?
// Use R-Trees to restrict search?
type_uid ImageSegmentation::MergeRegions(type_uid region1_id, type_uid region2_id)
{
    type_uid smaller_id = std::min( region1_id, region2_id );
    type_uid  larger_id = std::max( region1_id, region2_id );

    /*
    for(size_t i=0; i < m_grid->GetArraySize(); ++i)
    {
        if( m_gridlabels[i] == larger_id )
            m_gridlabels[i] = smaller_id;
    }
    */

    //int success = get_region( larger_id, region );
    MacroAssert(larger_id > 0);
    if( larger_id == 0 )
        return GridPoint::cInvalidID;

    vector<type_uid> region;
    int success = get_region( larger_id, larger_id-1, region, GridPoint::AllNeighbors );

    if( ! success )
        return GridPoint::cInvalidID;

    for( auto iter = region.begin(); iter != region.end(); ++iter )
    {
        SetLabel(*iter, smaller_id);
    }

    return smaller_id;
}

int ImageSegmentation::get_region( type_uid region_id, size_t seed_pos,
                                   std::vector<type_uid>& region, GridPoint::NeighborhoodMode mode ) const
{
    if( seed_pos >= GetGrid()->GetArraySize() )
    {
        MacroWarning("seed_pos out of bounds.");
        return 0;
    }

    int retValue = 1;

    if( m_data[seed_pos] != region_id )
    {
        MacroFatalError("Wrong seed position provided.");
        retValue = 0;
    }
    else
    {
        set<type_uid> r;
        bfs( mode, seed_pos, r );

        if( m_histogram )
        {
            // if histogram mode is set, remap to histogram bins, and get the voxel_ids from original image.
            m_histogram->GetVoxels(r, region);
        }
        else 
        {
            // else forward the voxels directly.
            region.insert( region.end(), r.begin(), r.end() );
        }
    }

    return retValue;
}

// This is an expensive function (based on experiments):
int ImageSegmentation::get_region( type_uid region_id,
                                   vector<type_uid>& region, GridPoint::NeighborhoodMode mode ) const
{
    int retValue = 1;

    // find a seed voxel position which belongs to the larger_id region.
    type_uid seed_voxel_id = GridPoint::cInvalidID;
    for(size_t i=0; i < m_data.GetArraySize(); ++i)
    {
        if( m_data[i] == region_id )
        {
            seed_voxel_id = (type_uid)i;
            break;
        }
    }

    if( seed_voxel_id == GridPoint::cInvalidID )
    {
        MacroWarning("Cannot find seed voxel for region: " << region_id );
        retValue = 0;
    }
    else if( m_data[seed_voxel_id] != region_id )
    {
        MacroFatalError("Wrong seed position provided.");
        retValue = 0;
    }
    else
    {
        set<type_uid> r;
        bfs( mode, seed_voxel_id, r );
        if( m_histogram )
        {
            // if histogram mode is set, remap to histogram bins, and get the voxel_ids from original image.
            m_histogram->GetVoxels(r, region);
        }
        else 
        {
            // else forward the voxels directly.
            region.insert( region.end(), r.begin(), r.end() );
        }
    }

    return retValue;
}

// Do a breadth-first-search on a given seed voxel, within the seed's region.
// return a set of voxel ids that belong to the same region.
void ImageSegmentation::bfs( GridPoint::NeighborhoodMode mode, size_t seed_pos, set<type_uid>& region ) const
{
    // To be expanded:
    std::queue<type_uid> front;

    // region_id of region to be explored:
    type_uid seed_region = GetLabel(seed_pos);

    // begin BFS:
    front.push((type_uid)seed_pos);
    region.insert((type_uid)seed_pos);

    while( !front.empty() )
    {
        type_uid curr_pos = front.front();
        front.pop();

        GridPoint p( curr_pos, m_data.GetGrid());
        p.SetMode(mode);
        p.StartNeighborIteration();
        type_uid nei_pos = p.GetNextNeighborID();

        while( nei_pos != GridPoint::cInvalidID )
        {
            if( region.find(nei_pos) == region.end() )
            {
                type_uid nei_region = GetLabel(nei_pos);

                if( nei_region == seed_region )
                {
                    region.insert( nei_pos );
                    front.push( nei_pos );
                }
            }

            nei_pos = p.GetNextNeighborID();
        }
    }

}

// Do a breadth-first-search on a given seed voxel, within the seed's region.
// return the count of voxels that are on the boundary of the region (external voxels).
size_t ImageSegmentation::external_voxels_bfs( GridPoint::NeighborhoodMode mode, size_t seed_pos ) const
{
    // To be expanded:
    std::queue<type_uid> front;
    std::set<type_uid> region;
    std::set<type_uid> boundary;

    // region_id of region to be explored:
    type_uid seed_region = GetLabel(seed_pos);

    // begin BFS:
    front.push((type_uid)seed_pos);
    region.insert((type_uid)seed_pos);

    while( !front.empty() )
    {
        type_uid curr_pos = front.front();
        front.pop();

        bool curr_pos_marked = false;
        GridPoint p(curr_pos, m_data.GetGrid());
        p.SetMode(mode);
        p.StartNeighborIteration();
        type_uid nei_pos = p.GetNextNeighborID();

        while( nei_pos != GridPoint::cInvalidID )
        {
            if( region.find(nei_pos) == region.end() )
            {
                type_uid nei_region = GetLabel(nei_pos);

                if( nei_region == seed_region )
                {
                    region.insert( nei_pos );
                    front.push( nei_pos );
                }
                else if( !curr_pos_marked )
                {
                    boundary.insert( curr_pos );
                    curr_pos_marked = true;
                }
            }

            nei_pos = p.GetNextNeighborID();
        }
    }

    return boundary.size();
}

size_t ImageSegmentation::external_voxels_brute( GridPoint::NeighborhoodMode mode, size_t seed_pos ) const
{
    // To be expanded:
    std::queue<type_uid> front;
    std::set<type_uid> region;
    std::set<type_uid> boundary;

    // region_id of region to be explored:
    type_uid seed_region = GetLabel(seed_pos);

    // begin BFS:
    front.push((type_uid)seed_pos);
    region.insert((type_uid)seed_pos);

    while( !front.empty() )
    {
        type_uid curr_pos = front.front();
        front.pop();

        bool curr_pos_marked = false;
        GridPoint p(curr_pos, m_data.GetGrid());
        p.SetMode(mode);
        p.StartNeighborIteration();
        type_uid nei_pos = p.GetNextNeighborID();

        while( nei_pos != GridPoint::cInvalidID )
        {
            if( region.find(nei_pos) == region.end() )
            {
                type_uid nei_region = GetLabel(nei_pos);

                if( nei_region == seed_region )
                {
                    region.insert( nei_pos );
                    front.push( nei_pos );
                }
                else if( !curr_pos_marked )
                {
                    boundary.insert( curr_pos );
                    curr_pos_marked = true;
                }
            }

            nei_pos = p.GetNextNeighborID();
        }
    }

    return boundary.size();
}

type_uid ImageSegmentation::CollapseRegionIDs()
{
    return CollapseRegionIDs(1);
}

type_uid ImageSegmentation::CollapseRegionIDs( type_uid begin_id )
{
    // save the remapping of region_ids while editing the labels array:
    std::map<type_uid,type_uid> reassign;
    type_uid curr_max = begin_id;

    for(size_t i=0; i < m_data.GetArraySize(); ++i)
    {
        type_uid old_label = m_data[i];
        type_uid new_label = 0;
        auto fnd = reassign.find(old_label);
        if( fnd != reassign.end() )
        {
            new_label = fnd->second;
        }
        else
        {
            new_label = curr_max++;
            reassign.insert(std::make_pair(old_label,new_label));
        }

        MacroAssert( new_label != 0 );
        m_data[i] = new_label;
    }
    return (type_uid)reassign.size();
}

bool ImageSegmentation::is2D() const
{
    return m_data.is2D();
}

bool ImageSegmentation::is3D() const
{
        return m_data.is3D();
}

bool ImageSegmentation::hasContiguousLabels() const
{
    const size_t sz = m_data.GetArraySize();
    Bitvector bits(sz);

    // Since segmentation region ID == 0 is reserved as invalid value,
    // we mark it as set (used) in the Bitvector.
    // Else, the labeling range will never be contiguous.
    bits.Set(0);

    for(size_t i=0; i < sz; ++i)
    {
        int success = bits.Set( m_data[i] );

        // If label_id is out of range, then clearly the labels are not contiguous.
        // Contiguous labels will never have an id that is greater than the size of the
        // data array used for ImageSegmentation object.
        if( !success )
            return false;
    }

    bool boundary = false;
    for( size_t i=0; i < sz; ++i )
    {
        bool b = bits.Get(i);
        if( !boundary && !b )
        {
            boundary = true;
            continue;
        }

        // boundary between set bits and zero bits was already found.
        // A non-zero element after this event indicates non-contiguous range of label ids.
        if( boundary && b )
            return false;
    }

    return true;
}

int ImageSegmentation::Write(const QString& filename) const
{
    QFile segfile(filename);
    if( !segfile.open(QIODevice::WriteOnly) )
    {
        MacroWarning("Cannot open file to write: " << filename.toLatin1().constData());
        return 0;
    }

    QDataStream out_stream( &segfile );
    return Write( out_stream );
}

int ImageSegmentation::Write(QDataStream& out_stream) const
{
    int ret = m_data.GetGrid()->Write( out_stream );

    if( ret )
    {
        size_t sz = m_data.GetArraySize();
        size_t bytesToWrite = sz * sizeof(m_data[0]);
        int bytesWritten = out_stream.writeRawData((const char*)(m_data.GetDataPointer()), static_cast<int>(bytesToWrite));
        if( bytesWritten != (int)bytesToWrite)
        {
            MacroWarning("Failed to write ImageSegmentation.");
            return 0;
        }
    }
    return ret;
}

int ImageSegmentation::read(QDataStream& in_stream)
{
    int ret = 1;
    Grid grid( in_stream );

    if( grid.ValidConstruction() )
    {
        size_t sz = grid.GetArraySize();
        vector<type_uid> buff(sz);
        size_t numOfBytes = sz*sizeof(type_uid);
        if( in_stream.readRawData((char*)buff.data(), static_cast<int>(numOfBytes)) != (int)numOfBytes )
        {
            MacroWarning("Unexpected end of data stream.");
            ret = 0;
        }
        m_data = Image(buff.data(), &grid);
    }
    else
    { 
        ret = 0;
    }

    return ret;
}

bool ImageSegmentation::isEqual(const ImageSegmentation& seg) const
{
    // Invalid objects are always unequal.
    if( !this->ValidConstruction() || !seg.ValidConstruction() )
        return false;

    // compare grid dimensions:
    /*
    const Grid* seg_grid = seg.GetGrid();

    if( m_grid.x() != seg_grid->x() )
        return false;

    if( m_grid.y() != seg_grid->y() )
        return false;

    if( m_grid.z() != seg_grid->z() )
        return false;
        */
    if( *(this->GetGrid()) != *(seg.GetGrid()) )
        return false;

    size_t array_sz = this->GetArraySize();

    // compare segmentations voxel-by-voxel:
    for(size_t i=0; i < array_sz; ++i)
    {
       if( this->GetLabel(i) != seg.GetLabel(i) )
           return false;
    }
    return true;
}

Image ImageSegmentation::ConvertToImage() const
{
    return m_data;
}

int ImageSegmentation::CountRegions() const
{
    size_t sz = this->GetArraySize();
    set<type_uid> uniqueRegionIds;
    for( size_t i=0; i < sz; ++i )
    {
        type_uid reg = m_data[i];
        uniqueRegionIds.insert(reg);
    }

    return static_cast<int>(uniqueRegionIds.size());
}

size_t ImageSegmentation::CountExternalVoxels(type_uid region1_id) const
{
    return external_voxels_bfs( GridPoint::OrthogonalNeighbors, region1_id-1 );
    //return external_voxels_brute( GridPoint::AllNeighbors, region1_id-1 );
}

bool ImageSegmentation::TestConnectedComponents(GridPoint::NeighborhoodMode mode) const
{
    //NullCheck(m_gridlabels,false);

    size_t sz = this->GetArraySize();
    set<type_uid> uniqueRegionIds;
    for( size_t i=0; i < sz; ++i )
    {
        type_uid reg = m_data[i];
        uniqueRegionIds.insert(reg);
    }

    for( auto&& uregId : uniqueRegionIds )
    {
        //if( uregId == 0 ) continue;
        vector<type_uid> tmp;
        get_region(uregId, tmp, mode );

        set<type_uid> region(tmp.begin(), tmp.end());

        for(size_t i=0; i < sz; ++i)
        {
            if( m_data[i] == uregId )
            {
                // If the found voxel has current region ID, but was not returned by a BFS
                // on the region, then current region is not connected. Return false.
                if( region.find((type_uid)i) == region.end() )
                    return false;
            }
        }
    }

    return true;
}

// Copy Constructor
ImageSegmentation::ImageSegmentation(const ImageSegmentation& A )
{
    init();
    this->m_data = A.m_data;
    m_histogram = A.m_histogram;
}

// Copy Assignment Operator
ImageSegmentation& ImageSegmentation::operator=(const ImageSegmentation& A )
{
    ImageSegmentation tmp(A); // re-use copy-constructor
    *this = std::move(tmp);   // re-use move-assignment
    return *this;
}

// Move Constructor
ImageSegmentation::ImageSegmentation(ImageSegmentation&& A) noexcept
{
    init();
    m_data = std::move(A.m_data);
    m_histogram = A.m_histogram;
    A.m_histogram = nullptr;
}

// Move Assignment Operator
ImageSegmentation& ImageSegmentation::operator=(ImageSegmentation&& A) noexcept
{
    init();
    m_histogram = A.m_histogram;
    m_data = std::move(A.m_data);
    return *this;
}

int ImageSegmentation::GetRegionBrut( type_uid region_id, std::vector<type_uid>& region ) const
{
    size_t array_sz = GetArraySize();
    set<type_uid> r;
    for(size_t i=0;i<array_sz;++i)
    {
        type_uid r_id = m_data[i];
        if( r_id == region_id )
            r.insert((type_uid)i);
    }

    // Re-evaluate voxels to histogram-image mapping if histogram mode is ON.
    if( m_histogram )
    {
        // if histogram mode is set, remap to histogram bins, and get the voxel_ids from original image.
        m_histogram->GetVoxels(r, region);
    }
    else 
    {
        // else forward the voxels directly.
        region.insert( region.end(), r.begin(), r.end() );
    }

    return 1;
}

int ImageSegmentation::GetRegion( type_uid region_id, std::vector<type_uid>& region,
                                  GridPoint::NeighborhoodMode mode ) const
{
    int retValue = get_region( region_id, region, mode );
    return retValue;
}

int ImageSegmentation::GetRegion( type_uid region_id, type_uid seed_voxel,
                                  vector<type_uid>& region, GridPoint::NeighborhoodMode mode) const
{
    int retValue = get_region( region_id, seed_voxel, region, mode );
    return retValue;
}

size_t ImageSegmentation::GetRegionSize( type_uid region_id, GridPoint::NeighborhoodMode mode ) const
{
    size_t retValue = 0;
    // find a seed voxel position which belongs to the larger_id region.
    type_uid seed_pos = GridPoint::cInvalidID;
    for(size_t i=0; i < this->GetArraySize(); ++i)
    {
        if( m_data[i] == region_id )
        {
            seed_pos = (type_uid)i;
            break;
        }
    }

    if( m_data[seed_pos] != region_id )
    {
        MacroFatalError("Wrong seed position provided.");
        retValue = 0;
    }
    else
    {
        set<type_uid> region;
        bfs( mode, seed_pos, region );
        retValue = region.size();
    }

    return retValue;
}

size_t ImageSegmentation::GetArraySize() const
{
    return m_data.GetArraySize();
}

bool ImageSegmentation::IsNull() const
{
    return (m_data.GetArraySize() == 0);
}

void ImageSegmentation::GetUniqueValues(std::vector<type_uint>& unique_values ) const
{
    std::set<type_uint> tmp;
    size_t numOfVoxels = GetArraySize();
    for( size_t i=0; i < numOfVoxels; ++i)
    {
        tmp.insert( m_data[i] );
    }

    unique_values.clear();
    unique_values.insert( unique_values.end(), tmp.begin(), tmp.end() );
}

void ImageSegmentation::SetHistogramMapping(const Histogram* histogram)
{
    m_histogram = histogram;
}

int ImageSegmentation::RelabelForRandomAccess(GridPoint::NeighborhoodMode mode)
{
    if( !TestConnectedComponents(mode) )
    {
        MacroWarning("Regions are not single connected components.");
        return 0;
    }

    vector<uint> unique_labels;
    GetUniqueValues(unique_labels);
    std::map<uint,uint> remapped;
    size_t array_sz = GetArraySize();

    for(size_t i=0; i < array_sz; ++i)
    {
        type_uid old_label = GetLabel(i);
        auto fnd = remapped.find(old_label);
        if( fnd != remapped.end() )
        {
            SetLabel( i, fnd->second );
        }
        else
        {
            remapped.insert( std::make_pair(old_label, (uint)i+1) );
            SetLabel( i, (type_uid)i+1 );
        }
    }

    return 1;
}

int ImageSegmentation::ScaleUp(const uint factor)
{
    if( factor % 2 != 0 )
    {
        MacroWarning("Only supports scaling factor as multiples of 2.");
        return 0;
    }

    // Invalidate the current histogram.
    m_histogram = nullptr;

    // make a copy of current label array.
    Grid old_grid( *m_data.GetGrid() );
    vector<uint> old_labels(m_data.GetArraySize());
    std::copy( &m_data[0], &m_data[old_grid.GetArraySize()], old_labels.data() );

    // scale up the grid
    Grid new_grid(old_grid);
    for(uint tmp_factor = factor; tmp_factor > 1; tmp_factor /= 2)
    { 
        new_grid.ScaleUpBy2();
    }

    m_data = Image(&new_grid);

    //size_t array_sz = new_grid.GetArraySize();
    //MacroDelete(m_gridlabels);
    //m_gridlabels = new type_uid[array_sz];

    size_t new_dim[3], old_dim[3];
    new_grid.GetDimensions(new_dim);
    old_grid.GetDimensions(old_dim);
    for(size_t k=0; k < new_dim[2]; ++k)
    for(size_t j=0; j < new_dim[1]; ++j)
    for(size_t i=0; i < new_dim[0]; ++i)
    { 
        size_t new_pos = i+j*new_dim[0]+k*new_dim[0]*new_dim[1];
        size_t old_pos = (i/factor) + (j/factor)*old_dim[0] + (k/factor)*old_dim[0]*old_dim[1];
        m_data[new_pos] = old_labels[old_pos];
    }

    return 1;
}

int ImageSegmentation::trimZ(size_t to_z)
{
    MacroConfirmOrReturn((to_z < m_data.z()), 0);
    Grid g(*(m_data.GetGrid()));
    g.SetDimensions(m_data.x(), m_data.y(), to_z);
    m_data = Image(m_data.GetDataPointer(), &g);
    return 1;
}

type_uint ImageSegmentation::operator[](size_t i) const
{
    return m_data[i];
}

type_uint& ImageSegmentation::operator[](size_t i)
{
    return m_data[i];
}

}

