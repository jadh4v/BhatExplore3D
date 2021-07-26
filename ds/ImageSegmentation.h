#ifndef IMAGESEGMENTATION_H
#define IMAGESEGMENTATION_H

#include <cstdint>
#include <set>
#include <vector>

#include <QString>

//#include "graphseg_globals.h"
#include "DSObject.h"
#include "Grid.h"
#include "GridPoint.h"
#include "Image.h"

class QDataStream;
class vtkImageData;

namespace sjDS{

class Grid;
class Histogram;

class ImageSegmentation : public DSObject
{
public:
    typedef unsigned int type_uid;
    typedef unsigned int type_uint;
    static const type_uid cInvalidLabel;

    // Default Constructor
    ImageSegmentation();
    // Destructor
    virtual ~ImageSegmentation() noexcept;
    // Copy Constructor
    ImageSegmentation(const ImageSegmentation&);
    // Copy Assignment Operator
    ImageSegmentation& operator=(const ImageSegmentation& );
    // Move Constructor
    ImageSegmentation(ImageSegmentation&& A) noexcept;
    // Move Assignment Operator
    ImageSegmentation& operator=(ImageSegmentation&& ) noexcept;

    // Other Constructors
    ImageSegmentation(const Grid& grid );
    ImageSegmentation(const type_uint* data, const Grid& grid );
    ImageSegmentation(const sjDS::Image& image);
    ImageSegmentation(sjDS::Image&& image);
    /// Construct by reading a data stream.
    ImageSegmentation( QDataStream& in_stream );

    /// initialize segment labels such that each grid-point is a separate segment.
    int InitializeLabels();
    int ClearLabels(uint value=0);
    /// Set the region or segment ID for a given grid point.
    int SetLabel(size_t point_id, type_uid label);
    /// Get the region or segment ID for a given grid point.
    type_uid GetLabel(size_t point_id) const;
    /// Get the region or segment ID for a given grid point.
    type_uid GetLabel(size_t ijk[3]) const;
    const Grid* GetGrid() const { return m_data.GetGrid(); }
    const type_uid* GetGridLabels() const { return m_data.GetDataPointer(); }
    type_uid* GetGridLabels() { return m_data.GetDataPointer(); }
    /// This method simply updated the region with larger id value
    /// to the id of region with smaller id value. It does not confirm
    /// whether the regions being merged are neighbors or not.
    /// This is assumed to be true.
    /// Returns the new id of the merged region.
    type_uid MergeRegions( type_uid region1_id, type_uid region2_id );
    /// Move all region IDs to consecutive integers starting from 1.
    /// Return the total number of regions.
    type_uid CollapseRegionIDs();
    /// Move all region IDs to consecutive integers starting from begin_id.
    /// Return the total number of regions.
    type_uid CollapseRegionIDs(type_uid begin_id );
    /// Count the total number of regions (unique IDs) in the segmentation image.
    int CountRegions() const;

    size_t CountExternalVoxels( type_uid region1_id ) const;

    /// Check if the base image is 2D or 3D grid.
    bool is2D() const;
    /// Check if the base image is 2D or 3D grid.
    bool is3D() const;

    bool hasContiguousLabels() const;
    /// Write the image segmentation to a file.
    int Write(const QString& filename) const;
    /// Write the image segmentation to a data stream.
    int Write(QDataStream& out_stream) const;
    /// Compare two image segmentation to decide wether they are equal:
    bool isEqual(const ImageSegmentation& seg) const;
    Image ConvertToImage() const;
    bool TestConnectedComponents(GridPoint::NeighborhoodMode mode) const;
    int GetRegion( type_uid region_id, std::vector<type_uid>& region, GridPoint::NeighborhoodMode mode ) const;
    int GetRegion( type_uid region_id, type_uid seed_voxel, std::vector<type_uid>& region, GridPoint::NeighborhoodMode mode) const;
    int GetRegionBrut( type_uid region_id, std::vector<type_uid>& region ) const;
    size_t GetRegionSize( type_uid region_id, GridPoint::NeighborhoodMode mode ) const;
    size_t GetArraySize() const;
    bool IsNull() const;
    void GetUniqueValues(std::vector<type_uint>& unique_values ) const;
    void SetHistogramMapping(const Histogram* histogram);

    /// Relabel regions so that the first voxel of each region is at label-1.
    /// This can be used to directly access any region using m_data[label-1]
    /// and then accessing all voxels of that region through flood-fill.
    /// Regions of the segmentation are required to be connected-components.
    /// Returns 1 if success, 0 if failure.
    int RelabelForRandomAccess(GridPoint::NeighborhoodMode mode);

    /// Replace a region with new id (from_id in any voxel will be replaced with to_id).
    /// This function does a complete scan of the entire grid. No floodfill is used.
    /// No other assumption is made about the segmentation.
    int ReplaceId(type_uid from, type_uid to);

    /// Scale the segmentation image by the provided factor.
    /// Currently implementation only supports scaling in multiples of 2.
    int ScaleUp(const uint factor);

    /// Trim the segmentation image by removing n number of z-slices from the end of the volume.
    int trimZ(size_t to_z);

    type_uint  operator[](size_t i) const;
    type_uint& operator[](size_t i);

    //
    //bool MakeConnectedComponents(GridPoint::NeighborhoodMode mode) const

private:

    void init();

    /// Read image segmentation from a data stream.
    int read(QDataStream& in_stream);

    //int get_region( type_uid region_id, std::set<type_uid>& region ) const;
    //int get_region( type_uid region_id, size_t seed_pos, std::set<type_uid>& region ) const;
    int get_region( type_uid region_id, std::vector<type_uid>& region, GridPoint::NeighborhoodMode ) const;
    int get_region( type_uid region_id, size_t seed_pos, std::vector<type_uid>& region, GridPoint::NeighborhoodMode ) const;
    void bfs( GridPoint::NeighborhoodMode mode, size_t seed_pos, std::set<type_uid>& region ) const;
    void process_label(type_uid voxel);
    size_t external_voxels_bfs( GridPoint::NeighborhoodMode mode, size_t seed_pos ) const;
    size_t external_voxels_brute( GridPoint::NeighborhoodMode mode, size_t seed_pos ) const;

    // Data Members 
    //Grid             m_grid;       /**< Image grid details to which this segmentation belongs.*/
    //type_uid*        m_gridlabels; /**< Array of labels corresponding to each grid point in the grid / image. */
    Image m_data;       /**< Image based representation of the segmentation. */
    const Histogram* m_histogram;

};

}

#endif // IMAGESEGMENTATION_H
