#include <vector>
#include "macros.h"
#include "ds/Image.h"
#include "ds/VoxelRow.h"
#include "VolumeStats.h"

using sjDS::Image;
using sjDS::VoxelRow;

bool VolumeStats::valid_bin_size(uint binSize) const
{
    if(binSize > 0)
        return true;
    else
    {
        MacroWarning("Illegal Bin Size.");
        return false;
    }
}

bool VolumeStats::valid_range(const uint range[2])
{
    if( range[0] < range[1] )
        return true;
    else
    {
        MacroWarning("Invalid range.");
        return false;
    }
}

bool VolumeStats::valid_data() const
{
    if( m_volume.GetArraySize() > 0 )
        return true;
    else
    {
        MacroWarning("Illegal volume.");
        return false;
    }
}

VolumeStats::VolumeStats(const Image& input_volume) : m_volume(input_volume)
{
}

VolumeStats::~VolumeStats()
{
}

int VolumeStats::ComputeHistogram(const std::vector<VoxelRow>& region, uint binSize)
{
    m_histogram.clear();
    MacroConfirmOrReturn( valid_data(), 0 );

    // Get scalar range within region
    uint range[2];
    m_volume.GetScalarRange( range, region );

    // Compute histogram buckets
    std::vector<uint> buckets;
    int success = ComputeBuckets( binSize, buckets, range );
    if( !success )
        return 0;

    // Compute histogram
    m_histogram.resize( buckets.size() );
    memset( &(m_histogram[0]), 0, sizeof(float)*m_histogram.size() );

    for( auto& r : region )
    {
        for( sjDS::voxelNo_t i = r.Start(); !r.atEnd(i); ++i )
        {
            uint val = m_volume[i] - range[0];
            uint offset = val / binSize;
            MacroAssert(offset < (uint)m_histogram.size());
            m_histogram[offset]++;
        }
    }

    //Get max bucket value
    auto fnd = std::max_element( m_histogram.begin(), m_histogram.end() );
    float maxBucketValue = fnd != m_histogram.end() ? *fnd : 0.0f;

    // Normalize based on max value
    std::for_each(m_histogram.begin(), m_histogram.end(), [maxBucketValue](float& value){ value /= maxBucketValue;} );

    return 1;
}

int VolumeStats::ComputeHistogram(uint binSize)
{
    m_histogram.clear();
    MacroConfirmOrReturn( valid_data(), 0 );

    // compute scalar range of volume
    uint range[2];
    m_volume.GetScalarRange( range );

    // compute histogram buckets
    std::vector<uint> buckets;
    int success = ComputeBuckets( binSize, buckets, range );
    if( !success )
        return 0;

    // compute histogram
    m_histogram.resize( buckets.size() );
    memset( &(m_histogram[0]), 0, sizeof(float)*m_histogram.size() );

    size_t dim[3];
    m_volume.GetDimensions(dim);

    size_t sz = dim[0]*dim[1]*dim[2];
    for( size_t i=0; i < sz; i++ )
    {
        uint val = m_volume[i] - range[0];
        uint offset = val / binSize;
        MacroAssert(offset < (uint)m_histogram.size());
        m_histogram[offset]++;
    }

    //Get max bucket value
    auto fnd = std::max_element(m_histogram.begin(), m_histogram.end() );
    float maxBucketValue = fnd != m_histogram.end() ? *fnd : 0.0f;

    // Normalize based on max value
    std::for_each(m_histogram.begin(), m_histogram.end(), [maxBucketValue](float& value){ value /= maxBucketValue;} );

    return 1;
}

int VolumeStats::ComputeBuckets( uint binSize, std::vector<uint>& buckets, const uint range[2] )
{
    MacroConfirmOrReturn( valid_data(), 0 );
    MacroConfirmOrReturn( valid_bin_size(binSize), 0 );
    MacroConfirmOrReturn( valid_range(range), 0 );

    uint diff = range[1] - range[0];

    // reserve approximate memory for efficiency.
    size_t approxBucketCount = (size_t)diff / binSize;
    buckets.reserve( approxBucketCount );

    for( int initvalue = (int)range[0]; initvalue <= (int)range[1]; initvalue += binSize )
        buckets.push_back(initvalue);

    return 1;
}

const std::vector<float> &VolumeStats::GetHistogram() const
{
    return m_histogram;
}


