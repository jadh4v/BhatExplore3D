#pragma once
#include <stddef.h>

namespace sjDS{
class Image;
class VoxelRow;
}

typedef unsigned int uint;

class VolumeStats
{
public:
    // Construct and set the input volume over which statistics will be computed.
    VolumeStats(const sjDS::Image& vol);
    virtual ~VolumeStats();

    const std::vector<float>& GetHistogram() const;

    // Compute Histogram for entire input volume.
    int ComputeHistogram(uint binSize);

    // Compute Histogram for a given region within the input volume.
    int ComputeHistogram(const std::vector<sjDS::VoxelRow>& region, uint binSize);

    int ComputeBuckets( uint binSize, std::vector<uint>& buckets, const uint range[2] );


private:
    bool valid_data() const;
    bool valid_bin_size(uint binSize) const;
    static bool valid_range(const uint range[2]);

    // Data Members
    const sjDS::Image& m_volume;
    std::vector<float> m_histogram;
};
