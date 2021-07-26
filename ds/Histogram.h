#pragma once

#include <vector>
#include <set>
#include "Image.h"
#include "DSObject.h"

namespace sjDS {

/**
 * @brief Abstract class Histogram
 * Interface to query a 2D or 3D histograms.
 */
class Histogram : public DSObject
{
public:
    enum Axis{ X_Axis=0, Y_Axis, Z_Axis };
    virtual Image ConvertToImage() const = 0;
    virtual const Image& AccessAsImage() const = 0;
    virtual size_t GetRegionSize(const std::set<type_uid>& i_Bins) const = 0;
    virtual int GetVoxels(const std::set<type_uid>& pi_Bins, std::vector<type_uid>& po_voxels) const = 0;
};

}
