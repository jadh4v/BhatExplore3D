#pragma once

#include "DSObject.h"

namespace sjDS{

class GridCell : public DSObject
{
public:
    enum InterpolationMode{ Nearest=0, Trilinear };
    static constexpr size_t cPointCount = 8;

    GridCell(const double* values);
    GridCell(double v0, double v1, double v2, double v3, double v4, double v5, double v6, double v7);

/// Interpolation is performed considering provided GridCell values as unsigned integers
/// and, they are considered as segmentation labels.
/// Each label is considered separately for interpolation, 
/// where the label is replaced with value 1 and all other points are initialized to 0.
/// The interpolation position is tested for a value greater than 0.5. 
/// If true, that point is considered to belong to the current segmentation label.
    unsigned int InterpolateSegmentLabels(const double point[3]) const;

/// Interpolate values as scalars, given normalized coordinates of a point inside the cell.
    double Interpolate(const double point[3]) const;

private:
    /// order of points is that x-coord varies first, y-coord varies second, and z-coord varies last.
    /**
                  6 ------ 7
                 /|       /|
               2 ------ 3  |
               |  |     |  |
      Y        |  4-----|--5
      ^  Z     | /      | /
      | /      0 ------ 1
      |/
      L----->X
    */
    double m_Values[cPointCount]; // any data-type is converted to double before processing.
    InterpolationMode m_InterpolationMode = Trilinear;

};

}