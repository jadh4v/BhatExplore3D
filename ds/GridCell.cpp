#include <algorithm>
#include <set>
#include <vtkMath.h>
#include "GridCell.h"
#include "ImageSegmentation.h"

namespace sjDS{

GridCell::GridCell(const double * values)
{
    for (size_t i = 0; i < cPointCount; ++i)
        m_Values[i] = values[i];
}

GridCell::GridCell(double v0, double v1, double v2, double v3, double v4, double v5, double v6, double v7)
{
    m_Values[0] = v0;
    m_Values[1] = v1;
    m_Values[2] = v2;
    m_Values[3] = v3;
    m_Values[4] = v4;
    m_Values[5] = v5;
    m_Values[6] = v6;
    m_Values[7] = v7;
}

double GridCell::Interpolate(const double point[3]) const
{
    // Clamp input normalized coordinates for further safe calculations.
    double x_coord = vtkMath::ClampValue(point[0], 0.0, 1.0);
    double y_coord = vtkMath::ClampValue(point[1], 0.0, 1.0);
    double z_coord = vtkMath::ClampValue(point[2], 0.0, 1.0);

    // compute interpolation on lines.
    double line_values[4] = {0,0,0,0};
    for (size_t i = 0; i < 4; ++i)
    {
        const double* v = &( m_Values[i*2] );
        line_values[i] = (1.0-x_coord) * v[0] + x_coord * v[1];
    }

    // compute interpolation on planes.
    double plane_values[2] = {0,0};
    for (size_t i = 0; i < 2; ++i)
    {
        const double* v = &( line_values[i*2] );
        plane_values[i] = (1.0-y_coord) * v[0] + y_coord * v[1];
    }

    // compute interpolation inside cell.
    double retValue = (1.0-z_coord) * plane_values[0] + z_coord * plane_values[1];
    return retValue;
}

unsigned int GridCell::InterpolateSegmentLabels(const double point[3]) const
{
    unsigned int cell_values[cPointCount];
    for (size_t i = 0; i < cPointCount; ++i)
        cell_values[i] = (unsigned int)m_Values[i];

    // Clamp input normalized coordinates for further safe calculations.
    double p[] = { point[0], point[1], point[2] };
    p[0] = vtkMath::ClampValue(p[0], 0.0, 1.0);
    p[1] = vtkMath::ClampValue(p[1], 0.0, 1.0);
    p[2] = vtkMath::ClampValue(p[2], 0.0, 1.0);

    // Collect all unique labels present in the cell.
    std::set<unsigned int> unique_labels;
    for (size_t i = 0; i < cPointCount; ++i)
    {
        unique_labels.insert(cell_values[i]);
    }

    // For each segmentation label present in the cell, perform the following:
    std::vector<double> inter;
    size_t max_label = 0;
    double max_interpolated = 0;
    for (auto label : unique_labels)
    {
        double unit_values[cPointCount];
        for(size_t i=0; i < cPointCount; ++i)
        { 
            unit_values[i] = cell_values[i] == label? 1 : 0;
        }

        GridCell unit_cell(unit_values);
        double interpolated = unit_cell.Interpolate(point);
        inter.push_back(interpolated);
        if (interpolated > 0.49)
            return label;
        else if (interpolated > max_interpolated)
        {
            max_interpolated = interpolated;
            max_label = label;
        }

    }

    // if none of the labels qualify, return background label which is always zero.
    /*
    std::cout << "interpolated: ";
    for (auto& i : inter)
        std::cout << i << ", ";
    std::cout << std::endl;
    */
    return max_label;
    //return ImageSegmentation::cInvalidLabel;
}

}