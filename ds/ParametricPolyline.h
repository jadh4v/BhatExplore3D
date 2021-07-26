#pragma once
#include <map>

class vtkPolyData;
class vtkPolyLine;

namespace sjDS {

    /// The class ParametricPolyline represents a database to evaluate a 
    /// polyline parametrically, defining a parameter t from 0.0 to 1.0
    /// across the total length of the polyline. This structure can be
    /// used to evaluate / calculate points along a vtkPolyLine curve
    /// based on normalized parameter t.
class ParametricPolyline
{
public:
    ParametricPolyline(vtkSmartPointer<vtkPolyLine> curve);
    /// Evaluate the actual point position on the polyline, given paramter t.
    /// Parameter t will be clamped from 0 to 1.
    bool Evaluate(double t, double x[3]) const;
    double TotalLength() const;
    vtkSmartPointer<vtkPolyData> ConvertToPolyData();

private:
    /// Orient the polyline in anti-clockwise direction.
    /// This function assumes that the given polyline is 2D residing in XY plane and is a closed loop.
    void _Orient();

    /// This function assumes that Orient was called and the polyline is planar 2D curve inside X-Y plane.
    /// Once orient is called, the loop is oriented in the anti-clockwise direction.
    /// Hence, locally, the normals will always lie on the right hand side of each line-segment.
    static int _CalculateNormals(vtkPolyData* curve);

    /// Compute the sum of cross products of consecutive line segments.
    /// Such a calculation can be used for determining the loop orientation of the polyline.
    void _TotalCross(double cross[3]);

    /// Flip the direction of polyline.
    void _Flip();

    /// Recalculate the parametric curve. i.e. the relationship between points and the normalized paramter
    /// 't' that goes from 0 to 1.
    void _UpdateParametricForm();

    vtkSmartPointer<vtkPolyLine> m_curve = nullptr;
    double m_totalLength = 0.0;
    std::multimap<double, vtkIdType> m_positionToID;

};

}