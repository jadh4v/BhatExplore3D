#include <vtkDoubleArray.h>
#include <vtkIdList.h>
#include <vtkMath.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyLine.h>
#include <ParametricPolyline.h>
#include <core/macros.h>

using sjDS::ParametricPolyline;

ParametricPolyline::ParametricPolyline(vtkSmartPointer<vtkPolyLine> curve)
{
    NullCheckVoid(curve);
    MacroAssert(curve->GetNumberOfPoints() >= 2);
    m_curve = curve;
    double totalLength = 0.0;
    _UpdateParametricForm();
}

bool ParametricPolyline::Evaluate(double t, double x[3]) const
{
    x[0] = 0; x[1] = 0; x[2] = 0;
    bool success = false;
    double range[] = { 0.0, 1.0 };
    vtkMath::ClampValue(&t, range);
    double L = t * m_totalLength;
    auto fnd = m_positionToID.lower_bound(L);
    if (fnd != m_positionToID.end())
    {
        vtkIdType id0 = fnd->second; // id of the first point of identified line-segment.
        double t0 = fnd->first / m_totalLength; // parametric value of the first point.
        double t_residue = t - t0;

        // move iterator to the second point of the line segment.
        fnd++;
        if (fnd != m_positionToID.end())
        {
            double t1 = fnd->first / m_totalLength; // parametric value of the second point.
            double t_segment = t_residue / (t1 - t0);
            vtkMath::ClampValue(&t_segment, range);
            vtkIdType id1 = fnd->second;
            double p0[3] = { 0,0,0 }, p1[3] = { 0,0,0 }, p0p1[3] = { 0,0,0 };
            m_curve->GetPoints()->GetPoint(id0, p0);
            m_curve->GetPoints()->GetPoint(id1, p1);
            vtkMath::Subtract(p1, p0, p0p1);
            vtkMath::MultiplyScalar(p0p1, t_segment);
            vtkMath::Add(p0, p0p1, x);
            success = true;
        }
        else
        {
            m_curve->GetPoints()->GetPoint(id0, x);
            success = true;
        }
    }
    return success;
}

double ParametricPolyline::TotalLength() const
{
    return m_totalLength;
}

vtkSmartPointer<vtkPolyData> ParametricPolyline::ConvertToPolyData()
{
    _Orient();
    vtkNew<vtkPolyData> ret;
    vtkNew<vtkCellArray> cells;
    vtkNew<vtkPoints> points;
    points->DeepCopy(m_curve->GetPoints());
    ret->SetPoints(points);
    ret->SetLines(cells);

    vtkIdType maxId = m_curve->GetPointIds()->GetNumberOfIds();
    for (vtkIdType id = 0; id < maxId-1; ++id)
    {
        vtkIdType pts[2] = { 0,0 };
        pts[0] = m_curve->GetPointId(id);
        pts[1] = m_curve->GetPointId((id+1)%maxId);
        ret->InsertNextCell(VTK_LINE, 2, pts);
    }

    _CalculateNormals(ret);

    return ret;
}

void ParametricPolyline::_Orient()
{
    NullCheckVoid(m_curve);
    if (m_curve->GetNumberOfPoints() < 3)
        return;

    double up[3] = { 0,0,1 }, totalCross[3] = { 0,0,0 };
    _TotalCross(totalCross);
    if (vtkMath::Dot(totalCross, up) < 0)
    {
        _Flip();
    }
}

int sjDS::ParametricPolyline::_CalculateNormals(vtkPolyData* curve)
{
    curve->BuildCells();
    curve->BuildLinks();
    vtkNew<vtkDoubleArray> normals;
    normals->SetNumberOfComponents(3);
    normals->SetNumberOfTuples(curve->GetNumberOfPoints());
    //normals->Allocate();
    curve->GetPointData()->SetNormals(normals);
    vtkNew<vtkIdList> cellIds;
    vtkNew<vtkIdList> ptIds;

    //for each point, get the line-segments connected to it.
    double up[3] = { 0,0,1 };
    for (vtkIdType pId = 0; pId < curve->GetNumberOfPoints(); ++pId)
    {
        cellIds->Reset();
        curve->GetPointCells(pId, cellIds);

        // for each line segment compute the normal.
        double n[2][3] = { {0,0,0}, {0,0,0} };
        for (vtkIdType i = 0; i < cellIds->GetNumberOfIds(); ++i)
        {
            vtkIdType cId = cellIds->GetId(i);
            if (curve->GetCellType(cId) != VTK_LINE)
                continue;

            ptIds->Reset();
            curve->GetCellPoints(cId, ptIds);
            MacroAssert(ptIds->GetNumberOfIds() == 2);

            double p0[3], p1[3];
            curve->GetPoint(ptIds->GetId(0), p0);
            curve->GetPoint(ptIds->GetId(1), p1);

            double v[3] = { 0,0,0 };
            vtkMath::Subtract(p1, p0, v);
            vtkMath::Normalize(v);
            vtkMath::Cross(v, up, n[i]);
            vtkMath::Normalize(n[i]);
        }

        // calculate average normal.
        double avg[3] = { 0,0,0 };
        vtkMath::Add(n[0], n[1], avg);
        double scaling = 1.0 / double(cellIds->GetNumberOfIds());
        vtkMath::MultiplyScalar(avg, scaling);

        // set as point normal.
        normals->SetTuple(pId, avg);
    }

    return 1;
}

void ParametricPolyline::_TotalCross(double total[3])
{
    double p0[3], p1[3], p2[3];
    vtkIdType maxId = m_curve->GetNumberOfPoints();
    for (vtkIdType ptId = 0; ptId < maxId; ++ptId)
    {
        m_curve->GetPoints()->GetPoint(ptId, p0);
        m_curve->GetPoints()->GetPoint((ptId+1)%maxId, p1);
        m_curve->GetPoints()->GetPoint((ptId+2)%maxId, p2);
        double v1[3], v2[3];
        vtkMath::Subtract(p1, p0, v1);
        vtkMath::Subtract(p2, p1, v2);
        vtkMath::Normalize(v1);
        vtkMath::Normalize(v2);
        double c[3];
        vtkMath::Cross(v1, v2, c);
        //vtkMath::Add(total, c, total);
        total[2] += asin(c[2]);
    }
}

void ParametricPolyline::_Flip()
{
    vtkNew<vtkPolyLine> newCurve;
    vtkIdType numOfPts = m_curve->GetPoints()->GetNumberOfPoints();
    for (vtkIdType id = numOfPts - 1; id >= 0; --id)
    {
        newCurve->GetPoints()->InsertNextPoint(m_curve->GetPoints()->GetPoint(id));
        //newCurve->GetPointIds()->InsertNextId(newId);
    }
    newCurve->GetPointIds()->Initialize();
    newCurve->GetPointIds()->DeepCopy(m_curve->GetPointIds());
    m_curve = newCurve;
    _UpdateParametricForm();
}

void sjDS::ParametricPolyline::_UpdateParametricForm()
{
    double totalLength = 0.0;
    m_positionToID.clear();

    m_positionToID.insert(std::make_pair(0, 0));
    vtkIdType numOfPts = m_curve->GetNumberOfPoints();
    double p0[3], p1[3];
    for (vtkIdType id = 0; id < numOfPts-1; ++id)
    {
        vtkIdType p0_id = m_curve->GetPointId(id);
        vtkIdType p1_id = m_curve->GetPointId(id+1);
        m_curve->GetPoints()->GetPoint(p0_id, p0);
        m_curve->GetPoints()->GetPoint(p1_id, p1);
        double dist = sqrt(vtkMath::Distance2BetweenPoints(p0, p1));
        totalLength += dist;
        m_positionToID.insert(std::make_pair(totalLength, id+1));
    }

    m_totalLength = totalLength;
}

