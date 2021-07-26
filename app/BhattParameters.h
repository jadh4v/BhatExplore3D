#pragma once

#include "core/macros.h"

class BhattParameters {
private:
    //const double alpha = 0.0004; // negative sign accomodates for negative K.
    double m_Alpha = 0.00005; // worked for microscopy data
    double m_StepSize = 0.2;
    double m_Del = 1.0;
    double m_NarrowBand = 3.0; // will be scaled to spacing
    const double cPi = 3.141592653589793;
    const double cEps = 1e-16;
    const int cNumOfThreads = 12;
    int m_NumOfIterations = 20000;
    int m_RecomputePhiIterations = 100;
    const int m_PrintIterations = 200;
    int m_DownSampleFactor = 2;
    unsigned short m_ScalarRange = 63;
    int m_AttributeDimensions = 1;
    double m_GlobalRange[2] = { 0.0, 255.0 };

public:
    void SetParameters(double alpha, double stepSize, double del, double narrow_band,
        int num_of_iterations, int recomputePhiIterations, int downSampleFactor, int attribDim = 1);

    size_t HistSize() const { return size_t(m_ScalarRange+1); }
    double ScalarRange() const { return double(m_ScalarRange); }
    size_t HistArraySize(int d) const { return (size_t)(pow(HistSize(), d)); }

    MacroConstRefMember(double, m_Alpha, Alpha)
    MacroConstRefMember(double, m_StepSize, StepSize)
    MacroConstRefMember(double, m_Del, Del)
    MacroConstRefMember(double, m_NarrowBand, NarrowBand)
    MacroSetMember(double, m_NarrowBand, NarrowBand)
    MacroConstRefMember(double, cPi, Pi)
    MacroConstRefMember(double, cEps, Eps)
    MacroConstRefMember(int, cNumOfThreads, NumOfThreads)
    MacroConstRefMember(int, m_NumOfIterations, NumOfIterations)
    MacroConstRefMember(int, m_RecomputePhiIterations, RecomputePhiIterations)
    MacroConstRefMember(int, m_DownSampleFactor, DownSampleFactor)
    MacroConstRefMember(int, m_AttributeDimensions, AttribDim)
    MacroConstRefMember(int, m_PrintIterations, PrintIterations)
    MacroGetMember(const double*, m_GlobalRange, GlobalRange)
    void SetGlobalRange(const double r[2]);

};