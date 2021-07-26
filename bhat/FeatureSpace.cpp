//Std
#include <numeric>

// Eigen
#include <Eigen/Dense>

// VTK
#include <vtkDataArray.h>
#include <vtkMath.h>
#include <vtkImageData.h>
#include <vtkPointData.h>

// Proj
#include <core/macros.h>
#include "NGrid.h"
#include "FeatureSpace.h"

using Bhat::FeatureSpace;
using Bhat::NGrid;

#define TEMPLATE_SIGN template<size_t _Dim, typename _Real>
#define CLASS_SIGN FeatureSpace<_Dim, _Real>

TEMPLATE_SIGN
CLASS_SIGN::FeatureSpace()
{
    if (_Dim > 3)
        MacroWarning("Feature space dimensions greater than 3 not supported.");
}


TEMPLATE_SIGN
CLASS_SIGN::FeatureSpace(const std::vector<Point>& voxel_features, const sjDS::Bitvector& mask)
{
    if (_Dim > 3)
        MacroWarning("Feature space dimensions greater than 3 not supported.");

    m_points = voxel_features;
    //_Normalize(m_points);
    m_mask = mask;
}

TEMPLATE_SIGN
void CLASS_SIGN::Push(Point z_p)
{
    m_points.push_back(z_p);
}

TEMPLATE_SIGN
double CLASS_SIGN::GetVariance(size_t d) const
{
    double r(0);

    // Early return if queried dimension is invalid.
    if (!_ValidDimension(d))
    {
        MacroWarning("Invalid dimension: " << d);
        return r;
    }

    std::vector<_Real> v(m_points.size());
    std::transform(m_points.begin(), m_points.end(), v.begin(), [d](Point x) { return x[d]; });

    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    std::vector<double> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    //double stdev = std::sqrt(sq_sum / v.size());
    r = sq_sum / v.size();

    return r;
}

TEMPLATE_SIGN
void CLASS_SIGN::BuildKernel()
{
    m_sigma.clear();
    double numPts = (double)m_points.size();
    for (size_t i = 0; i < _Dim; ++i)
    {
        double variance = GetVariance(i);
        double s = std::sqrt(variance) * pow(numPts, -0.2);
        s = s * 12.0;
        std::cout << "sigma[" << i << "] = " << s << std::endl;
        m_sigma.push_back((_Real)s);
        //m_sigma.push_back(0.1);
    }
}

TEMPLATE_SIGN
bool CLASS_SIGN::_ValidDimension(size_t d) const
{
    return (d < _Dim);
}

TEMPLATE_SIGN
void CLASS_SIGN::_Normalize(std::vector<Point>& points)
{
    Point init;
    init.setZero();

    Point max_values = std::accumulate(points.begin(), points.end(), init, 
        [](const Point& x, const Point& y) {
            Point ret;
            for(int i=0; i < _Dim; ++i)
                ret(i) = (x(i) > y(i) ? x(i) : y(i));

            return ret;
    });

    for (auto& p : points)
    {
        for(int i=0; i < _Dim; ++i)
            p[i] /= max_values[i];
    }
}

TEMPLATE_SIGN
void CLASS_SIGN::ComputeP1()
{
    NGrid<_Dim,_Real> grid(m_modelSampling);
    size_t numGridPoints = grid.NumberOfPoints();

    m_Pout.clear();
    m_Pout.resize(numGridPoints);
    m_Pin.clear();
    m_Pin.resize(numGridPoints);

    m_Ain = m_Aout = 0;
    for (size_t x = 0; x < m_points.size(); ++x)
    {
        if (m_mask.Get(x))
            ++m_Ain;
        else
            ++m_Aout;
    }

    size_t pId = 0;
    grid.StartIteration();
    while (1)
    {
        Point z = grid.NextGridPoint()*255;
        if (z[0] < 0)
            break;

        double p_out = 0, p_in = 0;
        //double p_out_area = 0, p_in_area = 0;
        for (size_t x=0; x < m_points.size(); ++x)
        {
            Point d = z - m_points[x];
            if (m_mask.Get(x))
            {
                p_in += _Kernel(d);
                //++p_in_area;
            }
            else
            {
                p_out += _Kernel(d);
                //++p_out_area;
            }
        }

        m_Pout[pId] = _Real(p_out);
        m_Pin[pId] = _Real(p_in);
        ++pId;
    }

    _Real sum_in = std::accumulate(m_Pin.begin(), m_Pin.end(), 0);
    std::transform(m_Pin.begin(), m_Pin.end(), m_Pin.begin(), [sum_in](_Real value) { return value / sum_in; });

    _Real sum_out = std::accumulate(m_Pout.begin(), m_Pout.end(), 0);
    std::transform(m_Pout.begin(), m_Pout.end(), m_Pout.begin(), [sum_out](_Real value) { return value / sum_out; });
}

template<typename T>
std::vector<T>
conv(std::vector<T> const &f, std::vector<T> const &g) {
    int const nf = f.size();
    int const ng = g.size();
    int const n = nf + ng - 1;
    std::vector<T> out(n, T());
    for (auto i(0); i < n; ++i) {
        int const jmn = (i >= ng - 1) ? i - (ng - 1) : 0;
        int const jmx = (i < nf - 1) ? i : nf - 1;
        for (auto j(jmn); j <= jmx; ++j) {
            out[i] += (f[j] * g[i - j]);
        }
    }
    return out;
}

template<typename T>
std::vector<T> my_conv( const std::vector<T>& kernel, const std::vector<T>& data)
{
    MacroAssert(kernel.size() % 2 == 1); // kernel should have odd size
    size_t paddingSize = (kernel.size() - 1) / 2;
    std::vector<T> padded_data(data.size() + 2*paddingSize);
    std::vector<T> padded_out(data.size() + 2*paddingSize);

    std::copy(data.begin(), data.end(), padded_data.begin() + paddingSize);

    for (size_t i = paddingSize; i < padded_data.size()-paddingSize; ++i)
    {
        padded_out[i] = std::inner_product(kernel.begin(), kernel.end(), padded_data.begin() + i - paddingSize, T(0));
    }

    std::vector<T> out(data.size());
    std::copy(padded_out.begin() + paddingSize, padded_out.end() - paddingSize, out.begin());
    my_norm(out);
    return out;
}

template<typename T>
void my_norm(std::vector<T>& data)
{
    T sum = std::accumulate(data.begin(), data.end(), T(0));
    std::for_each(data.begin(), data.end(), [&sum](T& value) { value = value / sum; });
}

TEMPLATE_SIGN
void CLASS_SIGN::ComputeP2()
{
    m_Ain = m_Aout = 0;
    m_Pout.clear();
    m_Pout.resize(256);
    m_Pin.clear();
    m_Pin.resize(256);

    for (size_t x = 0; x < m_points.size(); ++x)
    {
        if (m_mask.Get(x))
        {
            ++m_Pin[m_points[x](0)];
            ++m_Ain;
        }
        else
        {
            ++m_Pout[m_points[x](0)];
            ++m_Aout;
        }
    }

    double area = m_Ain;
    std::transform(m_Pin.begin(), m_Pin.end(), m_Pin.begin(), [area](_Real value) { if (value == 0) value = 1; return value / area; });
    area = m_Aout;
    std::transform(m_Pout.begin(), m_Pout.end(), m_Pout.begin(), [area](_Real value) { if (value == 0) value = 1; return value / area; });

#if 0
    // convolution
    std::vector<_Real> binom{ 3.05175781250000e-05, 0.000457763671875000, 0.00320434570312500, 0.0138854980468750,
                                0.0416564941406250, 0.0916442871093750,   0.152740478515625,   0.196380615234375,
                                0.196380615234375,  0.152740478515625,    0.0916442871093750,  0.0416564941406250,
                                0.0138854980468750, 0.00320434570312500, 0.000457763671875000, 3.05175781250000e-05 }; 
    //std::vector<_Real> binom{ 0.1, 0.2, 0.4, 0.2, 0.1 };
    auto conv_in = conv(m_Pin, binom);
    size_t offset = (conv_in.size() - m_Pin.size()) / 2;
    for (size_t i = offset; i < offset + m_Pin.size(); ++i)
        m_Pin[i-offset] = conv_in[i];

    auto conv_out = conv(m_Pout, binom);
    for (size_t i = offset; i < offset + m_Pout.size(); ++i)
        m_Pout[i-offset] = conv_out[i];
#endif

#if 1
    //std::vector<_Real> binom{ 0.1, 0.2, 0.4, 0.2, 0.1 };
    std::vector<_Real> binom{ 0.0, 3.05175781250000e-05, 0.000457763671875000, 0.00320434570312500, 0.0138854980468750,
                                0.0416564941406250, 0.0916442871093750,   0.152740478515625,   0.196380615234375,
                                0.196380615234375,  0.152740478515625,    0.0916442871093750,  0.0416564941406250,
                                0.0138854980468750, 0.00320434570312500, 0.000457763671875000, 3.05175781250000e-05 }; 
    m_Pin = my_conv(binom, m_Pin);
    m_Pout = my_conv(binom, m_Pout);
#endif
}

TEMPLATE_SIGN
void CLASS_SIGN::ComputeL()
{
    MacroAssert(m_Pin.size() == m_Pout.size());
    m_L.clear();
    m_L.resize(m_Pin.size());
    double Ain_inv = 1.0 / m_Ain;
    double Aout_inv = 1.0 / m_Aout;
    for (size_t i = 0; i < m_Pin.size(); ++i)
    {
        m_L[i] = Aout_inv * std::sqrt(m_Pin[i] / (eps + m_Pout[i])) - Ain_inv * std::sqrt(m_Pout[i] / (eps + m_Pin[i]));
    }
}

TEMPLATE_SIGN
double CLASS_SIGN::_Kernel(Point d) const
{
    double ret = 1.0;
    if (m_sigma.empty())
        ret = 0.0;

    double pi = vtkMath::Pi();
    for (size_t i=0; i < m_sigma.size(); ++i)
    {
        double z2 = d(i)*d(i);
        double sigma2 = m_sigma[i] * m_sigma[i];
        double denom = std::sqrt(2.0 * pi * sigma2);
        double numer = std::exp(-z2 / (2.0 * sigma2));
        ret *= numer / denom;
    }
    return ret;
}

TEMPLATE_SIGN
double CLASS_SIGN::_Compute_V_1st_Term() const
{
    MacroAssert(m_Pin.size() == m_Pout.size());
    double B = 0;
    for (size_t i = 0; i < m_Pin.size(); ++i)
        B += std::sqrt(m_Pin[i] * m_Pout[i]);

    //std::cout << "B = " << B << std::endl;
    double areaTerm = 1.0 / m_Ain - 1.0 / m_Aout;
    double ret = 0.5 * B * areaTerm;
    return ret;
}

TEMPLATE_SIGN
void CLASS_SIGN::ComputeV(std::vector<_Real>& V, double del, double pi) const
{
    double firstTerm = _Compute_V_1st_Term();

    NGrid<_Dim,_Real> grid(m_modelSampling);
    //size_t numGridPoints = grid.NumberOfPoints();

    MacroAssert(V.size() == m_points.size());
    //V.clear();
    //V.resize(m_points.size());
    for (size_t x = 0; x < m_points.size(); ++x)
    {
        /*
        grid.StartIteration();
        std::vector<double> k;
        k.reserve(m_L.size());
        double sum_k = 0;
        while (1)
        {
            Point z_p = grid.NextGridPoint()*255;
            if (z_p[0] < 0)
                break;

            auto d = z_p - m_points[x];

            // TODO: implement multi-dimensional dirac function.
            double dirac = 0.5 / del * (1.0 + std::cos(pi * d(0) / del));
            if (d(0) > del || d(0) < -del)
                dirac = 0.0;

            //dirac = _Kernel(d);

            k.push_back(dirac);
            sum_k += dirac;
        }
        for (size_t z = 0; z < m_L.size(); ++z)
            k[z] /= sum_k;

        //double secondTerm = 0.0;
        //for (size_t z = 0; z < m_L.size(); ++z)
        //    secondTerm += k[z] * m_L[z];
        */
        double secondTerm = m_L[m_points[x](0)];

        vtkVariant value = firstTerm + 0.5*secondTerm;
        //dataArray->SetVariantValue(x, value);
        V[x] = _Real(value.ToDouble());
    }
}



template class Bhat::FeatureSpace<1, float>;
template class Bhat::FeatureSpace<2, float>;
template class Bhat::FeatureSpace<2, double>;
