#ifndef VPTRANSFERFUNCTION_H
#define VPTRANSFERFUNCTION_H

#include <vector>
#include <QString>
#include <QXmlStreamReader>
#include <QColor>
#include "core/macros.h"

class QFile;
class QTextStream;
class QXmlStreamAttributes;

class vpTransferFunction
{

public:
    static const int cPaletteNames[8];

    struct RGBAColor
    {
    public:
        float r, g, b, a;

        RGBAColor() { r=0; g=0; b=0; a=0; }
        RGBAColor(float r, float g, float b, float a )
        {
            this->r = r; this->g = g; this->b = b; this->a = a;
        }
    };

    struct TFPoint
    {
    public:
        float x;
        RGBAColor color;

        TFPoint() { x = 0; }

        TFPoint(float x, float r, float g, float b, float a)
        {
            this->x = x; 
            this->color = RGBAColor( r, g, b, a );
        }

        bool operator<(const TFPoint & A) const
        {
            return (this->x < A.x);
        }
    };

    typedef std::vector<vpTransferFunction::TFPoint> TFPointArray;
    typedef std::vector<vpTransferFunction::TFPoint>::const_iterator TFIter;

    static const QString DefaultContextFile;
    static const QString DefaultPancreasFile;
    static const QString DefaultCystFile;
    static const QString DefaultDuctFile;

public:
    /// Empty Constructor.
    vpTransferFunction();
    vpTransferFunction(const std::pair<double,double>& range);

    /// Copy Constructor.
    vpTransferFunction(const vpTransferFunction& CopyFrom );

    /// construct by opening an xml file.
    vpTransferFunction(const QString& xml_filename);

    /// Construct from an XML Stream.
    vpTransferFunction( QXmlStreamReader& xml );

    /// Copy assignment operator
    vpTransferFunction& operator=(const vpTransferFunction& other);

    void AddPoint(const TFPoint& point);

    void SetReferenceRange(const std::pair<double,double>& range);
    void SetReferenceRange(double start, double end);
    void SetReferenceRange(const double range[2]);
    std::pair<double,double> GetReferenceRange() const;
    void GetReferenceRange(double range[2]) const;

    void Write( const QString& filename ) const;
    void Write(QTextStream &txtStream) const;
    void WriteXML( QXmlStreamWriter& xml ) const;

    void SampleFunction( const int numOfSamples, const double adjustedRange[2], float* sampleBuffer ) const;
    void SampleOpacityFunction( const int numOfSamples, const double adjustedRange[2], float* sampleBuffer ) const;
    void SampleFunction( const int numOfSamples, const double adjustedRange[2], uint* sampleBuffer ) const;
    double SampleOpacity(const double adjustedRange[2], const double value) const;

    TFIter Begin() const;

    TFIter End() const;

    size_t Size() const;

    MacroGetSetMember( QString, m_name, Name )
    MacroGetMember( int, m_error_code, ErrorCode )

    /// Auto compute a transfer function from a given "normalized" histogram.
    /// opacityRange -- specify minimum and maximum opacity that will be used for lowest regions and highest regions of the TF.
    /// maxFeatureCount -- specify the maximum number of features to recognize in the simplified 1D histogram curve.
    ///                    simplification will be done based on persistence-based topology.
    static vpTransferFunction AutoCompute(const std::vector<float>& histogram, 
                                          float opacityRange[2], 
                                          const std::pair<double,double>& scalar_reference_range,
                                          int maxFeatureCount=3, 
                                          float persistenceThreshold=0.2, QColor forceColor=QColor(0,0,0,0) );

    static vpTransferFunction AutoCompute2Color(const std::vector<float>& histogram,
                                          //float opacity,
                                          const std::pair<double, double>& scalar_reference_range,
                                          QColor firstColor, QColor secondColor);

    //MEMBER VARIABLES
private:
    void init();
    void rescale(const std::pair<double, double>& range);
    RGBAColor sample_color(double x) const;
    int  read_xml_tf_table( QXmlStreamReader& reader );
    bool valid_xml_attribs( const QXmlStreamAttributes& attribs );
    int  xml_read_element_transfer_function( QXmlStreamReader& reader );
    static double calc_parametric_coordinate(const double t_normalized, const double adjustedRange[2], const double referenceRange[2]);

    QString m_name;
    std::vector<TFPoint> m_points;
    int m_error_code;
    std::pair<double,double> m_reference_range;

};

#endif // VPTRANSFERFUNCTION_H
