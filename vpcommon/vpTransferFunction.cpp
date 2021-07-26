#include<vector>
#include<list>
#include<set>

#include <QFile>
#include <QTextStream>
#include <QString>
#include <QColor>
#include <QVector>
//#include <qiterator.h>
#include <QXmlStreamWriter>
#include <QXmlStreamReader>

#include <vtkMath.h>
#include <vtkColorSeries.h>
#include "colorbrewer.h"
#include "vpTransferFunction.h"
#include "core/macros.h"

typedef std::vector<vpTransferFunction::TFPoint> TFPointArray;

const QString vpTransferFunction::DefaultContextFile = "./context.xml";
const QString vpTransferFunction::DefaultPancreasFile = "./pancreas.xml";
const QString vpTransferFunction::DefaultCystFile = "./cyst.xml";
const QString vpTransferFunction::DefaultDuctFile = "./duct.xml";

using std::vector;
using std::list;

    const int vpTransferFunction::cPaletteNames[8] = {   vtkColorSeries::BREWER_QUALITATIVE_ACCENT,
                                                        vtkColorSeries::BREWER_QUALITATIVE_DARK2,
                                                        vtkColorSeries::BREWER_QUALITATIVE_SET2,
                                                        vtkColorSeries::BREWER_QUALITATIVE_PASTEL2,
                                                        vtkColorSeries::BREWER_QUALITATIVE_PASTEL1,
                                                        vtkColorSeries::BREWER_QUALITATIVE_SET1,
                                                        vtkColorSeries::BREWER_QUALITATIVE_PAIRED,
                                                        vtkColorSeries::BREWER_QUALITATIVE_SET3 
                                                    };

void vpTransferFunction::init()
{
    m_error_code = 0;
    m_name = "untitled";
    m_reference_range.first  = -1024;
    m_reference_range.second = 3072;
    m_points.clear();
}

vpTransferFunction::vpTransferFunction()
{
    init();
}

vpTransferFunction::vpTransferFunction(const std::pair<double,double>& range)
{
    init();
    m_reference_range = range;
}

vpTransferFunction::vpTransferFunction( QXmlStreamReader& xml )
{
    init();
    int success = read_xml_tf_table( xml );

    if( !success )
        m_error_code = -1; // construction failure
}

vpTransferFunction::vpTransferFunction(const vpTransferFunction &CopyFrom)
{
    init();

    for( TFPointArray::const_iterator iter = CopyFrom.Begin(); iter != CopyFrom.End(); iter++ )
        AddPoint( *iter );

    this->m_name = CopyFrom.m_name;
    this->m_reference_range = CopyFrom.m_reference_range;
}

bool vpTransferFunction::valid_xml_attribs( const QXmlStreamAttributes& attribs )
{
    if( attribs.hasAttribute("point") && attribs.hasAttribute("red") &&
        attribs.hasAttribute("green") && attribs.hasAttribute("blue") &&
        attribs.hasAttribute("alpha") )
        return true;
    else
        return false;
}

int vpTransferFunction::xml_read_element_transfer_function( QXmlStreamReader& reader )
{
    int ret = 1;
    while( !reader.atEnd() )
    {
        reader.readNext();
        QString name = reader.name().toString();

        // Break when the current TransferFunction Element is completely read.
        if( reader.isEndElement() && name == "TransferFunction")
            break;

        //if( reader.isStartElement() && name == "range_begin" )
        //{
        //    m_reference_range.first = reader.attributes().value("range_begin").toDouble();
        //}
        //else if( reader.isStartElement() && name == "range_end" )
        //{
        //    m_reference_range.second = reader.attributes().value("range_end").toDouble();
        //}
        if( reader.isStartElement() && name == "TFPoint" )
        {
            QXmlStreamAttributes attribs = reader.attributes();
            if( !valid_xml_attribs(attribs) )
            {
                MacroWarning("Error reading xml: skipping the TF Point");
                ret = 0;
                continue;
            }

            TFPoint tfpoint;
            bool ok[5] = {false, false, false, false, false};
            tfpoint.x = attribs.value("point").toFloat( &ok[0] );
            tfpoint.color.r = attribs.value("red").toFloat( &ok[1] );
            tfpoint.color.g = attribs.value("green").toFloat( &ok[2] );
            tfpoint.color.b = attribs.value("blue").toFloat( &ok[3] );
            tfpoint.color.a = attribs.value("alpha").toFloat( &ok[4] );

            if( ok[0] && ok[1] && ok[2] && ok[3] && ok[4] )
                AddPoint( tfpoint );
            else
            {
                MacroWarning("cannot read \"TFPoint\" element.");
                continue;
            }
        }
    }

    return ret;
}

int vpTransferFunction::read_xml_tf_table( QXmlStreamReader& reader )
{
    int ret_success = 0;
    while( !reader.atEnd() )
    {
        reader.readNext();
        if( reader.isStartDocument() )
            continue;

        QString tok = reader.name().toString();
        if( reader.isStartElement() && tok == "TransferFunction" )
        {
            if( reader.attributes().hasAttribute("name") )
            {
                m_name = reader.attributes().value("name").toString();
            }
            if( reader.attributes().hasAttribute("range_begin") )
            {
                m_reference_range.first = reader.attributes().value("range_begin").toDouble();
            }
            if( reader.attributes().hasAttribute("range_end") )
            {
                m_reference_range.second = reader.attributes().value("range_end").toDouble();
            }

            ret_success = xml_read_element_transfer_function( reader );
            break; // read only one XML TF Element
        }
    }

    return ret_success;
}

vpTransferFunction & vpTransferFunction::operator=(const vpTransferFunction & other)
{
    this->m_error_code = other.m_error_code;
    this->m_name = other.m_name;
    this->m_reference_range = other.m_reference_range;
    this->m_points = other.m_points;
    return *this;
}

void vpTransferFunction::AddPoint(const vpTransferFunction::TFPoint &point)
{
    /*
    MacroConfirm( point.x >= 0 && point.x <= 1 );
    MacroConfirm( point.color.r >= 0 && point.color.r <= 1 );
    MacroConfirm( point.color.g >= 0 && point.color.g <= 1 );
    MacroConfirm( point.color.b >= 0 && point.color.b <= 1 );
    MacroConfirm( point.color.a >= 0 && point.color.a <= 1 );
    */
    vtkMath::ClampValue(point.x, 0.0f, 1.0f);
    vtkMath::ClampValue(point.color.r, 0.0f, 1.0f);
    vtkMath::ClampValue(point.color.g, 0.0f, 1.0f);
    vtkMath::ClampValue(point.color.b, 0.0f, 1.0f);
    vtkMath::ClampValue(point.color.a, 0.0f, 1.0f);

    m_points.push_back( point );
}

void vpTransferFunction::Write(QTextStream &txtStream) const
{
    for(TFIter iter = m_points.begin(); iter != m_points.end(); iter++ )
    {
        TFPoint point = *iter;
        txtStream << point.x << " ";
        txtStream << point.color.r << " ";
        txtStream << point.color.g << " ";
        txtStream << point.color.b << " ";
        txtStream << point.color.a;
        txtStream << '\n';
    }

    txtStream.flush();
}

void vpTransferFunction::Write(const QString &filename) const
{
    QFile tfFile( filename );
    tfFile.open( QFile::WriteOnly | QFile::Text );

    if( !tfFile.isOpen() || !tfFile.isWritable() )
    {
        MacroWarning("Cannot write transfer function file: " << filename.toLatin1().data() );
        return;
    }

    QTextStream txtStream( &tfFile );
    Write(txtStream);
    tfFile.close();
}

void vpTransferFunction::WriteXML( QXmlStreamWriter& xml ) const
{
//    xml.setAutoFormatting(true);
//    xml.writeStartDocument();
    xml.writeStartElement("TransferFunction");
    xml.writeAttribute("name", m_name);
    xml.writeAttribute("range_begin", QString::number(m_reference_range.first));
    xml.writeAttribute("range_end", QString::number(m_reference_range.second));

    for(TFIter iter = m_points.begin(); iter != m_points.end(); iter++ )
    {
        xml.writeStartElement("TFPoint");
        TFPoint point = *iter;

        xml.writeAttribute("point", QString::number(point.x) );
        xml.writeAttribute("red",   QString::number(point.color.r) );
        xml.writeAttribute("green", QString::number(point.color.g) );
        xml.writeAttribute("blue",  QString::number(point.color.b) );
        xml.writeAttribute("alpha", QString::number(point.color.a) );

        xml.writeEndElement(); // TFPoint
    }

    xml.writeEndElement(); // TransferFunction

    //xml.writeEndDocument();
}

double vpTransferFunction::calc_parametric_coordinate(const double t_normalized, const double adjustedRange[2], const double referenceRange[2])
{
    if(referenceRange[0] == referenceRange[1] )
        return 0.0;

    double t_forAdjustedRange = t_normalized*(adjustedRange[1]-adjustedRange[0]) + adjustedRange[0];
    double t_forReferenceRange = vtkMath::ClampValue(t_forAdjustedRange, referenceRange[0], referenceRange[1]);
    double t = (t_forReferenceRange - referenceRange[0]) / (referenceRange[1] - referenceRange[0]);
    return t;
}

void vpTransferFunction::SampleFunction(const int numOfSamples, const double adjustedRange[2], float *sampleBuffer) const
{
    NullCheckVoid(sampleBuffer);
    MacroAssert(adjustedRange[1] > adjustedRange[0]);
    double refRange[2] = {m_reference_range.first, m_reference_range.second};
    for(int i=0; i <= (numOfSamples-1); i++)
    {
        double t_normalized = double(i)/double(numOfSamples-1);
        double t = calc_parametric_coordinate(t_normalized, adjustedRange, refRange);
        RGBAColor color = sample_color(t);
        sampleBuffer[i*4+0] = color.r;
        sampleBuffer[i*4+1] = color.g;
        sampleBuffer[i*4+2] = color.b;
        sampleBuffer[i*4+3] = color.a;
    }
}

void vpTransferFunction::SampleFunction(const int numOfSamples, const double adjustedRange[2], uint *sampleBuffer) const
{
    //NullCheckVoid(sampleBuffer);
    //MacroAssert(adjustedRange[1] > adjustedRange[0]);
    double refRange[2] = {m_reference_range.first, m_reference_range.second};
    for(int i=0; i <= numOfSamples-1; i++)
    {
        double t_normalized = double(i)/double(numOfSamples-1);
        double t = calc_parametric_coordinate(t_normalized, adjustedRange, refRange);
        RGBAColor color = sample_color(t);
        uint color32bit = (uint(color.r*255) << 24) + (uint(color.g*255) << 16) + (uint(color.b*255) << 8) + uint(color.a*255);
        sampleBuffer[i] = color32bit;
    }
}

void vpTransferFunction::SampleOpacityFunction(const int numOfSamples, const double adjustedRange[2], float *sampleBuffer) const
{
    double refRange[2] = {m_reference_range.first, m_reference_range.second};
    for(int i=0; i <= numOfSamples-1; i++)
    {
        double t_normalized = double(i)/double(numOfSamples-1);
        double t = calc_parametric_coordinate(t_normalized, adjustedRange, refRange);
        RGBAColor color = sample_color(t);
        sampleBuffer[i] = color.a;
    }
}

double vpTransferFunction::SampleOpacity(const double adjustedRange[2], const double value) const
{
    //MacroAssert(adjustedRange[1] > adjustedRange[0]);
    double refRange[2] = {m_reference_range.first, m_reference_range.second};
    double t_normalized = (value - adjustedRange[0]) / (adjustedRange[1] - adjustedRange[0]);
    double t = calc_parametric_coordinate(t_normalized, adjustedRange, refRange);
    RGBAColor color = sample_color(t);
    return (double)color.a;
}

TFPointArray::const_iterator vpTransferFunction::Begin() const
{
    return m_points.begin();
}

TFPointArray::const_iterator vpTransferFunction::End() const
{
    return m_points.end();
}

size_t vpTransferFunction::Size() const
{
    return m_points.size();
}

vpTransferFunction::RGBAColor vpTransferFunction::sample_color(double x) const
{
    MacroAssert( 0 <= x && x <= 1 );
    TFPoint tmp;
    tmp.x = x;
    TFPointArray::const_iterator iter = std::lower_bound( m_points.begin(), m_points.end(), tmp );

    RGBAColor color;
    if( iter == m_points.end() )
    {
        return color;
    }
    else if( iter == m_points.begin() )
    {
        color = iter->color;
        return color;
    }

    RGBAColor color1 = iter->color;
    float x1 = iter->x;
    iter--;
    RGBAColor color0 = iter->color;
    float x0 = iter->x;

    float w = x1 - x0;
    float d = x  - x0;

    float factor = d/w;

    color.r = color0.r + (color1.r - color0.r) * factor;
    color.g = color0.g + (color1.g - color0.g) * factor;
    color.b = color0.b + (color1.b - color0.b) * factor;
    color.a = color0.a + (color1.a - color0.a) * factor;

    return color;
}

vpTransferFunction vpTransferFunction::AutoCompute2Color(const std::vector<float>& histogram,
    //float opacity,
    const std::pair<double, double>& scalar_reference_range,
    QColor firstColor, QColor secondColor)
{
    // Step 3: Construct histogram.
    vpTransferFunction ret(scalar_reference_range);

    vpTransferFunction::TFPoint tf_point1(0.05f, firstColor.redF(), firstColor.greenF(), firstColor.blueF(), 0.8);
    ret.AddPoint( tf_point1 );

    tf_point1.x = 0.49;
    tf_point1.color.a = 0.5;
    ret.AddPoint( tf_point1 );

    vpTransferFunction::TFPoint tf_point2(0.51f, secondColor.redF(), secondColor.greenF(), secondColor.blueF(), 0.2);
    ret.AddPoint( tf_point2 );

    tf_point2.x = 0.95f;
    ret.AddPoint( tf_point2 );

    return ret;
}

vpTransferFunction vpTransferFunction::AutoCompute(const std::vector<float>& histogram, 
    float opacityRange[2], 
    const std::pair<double,double>& scalar_reference_range,
    int maxFeatureCount, 
    float persistenceThreshold,
    QColor forceColor)
{
    vpTransferFunction ret;
    vpTransferFunction::TFPoint tf_point( 0.0f, 0.25f, 0.25f, 0.8f, 0.5f );
    ret.AddPoint( tf_point );
    tf_point.x = 0.5f;
    ret.AddPoint( tf_point );
    tf_point.x = 1.0f;
    ret.AddPoint( tf_point );

    if(histogram.size() <= 2)
    {
        MacroWarning("Need a better histogram.");
        return ret;
    }
    if( opacityRange[0] < 0 || opacityRange[1] > 1 )
    {
        MacroWarning("Need opacity range between 0 and 1. Clamping.");
        opacityRange[0] = vtkMath::ClampValue( opacityRange[0], 0.0f, 1.0f );
        opacityRange[1] = vtkMath::ClampValue( opacityRange[1], 0.0f, 1.0f );
        if( opacityRange[0] > opacityRange[1] )
            std::swap( opacityRange[0], opacityRange[1] );
    }
    if( maxFeatureCount < 3 )
    {
        MacroWarning("Need feature count to be at least 3. Assuming 3.");
        maxFeatureCount = 3;
    }

    // Useful structures and definitions
    class CriticalPoint
    {
        public:
            enum Type{TypeMin=0, TypeMax=1};
            const size_t cInvalidID = SIZE_MAX;
            size_t id = cInvalidID;
            Type type = TypeMin;
            float value = 0.0f;
            float persistence = float(UINT_MAX);
            bool operator==(const CriticalPoint& other) const {
                return this->id == other.id;
            }
            bool operator!=(const CriticalPoint& other) const {
                return this->id != other.id;
            }
    };

    // Array of critical points.
    list<CriticalPoint> cps;

    // Step 1: COMPUTE CRITICAL POINTS
    // step 1.1 gather critical points
    for(size_t i=0; i < histogram.size(); ++i)
    {
        CriticalPoint p;
        p.id = i;
        p.value = histogram[i];

        // if first point
        if(i==0)
        {
            p.type = histogram[0] < histogram[1]? CriticalPoint::TypeMin : CriticalPoint::TypeMax;
            cps.push_back(p);
        }
        // if last point
        else if(i == histogram.size()-1)
        {
            p.type = histogram[i] < histogram[i-1]? CriticalPoint::TypeMin : CriticalPoint::TypeMax;
            cps.push_back(p);
        }
        else // if internal point
        {
            if( histogram[i] < histogram[i-1] && histogram[i] < histogram[i+1] )
            {
                p.type = CriticalPoint::TypeMin;
                cps.push_back(p);
            }
            else if( histogram[i] > histogram[i-1] && histogram[i] > histogram[i+1] )
            {
                p.type = CriticalPoint::TypeMax;
                cps.push_back(p);
            }
            // point is not critical and not inserted in cps.
        }
    }

    if( cps.empty() )
        return ret;

    if( cps.size() < 3 )
    {
        MacroWarning("Error computing critical points of 1D histogram curve.");
        return ret;
    }

    // step 1.2 Compute persistence
    for( auto curr = cps.begin(); curr != cps.end(); ++curr )
    {
        auto next = curr;
        ++next;
        if( next == cps.end() )
            break;

        curr->persistence = fabs( curr->value - next->value );
    }

    // Step 2: SIMPLIFY CURVE
    auto min_elem = std::min_element( cps.begin(), cps.end(), [](const CriticalPoint& a, const CriticalPoint&b){ return (a.persistence < b.persistence); } );
    float minPersist = min_elem->persistence;
    while( cps.size() > 3 && (cps.size() > maxFeatureCount || minPersist < persistenceThreshold) )
    {
        MacroAssert( *min_elem != cps.back() );
        auto next = min_elem;
        next++;
        if( min_elem == cps.begin() )
        {
            min_elem->persistence = fabs(next->persistence - min_elem->persistence);
            min_elem->type = next->type; // TODO : is this correct?
            cps.erase( next );
        }
        else
        {
            auto prev = min_elem;
            prev--;
            prev->persistence = fabs( prev->persistence + next->persistence - min_elem->persistence );
            cps.erase( min_elem );
            cps.erase( next );
        }

        min_elem = std::min_element( cps.begin(), cps.end(), [](const CriticalPoint& a, const CriticalPoint&b){ return (a.persistence < b.persistence); } );
        minPersist = min_elem->persistence;
    }

    // re-adjust location of first and last node in the curve.
    cps.front().id = 0;
    cps.back().id = histogram.size()-1;

    // Step 3: Construct histogram.
    ret = vpTransferFunction(scalar_reference_range);

    std::vector<int> colorNums;
    while(colorNums.size() < cps.size() )
    {
        int colNum = rand()%16;
        colorNums.push_back( colNum );
    }

    MacroNewVtkObject( vtkColorSeries, color_series );
    int paletteNum = rand()%8;
    color_series->SetColorScheme( cPaletteNames[ rand() % (sizeof(cPaletteNames)/sizeof(cPaletteNames[0])) ] );

    size_t i=0;
    for( auto& c : cps )
    {
        float pos = float(c.id) / float(histogram.size()-1);
        if( pos == 0 ) pos += 1e-6;
        if( pos == 1 ) pos -= 1e-6;

        vtkColor3ub brewed_color = color_series->GetColorRepeating( colorNums[i] );
        if( c.type == CriticalPoint::TypeMin )
            brewed_color = vtkColor3ub(255,255,255); 

        float opacity = vtkMath::ClampValue( c.value, opacityRange[0], opacityRange[1] );

        //tf_point = vpTransferFunction::TFPoint( pos, 0.25f, 0.25f, 0.8f, opacity );
        float denom = 300.0f;
        if (forceColor.isValid() && forceColor != QColor(0,0,0,0))
        {
            tf_point = vpTransferFunction::TFPoint(pos,
                forceColor.red() / denom, forceColor.green() / denom, forceColor.blue() / denom, opacity);
        }
        else
        {
            tf_point = vpTransferFunction::TFPoint(pos,
                brewed_color.GetRed() / denom, brewed_color.GetGreen() / denom, brewed_color.GetBlue() / denom, opacity);
        }

        ret.AddPoint( tf_point );
        i++;
    }

    return ret;
}

vpTransferFunction::vpTransferFunction(const QString& xml_filename)
{
    init();
    //std::cout << "Reading TF xml:" << xml_filename.toLatin1().constData() << std::endl;
    if( !QFile::exists(xml_filename) )
    {
        MacroWarning( "File not found: " << xml_filename.toLatin1().constData() );
        return;
    }

    QFile xml_file( xml_filename );
    xml_file.open( QFile::ReadOnly | QFile::Text );

    if( !xml_file.isOpen() || !xml_file.isReadable() )
    {
        MacroWarning("Cannot open transfer function file: " << xml_filename.toLatin1().constData() );
        return;
    }

    QXmlStreamReader reader( &xml_file );
    while( !reader.atEnd() )
    {
        reader.readNext();
        if( reader.hasError() )
        {
            MacroWarning("XML read error.");
            MacroWarning(reader.errorString().toLatin1().constData());
        }

        if( reader.isEndDocument() )
            continue;

        vpTransferFunction tmp(reader);
        if (tmp.GetErrorCode() == 0)
        {
            *this = tmp;
            break;
        }
        /*
        // Confirm that construction from xml file was successful:
        if( this->GetErrorCode() != 0 )
        {
            MacroWarning("Error reading xml transfer-function file.");
        }
        */
    }

    xml_file.close();
}

void vpTransferFunction::SetReferenceRange(const std::pair<double, double>& range)
{
    //rescale(range);
    m_reference_range = range;
}

void vpTransferFunction::SetReferenceRange(double start, double end)
{
    SetReferenceRange(std::make_pair(start,end));
}

void vpTransferFunction::SetReferenceRange(const double range[2])
{
    SetReferenceRange(std::make_pair(range[0],range[1]));
}

std::pair<double,double> vpTransferFunction::GetReferenceRange() const
{
    return m_reference_range;
}

void vpTransferFunction::GetReferenceRange(double range[2]) const
{
    range[0] = m_reference_range.first;
    range[1] = m_reference_range.second;
}

void vpTransferFunction::rescale(const std::pair<double, double>& range)
{
    double old_diff = abs(m_reference_range.first - m_reference_range.second);
    double new_diff = abs(range.first - range.second);
    if (old_diff == new_diff)
    {
        return;
    }

    double scaling_factor = old_diff / new_diff;
    for (auto& p : m_points)
    {
        p.x *= scaling_factor;
    }
}