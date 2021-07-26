#include <QDataStream>
#include <QColor>
#include "Bookmark.h"
#include "macros.h"

using std::vector;
using sjDS::VoxelRow;

void Bookmark::init()
{
    m_color = QColor(100,50,50,200); // default color.
    vpTransferFunction::TFPoint tf_point;
    tf_point.x = 0.1;
    tf_point.color.r = 0.25;
    tf_point.color.g = 0.25;
    tf_point.color.b = 0.8;
    tf_point.color.a = 0.5;
    //m_tfunc.AddPoint(tf_point);
    tf_point.x = 0.9;
    //m_tfunc.AddPoint(tf_point);
}

Bookmark::Bookmark(std::vector<VoxelRow>&& region, const QString& name ) : m_region(region)
{
    init();
    m_name = name;
}

Bookmark::Bookmark(const std::vector<VoxelRow>& region, const QString& name ) : m_region(region)
{
    init();
    m_name = name;
}

void Bookmark::SetName(const QString& name)
{
    m_name = name;
}

const QString& Bookmark::GetName() const
{
    return m_name;
}

bool Bookmark::IsChecked() const
{
    return m_checked;
}

void Bookmark::SetChecked(bool f)
{
    m_checked = f;
}

const std::vector<VoxelRow>& Bookmark::GetRegion() const
{
    return m_region;
}

const QColor & Bookmark::GetColor() const
{
    return m_color;
}

void Bookmark::SetColor(const QColor& color) 
{
    m_color = color;
}

vpTransferFunction Bookmark::GetTF() const
{
    return m_tfunc;
}

void Bookmark::SetTF(const vpTransferFunction & tf)
{
    m_tfunc = tf;
}

void Bookmark::SetMode(uint mode)
{
    m_mode = mode;
}

uint Bookmark::GetMode() const
{
    return m_mode;
}

bool Bookmark::Write(QDataStream& stream)
{
    if( stream.status() != QDataStream::Ok )
    {
        MacroWarning("Error writing to data stream.");
        return false;
    }

    if( m_region.empty() )
        return true;

    // Write name of bookmark
    //stream << (quint64)m_name.size();
    //stream.writeBytes( m_name.toLatin1().constData(), (uint)m_name.size() );
    stream << m_name;
    stream << m_color;
    stream << m_mode;
    QString tf_string;
    QXmlStreamWriter xml_writer(&tf_string);
    m_tfunc.WriteXML(xml_writer);
    stream << tf_string;
    // Write region Ids
    stream << (quint64) m_region.size();
    for( auto r: m_region)
    {
        stream << (quint32) r.Start();
        stream << (quint32) r.End();
    }

    return true;
}

Bookmark* Bookmark::Read(QDataStream& stream)
{
    if( stream.status() != QDataStream::Ok )
    {
        MacroWarning("Error writing to data stream.");
        return nullptr;
    }

    // Read name of bookmark:
    //uint nameSize = 0;
    //char* nameText = nullptr;
    //stream.readBytes( nameText, nameSize );
    //QString bookmarkName(nameText);
    //MacroDeleteArray(nameText);
    QString bookmarkName; 
    QColor color;
    uint mode;

    stream >> bookmarkName;
    stream >> color;
    stream >> mode;

    QString tf_string;
    stream >> tf_string;

    QXmlStreamReader xml_reader(tf_string);
    vpTransferFunction tf(xml_reader);

    // Read region Ids of bookmark:
    vector<VoxelRow> region;
    quint64 regIdsCount = 0;
    stream >> regIdsCount;
    for( quint64 i=0; i < regIdsCount; ++i )
    {
        quint32 start = 0, end = 0;
        stream >> start;
        stream >> end;
        region.push_back(VoxelRow(start,end));
    }

    Bookmark* ret = new Bookmark( region, bookmarkName );
    ret->SetColor(color);
    ret->SetMode(mode);
    ret->SetTF(tf);
    return ret;
}
