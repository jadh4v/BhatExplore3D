#pragma once

#include <vector>
#include <QString>
#include <QColor>
#include "DS/VoxelRow.h"
#include "vpTransferFunction.h"

class Bookmark
{
public:
    Bookmark() {}
    Bookmark(const std::vector<sjDS::VoxelRow>& regions, const QString& name );
    Bookmark(std::vector<sjDS::VoxelRow>&& region, const QString& name );

    void SetName(const QString& name);
    const QString& GetName() const;
    const std::vector<sjDS::VoxelRow>& GetRegion() const;
    const QColor& GetColor() const;
    void SetColor(const QColor&);
    vpTransferFunction GetTF() const;
    void SetTF(const vpTransferFunction& tf );
    void SetMode(uint mode);
    uint GetMode() const;
    bool IsChecked() const;
    void SetChecked(bool);
    size_t Size() const { return m_region.size(); }
    bool Write(QDataStream& stream);
    static Bookmark* Read(QDataStream& stream);

private:
    void init();

    std::vector<sjDS::VoxelRow> m_region;
    QString m_name;
    QColor m_color;
    vpTransferFunction m_tfunc;
    uint m_mode = 0;
    bool m_checked = false;
};
