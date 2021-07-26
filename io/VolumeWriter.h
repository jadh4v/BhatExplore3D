#pragma once

#include <QString>
#include "core/macros.h"

namespace sjDS {
    class Image;
    class Grid;
}

class VolumeWriter
{
public:
    VolumeWriter(const sjDS::Image* img);
    void SetFileName(const QString filename);
    int Write();

private:
    const sjDS::Image* m_input = nullptr;
    QString m_filename;

};