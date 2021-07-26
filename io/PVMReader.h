#ifndef PVMREADER_H
#define PVMREADER_H

#include <QFile>
#include <QString>
#include "Core/Algorithm.h"
#include "ds/Image.h"

namespace hseg{

class PVMReader : public sjCore::Algorithm
{
public:
    PVMReader();
    PVMReader(const QString& filename);
    ~PVMReader();
    void SetFileName(const QString& filename);
    MacroGetMember(sjDS::Image, m_img, Output)

private:
    void init();

    virtual int input_validation() const;
    virtual int primary_process();
    virtual int post_processing();

    sjDS::Image m_img;
    QFile    m_file;
    QString  m_filename;
};

}

#endif
