#include <iostream>
#include <QColorDialog>
#include <QCheckBox>
#include <QSlider>
#include <QLabel>
#include <QPushButton>
#include <QLayout>
#include "DialogOpticalProperties.h"
#include "vpTransferFunction.h"
#include "vpTransferFunctionEditor.h"
#include "core/macros.h"

DialogOpticalProperties::~DialogOpticalProperties()
{
}

DialogOpticalProperties::DialogOpticalProperties(const vpTransferFunction& tfunc, QWidget* parent,
                                     Qt::WindowFlags flags) : QDialog(parent, flags)
{
    this->setModal(false);
    QVBoxLayout* main_layout = new QVBoxLayout();
    this->setLayout(main_layout);
    m_tfEditor = new vpTransferFunctionEditor( &tfunc, this);
    main_layout->addWidget( m_tfEditor );

    // Connect necessary signals:
    QObject::connect( m_tfEditor, SIGNAL(sign_TransferFunctionUpdated(const vpTransferFunction*)), this, SIGNAL(sign_TransferFunctionUpdated(const vpTransferFunction*)));
}

QColor DialogOpticalProperties::GetColor() const
{
    QColor finalColor;
    //finalColor.setAlpha(m_alphaSlider->value());
    return finalColor;
}

void DialogOpticalProperties::slot_finished(int result)
{
    if( result == QDialog::Accepted )
        this->accept();
    else
        this->reject();
}

vpTransferFunction DialogOpticalProperties::GetTF() const
{
    vpTransferFunction ret;
    m_tfEditor->GetTransferFunction(ret);
    return ret;
}

void DialogOpticalProperties::SetHistogram(const std::vector<float> &hist)
{
    m_tfEditor->SetHistogram( hist );
}

void DialogOpticalProperties::slot_setTransferFunction(const vpTransferFunction& tfunc)
{
    this->layout()->removeWidget(m_tfEditor);
    MacroDelete(m_tfEditor);
    m_tfEditor = new vpTransferFunctionEditor( &tfunc, this);
    this->layout()->addWidget( m_tfEditor );
    // Connect necessary signals:
    QObject::connect( m_tfEditor, SIGNAL(sign_TransferFunctionUpdated(const vpTransferFunction*)), this, SIGNAL(sign_TransferFunctionUpdated(const vpTransferFunction*)));
}
