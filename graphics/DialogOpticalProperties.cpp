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

    m_surface_mode = new QCheckBox("Surface Rendering", this);
    QHBoxLayout* sliderLayout = new QHBoxLayout();
    sliderLayout->addWidget(m_surface_mode);
    m_visible = new QCheckBox("Visible", this);
    sliderLayout->addWidget(m_visible);
    main_layout->addLayout( sliderLayout );

    // Ok and Cancel Buttons
    m_okButton = new QPushButton("Ok", this);
    m_cancelButton = new QPushButton("Cancel", this);
    QHBoxLayout* buttonsLayout = new QHBoxLayout();
    buttonsLayout->addWidget(m_cancelButton);
    buttonsLayout->addWidget(m_okButton);
    main_layout->addLayout(buttonsLayout);

    // Connect necessary signals:
    QObject::connect( m_okButton,     SIGNAL(clicked(bool)), this, SLOT(accept()) );
    QObject::connect( m_cancelButton, SIGNAL(clicked(bool)), this, SLOT(reject()) );
}

QColor DialogOpticalProperties::GetColor() const
{
    QColor finalColor;
    //finalColor.setAlpha(m_alphaSlider->value());
    return finalColor;
}

bool DialogOpticalProperties::GetSurfaceMode() const
{
    return m_surface_mode->isChecked();
}

void DialogOpticalProperties::SetSurfaceMode(bool f)
{
    return m_surface_mode->setChecked(f);
}

bool DialogOpticalProperties::GetSegmentVisibility() const
{
    return m_visible->isChecked();
}

void DialogOpticalProperties::SetSegmentVisibility(bool f)
{
    return m_visible->setChecked(f);
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
