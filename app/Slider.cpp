#include "Slider.h"
#include <QBoxLayout>
#include <QLabel>
#include <QSlider>

Slider::~Slider()
{
}

Slider::Slider(
    const QString& displayName,
    Qt::Orientation orientation,
    int minValue,
    int maxValue,
    QWidget* parent) : QWidget(parent)
{
    if (orientation == Qt::Horizontal) {
        this->setLayout(new QHBoxLayout());
    } else {
        this->setLayout(new QVBoxLayout());
    }
    m_property = new QLabel(displayName);
    m_slider = new QSlider(orientation);
    m_slider->setRange(minValue, maxValue);
    m_value = new QLabel();
    this->layout()->addWidget(m_property);
    this->layout()->addWidget(m_slider);
    this->layout()->addWidget(m_value);
    QObject::connect(m_slider, SIGNAL(valueChanged(int)), m_value, SLOT(setNum(int)));
    QObject::connect(m_slider, SIGNAL(valueChanged(int)), this, SLOT(slot_valueChanged(int)));
    m_slider->setValue((minValue + maxValue) / 2);
}

int Slider::GetValue() const
{
    return m_slider->value();
}

void Slider::SetValue(int value) const
{
    m_slider->setValue(value);
}
