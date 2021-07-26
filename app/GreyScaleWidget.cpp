#include <QBoxLayout>
#include <QLabel>
#include "Slider.h"
#include "GreyScaleWidget.h"

GreyScaleWidget::~GreyScaleWidget()
{
}

GreyScaleWidget::GreyScaleWidget(QWidget * parent) : QWidget(parent)
{
    this->DisplayName = new QLabel("GreyScale Widget");
    this->Range = new Slider("Range", Qt::Vertical, 0, 4096);
    this->Level = new Slider("Level", Qt::Vertical, 0, 4096);

    this->setLayout(new QHBoxLayout());
    this->layout()->addWidget(this->DisplayName);
    this->layout()->addWidget(this->Range);
    this->layout()->addWidget(this->Level);

    QObject::connect(this->Range, SIGNAL(sign_valueChanged(int)), this, SLOT(slot_RangeChanged(int)));
    QObject::connect(this->Level, SIGNAL(sign_valueChanged(int)), this, SLOT(slot_LevelChanged(int)));
}

void GreyScaleWidget::slot_RangeChanged(int range)
{
    int level = this->Level->GetValue();
    emit(sign_valueChanged(level, range));
}

void GreyScaleWidget::slot_LevelChanged(int level)
{
    int range = this->Range->GetValue();
    emit(sign_valueChanged(level, range));
}

void GreyScaleWidget::SetValues(int level, int range)
{
    this->Level->SetValue(level);
    this->Range->SetValue(range);
}

void GreyScaleWidget::GetValues(int & level, int & range) const
{
    level = this->Level->GetValue();
    range = this->Range->GetValue();
}

