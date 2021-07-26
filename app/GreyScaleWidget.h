#pragma once
#include <QWidget>

class QLabel;
class Slider;

class GreyScaleWidget : public QWidget
{
    Q_OBJECT
public:
    GreyScaleWidget(QWidget* parent=nullptr);
    virtual ~GreyScaleWidget();
    void SetValues(int level, int range);
    void GetValues(int& level, int& range) const;

signals:
    void sign_valueChanged(int level, int range);

private slots:
    void slot_RangeChanged(int);
    void slot_LevelChanged(int);

private:
    QLabel* DisplayName = nullptr;
    Slider* Range = nullptr;
    Slider* Level = nullptr;
};