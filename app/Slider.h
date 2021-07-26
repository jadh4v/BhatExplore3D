#pragma once
#include <QWidget>

class QLabel;
class QSlider;
class QString;

class Slider : public QWidget
{
    Q_OBJECT
public:
    Slider(const QString& displayName, Qt::Orientation orientation, int minValue, int maxValue, QWidget*parent=0);
    virtual ~Slider();
    int GetValue() const;
    void SetValue(int value) const;

signals:
    void sign_valueChanged(int);

private slots:
    void slot_valueChanged(int v) { emit(sign_valueChanged(v)); }

private:
    QLabel* m_property = nullptr;
    QSlider* m_slider = nullptr;
    QLabel* m_value = nullptr;
};