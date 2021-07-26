#ifndef DIALOG_OPTICAL_PROPERTIES_H
#define DIALOG_OPTICAL_PROPERTIES_H

#include <QDialog>
#include <QString>
#include "vpTransferFunctionEditor.h"

class QColorDialog;
class QCheckBox;
class QLabel;
class QSlider;
class QPushButton;
class vpTransferFunctionEditor;

class DialogOpticalProperties : public QDialog
{
    Q_OBJECT
public:
    DialogOpticalProperties(const vpTransferFunction& tfunc, QWidget* parent=0, Qt::WindowFlags flags=0);
    virtual ~DialogOpticalProperties();
    vpTransferFunction GetTF() const;
    QColor GetColor() const;
    void SetSurfaceMode(bool f);
    bool GetSurfaceMode() const;
    void SetSegmentVisibility(bool f);
    bool GetSegmentVisibility() const;
    void SetHistogram(const std::vector<float> &hist);

    const QString cAlphaLabelFormat = " Alpha: ";

private:
    // made private to disable.
    // Copying of this dialog is not allowed.
    DialogOpticalProperties& operator=(const DialogOpticalProperties& );

    QColor        m_currColor;
    //QSlider*      m_alphaSlider  = nullptr;
    //QLabel*       m_alphaLabel   = nullptr;
    QCheckBox*    m_surface_mode = nullptr;
    QCheckBox*    m_visible      = nullptr;
    //QColorDialog* m_color_widget = nullptr;
    QPushButton*  m_okButton     = nullptr;
    QPushButton*  m_cancelButton = nullptr;
    vpTransferFunctionEditor* m_tfEditor = nullptr;


private slots:
    //void slot_colorChanged(QColor selectedColor);
    void slot_finished(int);

};

#endif
