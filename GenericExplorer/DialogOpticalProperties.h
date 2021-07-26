#ifndef DIALOG_OPTICAL_PROPERTIES_H
#define DIALOG_OPTICAL_PROPERTIES_H

#include <QDialog>
#include <QString>
#include "vpTransferFunctionEditor.h"
#include "core/macros.h"

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
    void SetHistogram(const std::vector<float> &hist);
    MacroGetMember(const vpTransferFunctionEditor*, m_tfEditor, TFEditor)

    const QString cAlphaLabelFormat = " Alpha: ";

private:
    // Copying of this dialog is not allowed.
    DialogOpticalProperties& operator=(const DialogOpticalProperties& ) = delete;

    QColor        m_currColor;
    vpTransferFunctionEditor* m_tfEditor = nullptr;


private slots:
    void slot_finished(int);

public slots:
    void slot_setTransferFunction(const vpTransferFunction& tfunc);

signals:
    void sign_TransferFunctionUpdated( const vpTransferFunction* tfunc );

};

#endif
