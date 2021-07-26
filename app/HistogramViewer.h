#pragma once
#include <QWidget>
#include <vtkNew.h>
#include <vtkSmartPointer.h>
#include <vtkNamedColors.h>
#include <QVTKOpenGLNativeWidget.h>
using QVTKRenderWidget = QVTKOpenGLNativeWidget;

class vtkImageData;
class vtkRenderer;
class vtkImageSlice;
class GreyScaleWidget;

class HistogramViewer :
    public QWidget
{
    Q_OBJECT
public:
    typedef vtkSmartPointer<vtkImageData> ImagePtr;
    HistogramViewer(ImagePtr pin_image, ImagePtr  pout_image);
    virtual ~HistogramViewer();
    void SetGreyScaleValues(int level, int range);
    void GetGreyScaleValues(int& level, int& range) const;

signals:
    void sign_valueChanged(int level, int range);

private slots:
    void slot_SetHistGreyScale(int level, int range);

private:
    void _ConstructHistView();

    vtkNew<vtkNamedColors> m_Colors;
    ImagePtr m_PinImage = nullptr;
    ImagePtr m_PoutImage = nullptr;
    const unsigned long m_RefreshTime = 2000;

    struct {
        GreyScaleWidget* wGreyScale = nullptr;
    } m_ui;

    struct {
        vtkNew<vtkRenderer> Ren;
        vtkNew<vtkImageSlice>  pinActor;
        vtkNew<vtkImageSlice> poutActor;
        QVTKRenderWidget* Widget = nullptr;
    }m_hist;
};

