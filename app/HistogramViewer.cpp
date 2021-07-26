#include <QHBoxLayout>
#include <vtkImageData.h>
#include <vtkImageProperty.h>
#include <vtkImageSlice.h>
#include <vtkImageSliceMapper.h>
#include <vtkRenderer.h>
#include <vtkInteractorStyleImage.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <QVTKInteractor.h>
#include "GreyScaleWidget.h"
#include "RenderCallback.h"
#include "HistogramViewer.h"

HistogramViewer::~HistogramViewer()
{
    delete m_ui.wGreyScale;
}

HistogramViewer::HistogramViewer(ImagePtr pin_image, ImagePtr pout_image)
    : QWidget(nullptr), m_PinImage(pin_image), m_PoutImage(pout_image)
{
    this->setLayout(new QHBoxLayout());
    m_ui.wGreyScale = new GreyScaleWidget();
    m_ui.wGreyScale->SetValues(6, 15);
    //this->setLayoutDirection(Qt::LayoutDirection::LeftToRight);
    this->layout()->addWidget(m_ui.wGreyScale);
    _ConstructHistView();

    QObject::connect(m_ui.wGreyScale, SIGNAL(sign_valueChanged(int, int)), this, SIGNAL(sign_valueChanged(int, int)));
}

void HistogramViewer::SetGreyScaleValues(int level, int range)
{
    m_ui.wGreyScale->SetValues(level, range);
}

void HistogramViewer::GetGreyScaleValues(int & level, int & range) const
{
    m_ui.wGreyScale->GetValues(level, range);
}

void HistogramViewer::_ConstructHistView()
{
    m_hist.Widget = new QVTKRenderWidget();
    this->layout()->addWidget(m_hist.Widget);
    m_hist.Widget->resize(512, 512);

    vtkNew<vtkGenericOpenGLRenderWindow> window;
    window->AddRenderer(m_hist.Ren);
    window->SetPosition(900, 0);
    window->SetMultiSamples(8);
    window->SetSize(1000, 1000);
    m_hist.Widget->setRenderWindow(window);

    auto surface = QVTKRenderWidget::defaultFormat();
    surface.setSamples(8);
    //m_ui.RenWidget->setFormat(surface);
    m_hist.Widget->setFormat(surface);
    //m_ui.RenWidget->setEnableHiDPI(true);
    m_hist.Widget->setEnableHiDPI(true);

    vtkNew<vtkImageSliceMapper> PinMapper;
    vtkNew<vtkImageSliceMapper> PoutMapper;
    PinMapper->SetInputData(m_PinImage);
    PoutMapper->SetInputData(m_PoutImage);
    m_hist.pinActor->SetMapper(PinMapper);
    m_hist.poutActor->SetMapper(PoutMapper);
    //m_hist.poutActor->SetOrigin(300, 0, 0);
    m_hist.Ren->AddActor(m_hist.pinActor);
    m_hist.Ren->AddActor(m_hist.poutActor);

    vtkNew<vtkInteractorStyleImage> imageInteractor;
    m_hist.Widget->interactor()->SetInteractorStyle(imageInteractor);
    m_hist.Widget->interactor()->Initialize();

    m_hist.Ren->ResetCamera();
    m_hist.Ren->SetBackground(m_Colors->GetColor3d("White").GetData());

    // Sign up to receive TimerEvent
    vtkNew<RenderCallback> cb;
    cb->SetWindow(m_hist.Widget->renderWindow());
    m_hist.Widget->interactor()->AddObserver(vtkCommand::TimerEvent, cb);
    int timerId = m_hist.Widget->interactor()->CreateRepeatingTimer(m_RefreshTime);

    int level = 0, range = 0;
    this->GetGreyScaleValues(level, range);
    this->slot_SetHistGreyScale(level, range);
}

void HistogramViewer::slot_SetHistGreyScale(int level, int range)
{
    m_hist.pinActor->GetProperty()->SetColorLevel(double(level));
    m_hist.poutActor->GetProperty()->SetColorLevel(double(level));
    m_hist.pinActor->GetProperty()->SetColorWindow(double(range));
    m_hist.poutActor->GetProperty()->SetColorWindow(double(range));
    m_hist.Widget->renderWindow()->Render();
}
