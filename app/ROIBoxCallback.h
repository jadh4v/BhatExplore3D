#pragma once

#include <vtkBoxWidget.h>
#include <vtkCommand.h>
#include <vtkLinearTransform.h>
#include <vtkNew.h>
#include <vtkProp3D.h>
#include <vtkTransform.h>

class ROIBoxCallback : public vtkCommand
{
public:
    static ROIBoxCallback *New()
    {
        return new ROIBoxCallback;
    }
    virtual void Execute(vtkObject *caller, unsigned long, void*)
    {
        // Here we use the vtkBoxWidget to transform the underlying coneActor
        // (by manipulating its transformation matrix).
/*
        vtkNew<vtkTransform> t;
        vtkBoxWidget *widget = reinterpret_cast<vtkBoxWidget*>(caller);
        widget->GetTransform(t);
        widget->GetProp3D()->SetUserTransform(t);
*/
    }
};
