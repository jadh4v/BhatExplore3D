#pragma once
#include <vtkCommand.h>
#include <vtkSmartPointer.h>
#include <vtkRenderWindow.h>

class RenderCallback : public vtkCommand
{
public:
    static RenderCallback *New()
    {
        RenderCallback *cb = new RenderCallback;
        cb->m_TimerCount = 0;
        return cb;
    }

    virtual void Execute(vtkObject *caller, unsigned long eventId,
        void *vtkNotUsed(callData))
    {
        if (vtkCommand::TimerEvent == eventId)
        {
            ++this->m_TimerCount;
            //if (m_Window && m_ContourActor)
            if (m_Window)
            {
                m_Window->Render();
            }
        }
    }

    void SetWindow(vtkSmartPointer<vtkRenderWindow> window)
    {
        m_Window = window;
    }

private:
    int m_TimerCount;
    vtkSmartPointer<vtkRenderWindow> m_Window = nullptr;
};
