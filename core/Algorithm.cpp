#include "macros.h"
#include "Algorithm.h"
//#include "graphseg_globals.h"

using sjCore::Algorithm;

Algorithm::~Algorithm()
{
}

Algorithm::Algorithm()
//    : m_error_code(0), m_run_called(0)
{
    m_error_code = 0;
    m_run_called = false;
    m_output_object_ownership = true;
}

int Algorithm::Run()
{
    if(m_run_called)
    {
        MacroWarning("Run function cannot be called twice. Construct a new object.");
        SetErrorCode(ErrorAlreadyExecuted);
        return 0;
    }

    m_run_called = true;

    if( !input_validation() )
    {
        MacroWarning("Input Validation Failed.");
        SetErrorCode(ErrorInputValidation);
        return 0;
    }

    int returnValue = primary_process();
    if( returnValue == 0 )
        SetErrorCode(ErrorExecution);

    returnValue &= post_processing();

    return returnValue;
}

int Algorithm::GetErrorCode() const
{
    return m_error_code;
}

bool Algorithm::GetRunCalled() const
{
    return m_run_called;
}


void Algorithm::SetErrorCode(int error_code)
{
    m_error_code = error_code;
}

void Algorithm::EnableOutputOwnership()
{
    m_output_object_ownership = true;
}

void Algorithm::DisableOutputOwnership()
{
    m_output_object_ownership = false;
}

bool Algorithm::GetOutputOwnership() const
{
    return m_output_object_ownership;
}
