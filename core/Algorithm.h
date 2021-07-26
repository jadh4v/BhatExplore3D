#ifndef ALGORITHM_H
#define ALGORITHM_H

#include "macros.h"

namespace sjCore{

class Algorithm
{
public:
    enum ErrorCodes{ NoError=0, ErrorConstruction=-1, ErrorExecution=-2, ErrorFileIO=-3,
                     ErrorInputValidation=-4, ErrorAlreadyExecuted=-5 };

    Algorithm();
    virtual ~Algorithm();

    virtual int Run() final;
    virtual int  GetErrorCode() const final;
    virtual bool GetRunCalled() const final;
    virtual void EnableOutputOwnership()  final;
    virtual void DisableOutputOwnership() final;
    virtual bool GetOutputOwnership() const final;

    virtual bool VerboseOn()     const { return m_verbose;  }
    virtual void SetVerboseOn()  final { m_verbose = true;  }
    virtual void SetVerboseOff() final { m_verbose = false; }

protected:
    virtual void SetErrorCode(int error_code) final;
    int  m_error_code = 0;

private:
    // Non copyable class by hiding copy constructor
    Algorithm(const Algorithm&) = delete;
    virtual int input_validation() const=0;
    virtual int primary_process()=0;
    virtual int post_processing()=0;
    
    /// Member Variables
    bool m_run_called = false;  /**< Determine if Run has been called earlier or not. */
    bool m_verbose = false;     /**< Set progress messages and information ON.        */
    bool m_output_object_ownership = true; /**< Determine if output object(s) (if any) is owned by the Algorithm Object.*/
};

}

#endif // ALGORITHM_H
