#ifndef CORE_MACROS_H
#define CORE_MACROS_H

#include <iostream>
#include <cassert>
#include <fstream>
#include <thread>

extern std::fstream gLog;//("./log.txt", std::fstream::out );

#define MacroWarning(x) \
    do { \
        std::cout << "thread: " << std::this_thread::get_id() << " WRN: " << __FUNCTION__ << " : " << __LINE__ << " - " << x << std::endl; \
        gLog << "WRN: " << __FUNCTION__ << " : " << __LINE__ << " - " << x << std::endl; \
    } while(0)

#define __FILENAME__ (strrchr("\\" __FILE__, '\\')+1)
#define MacroPrint(x) \
    do { \
        std::cout << "thread: " << std::this_thread::get_id() << " " << __FILE__ << " (" << __LINE__ << "): " << #x << " = " << x << std::endl;\
        gLog << __FUNCTION__ << " : " << #x << " = " << x << std::endl;\
    } while(0)

#define MacroPrintVec(data, sz) \
    do { \
        std::cout << "thread: " << std::this_thread::get_id() << " " << __FUNCTION__ << " : " << #data << " = ";\
        for(size_t i=0; i < sz; ++i) std::cout << data[i] << ", ";\
        std::cout << std::endl;\
    } while(0)

#define MacroFatalError(x) \
    do { \
        std::cout << "\n\nFATAL: " << __FUNCTION__ << " : " << __LINE__ << " - " << x << std::endl; \
        gLog << "\n\nFATAL: " << __FUNCTION__ << " : " << __LINE__ << " - " << x << std::endl; \
        exit(0); \
    } while(0)

#define MacroMessage(x) \
    do { \
        std::cout << "thread: " << std::this_thread::get_id() <<  " Msg: " << __FUNCTION__ << "() : " << __LINE__ << " - " << x << std::endl; \
             gLog << "thread: " << std::this_thread::get_id() <<  " Msg: " << __FUNCTION__ << "() : " << __LINE__ << " - " << x << std::endl; \
    } while(0)

#define MacroDelete(x) \
    do { \
        if( x != 0 ) \
        { \
            delete x; \
            x = 0; \
        } \
    } while(0)

#define MacroDeleteVtkObject(x) \
    do{ \
        if( x != 0 ) \
        { \
            x->Delete(); \
            x = 0; \
        } \
    }while(0)

#define MacroDeleteArray(x) \
    do{ \
        if( x != 0 ) \
        { \
            delete[] x; \
            x = 0; \
        } \
    }while(0)

#define MacroFree(x) \
    do{ \
        if( x != 0 ) \
        { \
            free(x); \
            x = 0; \
        } \
    }while(0)

#define MacroNewVtkObject( type, varName ) \
    vtkSmartPointer<type> varName = vtkSmartPointer<type>::New(); 

#define MacroAssert(x) \
    do{ \
        assert(x); \
        if( !(x) ) \
        { \
        MacroWarning( "MacroAssert Failure: " << #x ); \
        } \
    }while(0)

#define MacroConfirm(x) \
    do{ \
        if( !(x) ) \
        { \
            std::cout << "thread: " << std::this_thread::get_id() << " " << __FILE__ << " : line(" << __LINE__ << ")- " << "MacroConfirm Failure: " << #x  << std::endl; \
                 gLog << __FILE__ << " : line(" << __LINE__ << ")- " << "MacroConfirm Failure: " << #x  << std::endl; \
            return; \
        } \
    }while(0)

#define MacroConfirmOrReturn(x,y) \
    do{ \
        if( !(x) ) \
        { \
            std::cout << "MacroConfirm Failure: " << #x  << std::endl; \
            gLog << "MacroConfirm Failure: " << #x  << std::endl; \
            return y; \
        } \
    }while(0)



#define NullCheckVoid( var ) \
    do{ \
        if( !var ) \
        { \
            MacroWarning( "Null Check Failure: " << #var ); \
            return; \
        } \
    }while(0)

#define NullCheck( var, retValue ) \
    do{ \
        if( !var ) \
        { \
            MacroWarning( "Null Check Failure: " << #var ); \
            return retValue; \
        } \
    }while(0)

#define MacroGetMember( type, var, funcName ) \
        type Get##funcName(void) const {  return var;  }

#define MacroConstRefMember( type, var, funcName ) \
        const type& ##funcName(void) const {  return var;  }

#define MacroSetMember( type, var, funcName ) \
        void Set##funcName(type in) { var = in; }


#define MacroSetVector3(type, var, funcName) \
virtual void Set##funcName (type _arg1, type _arg2, type _arg3) \
{ \
    this->var[0] = _arg1; \
    this->var[1] = _arg2; \
    this->var[2] = _arg3; \
} \
virtual void Set##funcName (type _arg1[3]) \
{ \
    Set##funcName(_arg1[0], _arg1[1], _arg1[2]); \
} \

#define MacroGetVector3(type, var, funcName) \
MacroGetMember(const type*, var, funcName) \
virtual void Get##funcName (type _arg1[3]) const \
{ \
    _arg1[0] = this->var[0]; \
    _arg1[1] = this->var[1]; \
    _arg1[2] = this->var[2]; \
} \

#define MacroGetSetVector3(type, var, funcName) \
        MacroGetVector3(type, var, funcName) \
        MacroSetVector3(type, var, funcName)

#define MacroGetSetMember( type, var, funcName ) \
        MacroGetMember( type, var, funcName ) \
        MacroSetMember( type, var, funcName )

#define MacroOpenQFileToRead( qFileObject ) \
    do{ \
    if( !qFileObject.open(QIODevice::ReadOnly) ) \
        MacroWarning("Cannot open file to read: " << qFileObject.fileName().toLatin1().constData()); \
    }while(0)

#define MacroOpenQFileToWrite( qFileObject ) \
    do{ \
    if( !qFileObject.open(QIODevice::WriteOnly) ) \
        MacroWarning("Cannot open file to read: " << qFileObject.fileName().toLatin1().constData()); \
    }while(0)


#define MacroWarnAndContinue(condition,warning_text) \
    { \
        if((condition)) \
        { \
            std::cout << "WarnContinue: " << __FILE__ << "(" << __LINE__ << "): " << warning_text << std::endl; \
            gLog << "WarnContinue: " << __FILE__ << "(" << __LINE__ << "): " << warning_text << std::endl; \
            continue; \
        } \
    }

#define MacroNewITKObject( type, varName ) \
    type::Pointer varName = type::New(); 

#define MacroThreeTimes( x_statement ) \
    do{ \
        for(int i=0; i < 3; ++i){ \
            x_statement; \
        } \
    }while(0)

template <typename T>
void cpvec(const T* src, T* dst, size_t count)
{
    for(size_t i=0; i < count; i++)
        dst[i] = src[i];
}

inline int compute_array_idx(const int dim[3], const int ijk[3], const int numOfComponents)
{
    return (numOfComponents*(ijk[0] + ijk[1]*dim[0] + ijk[2]*dim[0]*dim[1]));
}

inline float fetchValue( const float data[], const int dim[3], const int ijk[3], int numOfComponents )
{
    int idx = compute_array_idx( dim, ijk, numOfComponents );
    return data[idx];
}

inline bool same_pos(const int p1[3], const int p2[3])
{
    return (p1[0] == p2[0] && p1[1] == p2[1] && p1[2]==p2[2]);
}

#define MacroglErrorCheck() glErrorCheck(__FILE__,__LINE__)
unsigned int glErrorCheck(const char* file, const int line);

#endif
