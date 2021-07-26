#ifndef VPINLINES_H
#define VPINLINES_H

#include <cstring>

namespace vpn
{

typedef unsigned int uint;
template <typename _Tp>
void copyvec( const _Tp* src, _Tp* dst, uint elemCnt )
{
    memcpy( (void*)dst, (const void*)src, elemCnt*sizeof(_Tp) );
}

template <typename _Tp>
_Tp MaxOfThree( const _Tp val0, const _Tp val1, const _Tp val2 )
{
    return std::max( val0, std::max(val1, val2) );
}

template <typename _Tp>
_Tp MaxOfThree( const _Tp values[3] )
{
    return std::max( values[0], std::max(values[1], values[2]) );
}

}

#endif // VPINLINES_H

