#ifndef VPTYPES_H
#define VPTYPES_H

#include <climits>
#include <cstddef>
#include <vector>

#define VPN_SIZET_MAX  size_t(~0);
#define VPN_SHORT_MAX  SHRT_MAX
#define VPN_SHORT_MIN  SHRT_MIN

//#define USE_UCHAR_TF
//#define USE_USHORT_VOLUME

typedef unsigned char  uchar;
typedef unsigned short ushort;
typedef unsigned int   uint;
typedef std::vector<short> shortarray;
typedef ushort segType;

namespace vpn{

#ifdef USE_UCHAR_TF
    typedef uchar tfType;
#else
    typedef float tfType;
#endif

#ifdef USE_USHORT_VOLUME
typedef ushort volPixelType;
#else
typedef float volPixelType;
#endif

}

#endif // VPTYPES_H

