#include "core/macros.h"
#include "vpInlines.h"
#include "vpVolume.h"

template <typename T>
void vpVolume<T>::init()
{
    m_dim[0] = m_dim[1] = m_dim[2] = 0;
    m_range[0] = m_range[1] = 0;
    m_shallowCopy = false;
    m_voxels = NULL;
}

template <typename T>
vpVolume<T>::vpVolume()
{
    init();
}

template <typename T>
vpVolume<T>::vpVolume(const vpVolume &A)
{
    init();

    SetShallowCopy(true);
    A.GetDimensions(m_dim);
    A.GetRange(m_range);
    SetDataPointer( A.GetDataPointer() );
}

template <typename T>
vpVolume<T>::~vpVolume()
{
    if(!m_shallowCopy)
    {
        MacroDeleteArray(m_voxels);
    }
}

template <typename T>
void vpVolume<T>::DeepCopy(const vpVolume &A)
{
    init();
    A.GetDimensions(m_dim);
    A.GetRange(m_range);
    CopyData( A.GetDataPointer(), A.GetArraySize() );
}

template <typename T>
void vpVolume<T>::CopyData(const T *data, const size_t arraySize)
{
    if( m_shallowCopy )
    {
        MacroWarning("Cannot change data in shallow copy.");
        return;
    }

    MacroDeleteArray(m_voxels);
    m_voxels = new T[arraySize];
    memcpy( m_voxels, data, arraySize*sizeof(T));
}

template <typename T>
void vpVolume<T>::Initialize(const size_t arraySize)
{
    MacroDeleteArray(m_voxels);
    m_voxels = new T[arraySize];
    memset( m_voxels, 0, arraySize*sizeof(T) );
}

template <typename T>
void vpVolume<T>::Initialize()
{
    size_t arraySize = GetArraySize();
    if( arraySize )
        Initialize(arraySize);
}

template <typename T>
T vpVolume<T>::operator[](size_t index) const
{
#ifdef DEBUG
    if( out_of_bounds(index) )
    {
        MacroWarning("Voxel index out of bounds.");
        return T(0);
    }
#endif

    return m_voxels[index];
}

template <typename T>
T &vpVolume<T>::operator[](size_t index)
{
#ifdef DEBUG
    if( out_of_bounds(index) )
    {
        MacroWarning("Voxel index out of bounds.");
        //return T(0);
        return m_voxels[0];
    }
#endif

    return m_voxels[index];
}

template <typename T>
void vpVolume<T>::GetRange(T range[2]) const
{
    vpn::copyvec<T>( m_range, range, 2);
}

template <typename T>
std::pair<double,double> vpVolume<T>::GetRange() const
{
    return std::make_pair(double(m_range[0]), double(m_range[1]));
}

template <typename T>
void vpVolume<T>::GetDimensions(size_t dim[3]) const
{
    vpn::copyvec<size_t>( m_dim, dim, 3);
}

template <typename T>
void vpVolume<T>::GetSpacing(double spacing[3]) const
{
    vpn::copyvec<double>(m_spacing, spacing, 3);
}

template <typename T>
T* vpVolume<T>::GetDataPointer() const
{
    return m_voxels;
}

template <typename T>
size_t vpVolume<T>::GetArraySize() const
{
    return m_dim[0] * m_dim[1] * m_dim[2];
}

template <typename T>
void vpVolume<T>::SetDimensions(const size_t dim[3])
{
    NullCheckVoid(dim);
    vpn::copyvec<size_t>( dim, m_dim, 3);
}

template <typename T>
void vpVolume<T>::SetSpacing(const double spacing[3])
{
    NullCheckVoid(spacing);
    vpn::copyvec<double>( spacing, m_spacing, 3);
}

template <typename T>
bool vpVolume<T>::ValidDimensions() const
{
    if( m_dim[0] > 0 && m_dim[1] > 0 && m_dim[2] > 0 )
        return true;
    else
    {
        MacroWarning("Invalid volume dimensions.");
        return false;
    }
}

template <typename T>
bool vpVolume<T>::valid_data() const
{
    if( m_voxels )
        return true;
    else
    {
        MacroWarning("Invalid voxel data.");
        return false;
    }
}

template <typename T>
bool vpVolume<T>::valid_range() const
{
    if( m_range[0] != m_range[1] )
        return true;
    else
    {
        MacroWarning("Invalid data range.");
        return false;
    }
}

template <typename T>
inline bool vpVolume<T>::valid_indices(size_t i, size_t j, size_t k) const
{
    return (i < m_dim[0] && j < m_dim[1] && k < m_dim[2]);
}

template <typename T>
void vpVolume<T>::ComputeRange()
{
    MacroConfirm( valid_data() );
    MacroConfirm( ValidDimensions() );

    short minValue = VPN_SHORT_MAX, maxValue = VPN_SHORT_MIN;

    size_t sz = m_dim[0]*m_dim[1]*m_dim[2];
    for( size_t i=0; i < sz; i++ )
    {
        short value = m_voxels[i];
        if( value < minValue) minValue = value;
        if( value > maxValue) maxValue = value;
    }

    m_range[0] = minValue;
    m_range[1] = maxValue;
}

template <typename T>
T vpVolume<T>::GetValue(size_t i, size_t j, size_t k) const
{
    MacroConfirmOrReturn( valid_indices(i,j,k), 0);
    size_t idx = ToArrayIdx(i,j,k);
    return m_voxels[idx];
}

template <typename T>
T vpVolume<T>::GetValue( size_t index[3] ) const
{
    MacroAssert(index);
    return GetValue( index[0], index[1], index[2] );
}

template <typename T>
void vpVolume<T>::SetValue(size_t i, size_t j, size_t k, const T& val) const
{
    MacroConfirm(valid_indices(i,j,k));
    size_t idx = ToArrayIdx(i,j,k);
    m_voxels[idx] = val;
}

template <typename T>
void vpVolume<T>::SetValue(size_t index[3], const T& val)
{
    MacroAssert(index);
    SetValue( index[0], index[1], index[2], val);
}

template <typename T>
inline size_t vpVolume<T>::ToArrayIdx( size_t i, size_t j, size_t k ) const
{
    MacroConfirmOrReturn( valid_indices(i,j,k), VPN_SIZET_MAX);
    return (i + j*m_dim[0] + k*m_dim[0]*m_dim[1]);
}


// Explicit Instantiation
//template class vpVolume<uchar>;
//template class vpVolume<short>;

