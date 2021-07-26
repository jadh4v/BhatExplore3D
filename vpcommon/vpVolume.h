#ifndef VPVOLUME_H
#define VPVOLUME_H

#include "vpTypes.h"
#include "core/macros.h"

template <typename T>
class vpVolume
{
private:
    size_t              m_dim[3];
    T*                  m_voxels;
    T                   m_range[2];
    double              m_spacing[3];
    bool                m_shallowCopy;

    void init();
    bool valid_data()    const;
    bool valid_range()   const;
    bool valid_indices(size_t i, size_t j, size_t k) const;
    bool out_of_bounds(size_t index) const { return index >= GetArraySize(); }

public:
    vpVolume();
    vpVolume( const vpVolume& A );
    ~vpVolume();

    void DeepCopy( const vpVolume& A );

    //Get Values
    void GetRange(T range[2]) const;
    std::pair<double,double> GetRange() const;
    void GetDimensions(size_t dim[2]) const;
    void GetSpacing(double spacing[2]) const;
    T* GetDataPointer() const;
    size_t GetArraySize() const;
    T GetValue(size_t index[3]) const;
    T GetValue(size_t i, size_t j, size_t k) const;

    size_t GetXSize() const { return m_dim[0]; }
    size_t GetYSize() const { return m_dim[1]; }
    size_t GetZSize() const { return m_dim[2]; }
    inline size_t ToArrayIdx( size_t i, size_t j, size_t k ) const;

    //Set Values
    void SetDimensions(const size_t dim[3]);
    void SetSpacing(const double spacing[3]);
    MacroSetMember(bool, m_shallowCopy, ShallowCopy)
    MacroSetMember(T*, m_voxels, DataPointer)
//    void SetDataPointer(T* ptr);
    void SetValue(size_t index[3], const T& val);
    void SetValue(size_t i, size_t j, size_t k, const T& val) const;
    void CopyData(const T* data, const size_t arraySize);

    void Initialize(const size_t arraySize);
    void Initialize();

    T operator[](size_t index) const;
    T& operator[](size_t index);

    void ComputeRange();
    bool ValidDimensions() const;
    const double* Spacing() const { return m_spacing; }
};

#include "vpVolume.hpp"

#endif // VPVOLUME_H
