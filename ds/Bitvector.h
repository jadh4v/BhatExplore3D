#pragma once
#include <vector>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include "core/macros.h"

namespace sjDS {

/**
 * @brief The Bitvector class
 * Bitvector class provides a vector of booleans in the form of bits.
 * This is different from std::bitset in the manner that, Bitvector can resize itself on the fly.
 * It doesn't require a constexpr template argument, which seems to be a limitation for std::bitset.
 * Elements are maintained as a vector of uint8_t type. Individual bits are accessed by reading
 * writing the uint8_t elements.
 */
class Bitvector
{
public:
    typedef uint8_t elem_t;

    static inline size_t elem_size()
    {
        return (sizeof(elem_t)*8);
    }

    Bitvector()
    { }

    Bitvector(size_t initial_sz)
    {
        m_data.resize( initial_sz / elem_size() + 1 );
        for(auto iter = m_data.begin(); iter != m_data.end(); ++iter)
            *iter = elem_t(0);

        m_size = initial_sz;
    }

    Bitvector(vtkImageData* maskImage)
    {
        this->Resize((size_t)maskImage->GetNumberOfPoints());
        auto data = maskImage->GetPointData()->GetScalars();
        MacroAssert(data->GetNumberOfComponents() == 1);
        if (data->GetNumberOfComponents() != 1)
            return;

        for (vtkIdType i = 0; i < data->GetNumberOfValues(); ++i)
        {
            if(data->GetVariantValue(i).ToDouble() > 0)
                this->Set(i);
        }
    }

    /// Resize the vector to specified size. All previous data is erased.             
    void Resize(size_t new_sz)
    {
        m_fixed = true;
        m_size = new_sz;
        m_data.resize( new_sz / elem_size() + 1 );
        for(auto iter = m_data.begin(); iter != m_data.end(); ++iter)
            *iter = elem_t(0);
    }

    void SetFixedLength(bool fixed)
    {
        m_fixed = fixed;
    }

    size_t ArraySize() const
    {
        return m_data.size();
    }

    size_t Size() const
    {
        return m_size;
    }

    int Set(size_t bit_pos)
    {
        if( m_fixed )
            MacroConfirmOrReturn( !out_of_range(bit_pos), 0 );
        else
        {
            MacroWarning( "non-fixed length not supported yet." );
            return 0;
        }

        size_t elem_pos = bit_pos / elem_size();
        size_t offset   = bit_pos % elem_size();

        elem_t or_mask = elem_t(0x1) << offset;

        m_data[elem_pos] |= or_mask;

        return 1;
    }

    int Clear(size_t bit_pos)
    {
        if( m_fixed )
            MacroConfirmOrReturn( !out_of_range(bit_pos), 0 );
        else
        {
            MacroWarning( "non-fixed length not supported yet." );
            return 0;
        }

        size_t elem_pos = bit_pos / elem_size();
        size_t offset   = bit_pos % elem_size();

        elem_t and_mask = ~(elem_t(0x1) << offset);

        m_data[elem_pos] &= and_mask;

        return 1;
    }

    bool IsClear() const
    {
        for (size_t i = 0; i < this->ArraySize(); ++i)
        {
            if (m_data[i] != 0)
                return false;
        }
        return true;
    }

    bool Get(size_t bit_pos) const
    {
        MacroConfirmOrReturn( !out_of_range(bit_pos), false );

        size_t elem_pos = bit_pos / elem_size();
        size_t offset   = bit_pos % elem_size();

        elem_t and_mask = elem_t(0x1) << offset;
        elem_t value = m_data[elem_pos] & and_mask;

        return ( value != 0 );
    }

    void ClearBits()
    {
        for(size_t i=0; i < m_data.size(); ++i)
            m_data[i] = 0;
    }

    const void* GetRawPointer() const
    {
        return (void*) &(m_data[0]);
    }


private:
    std::vector<elem_t> m_data;
    bool m_fixed = true;
    size_t m_size = 0;

    bool out_of_range(size_t bit_pos) const
    {
        size_t array_sz = m_data.size();
        size_t total_bits = array_sz * elem_size();
        return ( bit_pos >= total_bits );
    }

};

}