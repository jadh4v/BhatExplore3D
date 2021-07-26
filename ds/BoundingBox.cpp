#include <cstdint>
#include "BoundingBox.h"
#include "core/macros.h"

namespace sjDS {

    BoundingBox::BoundingBox()
    {
        for (size_t i = 0; i < 3; ++i)
        {
            m_min[i] = UINT64_MAX;
            m_max[i] = 0;
        }
    }

    BoundingBox::BoundingBox(size_t box[6])
    {
        for (auto i : { 0,1,2 })
        {
            m_min[i] = box[i];
            m_max[i + 3] = box[i + 3];
        }
    }

    void BoundingBox::GetDimensions(size_t dim[3])
    {
        for (int i = 0; i < 3; ++i)
            dim[i] = m_max[i] - m_min[i] + 1;
    }

    bool BoundingBox::isValid() const
    {
        return !((m_max[0] == 0 && m_max[1] == 0 && m_max[2] == 0)
            && (m_min[0] == UINT64_MAX && m_min[1] == UINT64_MAX && m_min[2] == UINT64_MAX));
    }

    bool BoundingBox::Contains(const size_t p[3]) const
    {
        NullCheck(p, false);
        bool retValue = true;
        for (size_t i = 0; i < 3; ++i)
            retValue = retValue && (m_min[i] <= p[i]) && (p[i] <= m_max[i]);

        return retValue;
    }

    bool BoundingBox::Intersects(const BoundingBox & B) const
    {
        bool retValue = true;
        for (size_t i = 0; i < 3; ++i)
            retValue = retValue && !((m_max[i] < B.m_min[i]) || (B.m_max[i] < m_min[i]));

        return retValue;
    }

    void BoundingBox::Expand(const size_t p[3])
    {
        for (int i = 0; i < 3; ++i)
        {
            if (m_min[i] > p[i])
                m_min[i] = p[i];

            if (m_max[i] < p[i])
                m_max[i] = p[i];
        }
    }

    void BoundingBox::ExpandBy(size_t offset)
    {
        for (size_t i = 0; i < offset; ++i)
        {
            if (m_min[i] > 0)
                --m_min[i];
            ++m_max[i];
        }
    }

    void BoundingBox::operator+=(const BoundingBox& B)
    {
        this->Expand(B.m_min);
        this->Expand(B.m_max);
    }

    void BoundingBox::GetBox(size_t box[6]) const
    {
        NullCheckVoid(box);

        for (int i = 0; i < 3; ++i)
        {
            box[i] = m_min[i];
            box[i + 3] = m_max[i];
        }
    }

    void BoundingBox::Print(std::ostream& s)
    {
        s <<   "\tmin = {" << m_min[0] << ", " << m_min[1] << ", " << m_min[2] << "}";
        s << "\n\tmax = {" << m_max[0] << ", " << m_max[1] << ", " << m_max[2] << "}";
    }

}