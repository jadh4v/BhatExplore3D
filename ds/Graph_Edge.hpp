#pragma once

class Edge
{
public:
    size_t m_id[2]={0,0};

    Edge(size_t id0, size_t id1)
    {
        if( id0 > id1 )
            std::swap(id0,id1);

        m_id[0] = id0;
        m_id[1] = id1;
    }

    bool ValidConstruction() const
    {
        return (m_id[0] != m_id[1]);
    }

    bool operator==(const Edge& A) const
    {
        return (m_id[0] == A.m_id[0] && m_id[1] == A.m_id[1]);
    }

    bool operator!=(const Edge& A) const
    {
        return !(*this == A);
    }

    bool operator<(const Edge& A) const
    {
        if( m_id[0] == A.m_id[0] )
            return m_id[1] < A.m_id[1];
        else
            return m_id[0] < A.m_id[0];
    }
};
