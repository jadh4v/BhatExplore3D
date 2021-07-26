#include <set>
#include "Graph.h"
#include "Graph_Node.h"
#include "core/macros.h"

using std::set;
using sjDS::Graph;

Graph::Node::Node(size_t id)
{
    m_Id = id;
    m_Iter = m_Nei.end();
}

size_t Graph::Node::Id() const 
{ 
    return m_Id; 
}

void Graph::Node::BeginNeighborIteration() const
{ 
    if( m_Nei.empty() )
        m_Iter = m_Nei.end(); 
    else
        m_Iter = m_Nei.begin(); 
}

size_t Graph::Node::GetNextNeighbor() const
{
    size_t ret = 0;
    if( m_Iter != m_Nei.end() )
    {
        ret = *m_Iter;
        ++m_Iter;
    }
    else
        ret = Graph::cInvalidNodeID;

    return ret;
}

size_t Graph::Node::NeighborCount() const
{
    return m_Nei.size();
}

bool Graph::Node::IsValid() const
{
    return m_Id != 0;
}

int Graph::Node::InsertNeighbor(size_t id)
{
    // Check if neighbor already exists.
    // TODO: This is a linear search. It will work for small list of neighbors. \
    // Can we perform these operations on a sorted vector?
    //auto fnd = std::find(m_Nei.begin(), m_Nei.end(), id);
    auto fnd = m_Nei.find(id);

    if( fnd == m_Nei.end() )
    {
        m_Nei.insert(id);
        return 1;   // neighbor successfully inserted.
    }
    else
    {
        MacroWarning("Neigbhor already exists.");  
        return 0;   // neighbor already exists.
    }
}

int Graph::Node::DeleteNeighbor(size_t id)
{
    //auto fnd = std::find(m_Nei.begin(), m_Nei.end(), id);
    auto fnd = m_Nei.find(id);
    if( fnd == m_Nei.end() )
    {
        MacroWarning("Neigbhor doesn't exists.");  
        return 0;   // neighbor already exists.
    }
    else
    {
        m_Nei.erase(fnd);
        return 1;   // neighbor successfully deleted.
    }
}

void Graph::Node::SetFlag(size_t index) const
{
    if( index >= sizeof(m_flags) )
        MacroWarning("flag index out of bounds.");

    m_flags |= unsigned char(0x1) << index;
}

void Graph::Node::ClearFlag(size_t index) const
{
    if( index >= sizeof(m_flags) )
        MacroWarning("flag index out of bounds.");

    m_flags &= ~(unsigned char(0x1) << index);
}

bool Graph::Node::GetFlag(size_t index) const
{
    if( index >= sizeof(m_flags) )
        MacroWarning("flag index out of bounds.");

    if( (m_flags & (unsigned char(0x1) << index)) == 0 )
        return false;
    else
        return true;
}

void Graph::Node::ClearAllFlags() const
{
    m_flags = 0;
}

void Graph::Node::SetAllFlags() const
{
    m_flags = ~0;
}

bool Graph::Node::operator==(const Graph::Node& other) const
{
    // Must have the same ID.
    if( this->m_Id != other.m_Id )
        return false;

    // Must have the same number of neighbors.
    if( this->NeighborCount() != other.NeighborCount() )
        return false;

    if( this->m_Nei.empty() && other.m_Nei.empty() )
        return true;

    // Test that all neighbors are the same.
    auto this_v  = this->m_Nei.begin();
    auto other_v = other.m_Nei.begin();
    while( this_v != m_Nei.end() && other_v != other.m_Nei.end() )
    {
        if( *this_v != *other_v )
            return false;

        ++this_v;
        ++other_v;
    }

    // If all tests pass, then the Nodes are equal.
    return true;
}

bool Graph::Node::operator!=(const Graph::Node& other) const
{
    return !(*this==other);
}
