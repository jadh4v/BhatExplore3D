#include <vector>
#include "Graph.h"
#include "core/macros.h"

using std::pair;
using std::make_pair;
using std::vector;
using sjDS::Graph;

typedef unsigned int uint;

Graph::Graph()
{
    m_NodeIter = m_Nodes.end();
    m_EdgeIter = m_EdgeWeights.end();
}

int Graph::InsertNode(size_t node_id)
{
    if( m_Nodes.count(node_id) == 0 )
    {
        m_Nodes.insert( make_pair(node_id, Node(node_id)) );
        return 1;
    }
    else
        return 0;
}

int Graph::InsertEdge(size_t node0, size_t node1, double weight)
{
    // Return Value. return 1 if successful, return 0 if failure.
    int retValue = 1;

    // Check if edge weight entry was already inserted.
    auto fnd = m_EdgeWeights.find( Graph::Edge(node0,node1) );
    if( fnd != m_EdgeWeights.end() )
    {
        MacroWarning("Edge already inserted.");
        retValue = 0;
    }

    // If Edge is not already inserted, continue insertion procedure.
    if( retValue )
    {
        auto fnd0 = m_Nodes.find(node0);
        auto fnd1 = m_Nodes.find(node1);

        // Both nodes should already exist. Else, return failure.
        /*
        if( fnd0 == m_Nodes.end() || fnd1 == m_Nodes.end() )
        {
            MacroWarning("Both nodes should already exist.");
            retValue = 0;
        }
        */

        // Insert nodes if they don't exist in the graph.
        if( fnd0 == m_Nodes.end() )
        {
            InsertNode(node0);
            fnd0 = m_Nodes.find(node0);
        }
        if( fnd1 == m_Nodes.end() )
        {
            InsertNode(node1);
            fnd1 = m_Nodes.find(node1);
        }

        // Insert neighborhood relations within each node if they both exist.
        if( retValue )
        {
            retValue &= fnd0->second.InsertNeighbor(node1);
            if( retValue )
                retValue &= fnd1->second.InsertNeighbor(node0);
        }

        // If everything else is successful, insert the edge weight entry.
        if( retValue )
            m_EdgeWeights.insert( make_pair( Edge(node0,node1), weight) );
    }

    return retValue;
}

Graph::Node* Graph::_GetNode(size_t id)
{
    auto fnd = m_Nodes.find(id);
    if( fnd != m_Nodes.end() )
        return &(fnd->second);
    else
        return nullptr;
}

const Graph::Node* Graph::GetNode(size_t id) const
{
    auto fnd = m_Nodes.find(id);
    if( fnd != m_Nodes.end() )
        return &(fnd->second);
    else
        return nullptr;
}

void Graph::BeginNodeIteration() const
{
    if( m_Nodes.empty() )
        m_NodeIter = m_Nodes.end();
    else
        m_NodeIter = m_Nodes.begin();
}

const Graph::Node* Graph::GetNextNode() const
{
    const Graph::Node* ret = nullptr;

    if( m_NodeIter != m_Nodes.end() )
    {
        ret = &(m_NodeIter->second);
        ++m_NodeIter;
    }

    return ret;
}

void Graph::BeginEdgeIteration() const
{
    if( m_EdgeWeights.empty() )
        m_EdgeIter = m_EdgeWeights.end();
    else
        m_EdgeIter = m_EdgeWeights.begin();
}

std::pair<Graph::Edge,double> Graph::GetNextEdge() const
{
    std::pair<Graph::Edge,double> ret = make_pair(Edge(0,0),0.0);

    if( m_EdgeIter != m_EdgeWeights.end() )
    {
        ret = *m_EdgeIter;
        ++m_EdgeIter;
    }

    return ret;
}

pair<bool,double> Graph::GetEdgeWeight(const Graph::Edge & e) const
{
    auto fnd = m_EdgeWeights.find( e );
    if( fnd != m_EdgeWeights.end() )
        return make_pair(true,fnd->second);
    else
    {
        MacroWarning("Cannot find edge.");
        return make_pair(false,0.0);
    }
}

int Graph::DeleteEdge(const Edge& e)
{
    int success = 1;
    auto fnd = m_EdgeWeights.find( e );
    if( fnd == m_EdgeWeights.end() )
    {
        MacroWarning("Cannot find edge weight entry.");
        success = 0;
    }
    else
        m_EdgeWeights.erase(fnd);

    success &= _DeleteNeighbor( e.m_id[0], e);
    success &= _DeleteNeighbor( e.m_id[1], e);

    return success;
}

int Graph::CollapseEdge(const Edge & e)
{
    // Delete the given edge first.
    if( !DeleteEdge(e) )
        return 0;

    // id0 is always smaller (by construction of Edge).
    size_t id0 = e.m_id[0];
    size_t id1 = e.m_id[1];

    // Get edges and corresponding weights of id1 (larger id).
    Graph::Node* N = _GetNode(id1);
    if( N == nullptr)
        return 0;

    int success = 1;

    N->BeginNeighborIteration();
    size_t nei_id = N->GetNextNeighbor();
    vector<pair<Edge,double> > edges;
    while( nei_id != Graph::cInvalidNodeID )
    {
        Edge e(id1, nei_id);

        auto w = GetEdgeWeight(e);
        if( w.first )
            edges.push_back( make_pair(e,w.second) );

        nei_id = N->GetNextNeighbor();
    }

    // Delete each edge, and re-insert edge for id0 with the same weights.
    for( auto ew : edges )
    {
        //if( ew.first.m_id[0] != id0 && ew.first.m_id[1] != id0 )
        // Delete edge of larger node.
        success &= DeleteEdge(ew.first);

        // get id of the neighbor involved.
        size_t other_id = _OtherNode( id1, ew.first );
        // insert new edge with the same edge weight.
        if( other_id != id0)
        {
            // check if an edge already exists, if true, then retain that edge and its weight, do not insert new edge.
            auto w = GetEdgeWeight(Edge(id0,other_id));
            if(w.first != true)
                success &= InsertEdge( id0, other_id, ew.second );
        }
        else
            MacroWarning("Cannot insert self pointing edge.");
    }

    // Remove the larger id node. Node should be orphan by now.
    success &= _DeleteNode(id1);

    return success;
}

int Graph::MergeNodes(size_t keepMe, size_t deleteMe)
{
    Edge e( keepMe, deleteMe);
    // Delete the given edge first, if it exists.
    if(Exists(e))
        DeleteEdge(e);

    // Get edges and corresponding weights of deleteMe.
    Graph::Node* N = _GetNode(deleteMe);
    if( N == nullptr)
        return 0;

    int success = 1;

    N->BeginNeighborIteration();
    size_t nei_id = N->GetNextNeighbor();
    vector<pair<Edge,double> > edges;
    while( nei_id != Graph::cInvalidNodeID )
    {
        Edge e(deleteMe, nei_id);

        auto w = GetEdgeWeight(e);
        if( w.first )
            edges.push_back( make_pair(e,w.second) );

        nei_id = N->GetNextNeighbor();
    }

    // Delete each edge, and re-insert edge for keepMe with the same weights.
    for( auto ew : edges )
    {
        // Delete edge from deleteMe node.
        success &= DeleteEdge(ew.first);

        // get id of the neighbor involved.
        size_t other_id = _OtherNode( deleteMe, ew.first );
        // insert new edge with the same edge weight.
        if( other_id != keepMe)
        {
            // check if an edge already exists, if true, then retain that edge and its weight, do not insert new edge.
            //auto w = GetEdgeWeight(Edge(keepMe,other_id));
            //if(w.first != true)
            if( !Exists(Edge(keepMe,other_id) ) )
                success &= InsertEdge( keepMe, other_id, ew.second );
        }
        else
            MacroWarning("Cannot insert self pointing edge.");
    }

    // Remove the larger id node. Node should be orphan by now.
    success &= _DeleteNode(deleteMe);

    return success;
}

bool sjDS::Graph::Exists(const Edge & e) const
{
    auto fnd = m_EdgeWeights.find(e);
    /*
    const Node* N0 = GetNode(e.m_id[0]);
    const Node* N1 = GetNode(e.m_id[1]);

    if( fnd != m_EdgeWeights.end() && (!N0 || !N1) )   
    {
        MacroWarning("Inconsistency: Cannot find nodes." );
        return false;
    }
    */

    return ( fnd != m_EdgeWeights.end() );
}

bool Graph::operator==(const Graph& A) const
{
    // All list sizes should match.
    if( m_Nodes.size() != A.m_Nodes.size() || m_EdgeWeights.size() != A.m_EdgeWeights.size() )
        return false;

    // Test equality for individual nodes.
    auto n0_iter = m_Nodes.begin();
    auto n1_iter = A.m_Nodes.begin();
    while( n0_iter != m_Nodes.end() && n1_iter != A.m_Nodes.end() )
    {
        if( n0_iter->first != n1_iter->first )
            return false;

        if( n0_iter->second != n1_iter->second )
            return false;

        ++n0_iter;
        ++n1_iter;
    }

    // Test equality for edge weights.
    auto e0_iter = m_EdgeWeights.begin();
    auto e1_iter = A.m_EdgeWeights.begin();
    while( e0_iter != m_EdgeWeights.end() && e1_iter != A.m_EdgeWeights.end() )
    {
        if( e0_iter->first != e1_iter->first )
            return false;
        
        if( e0_iter->second != e1_iter->second )
            return false;

        ++e0_iter;
        ++e1_iter;
    }

    return true;
}

int Graph::_DeleteNeighbor(size_t nodeId, Edge e)
{
    int success = 1;
    MacroAssert( nodeId == e.m_id[0] || nodeId == e.m_id[1] );
    auto fnd0 = m_Nodes.find(nodeId);

    if( fnd0 != m_Nodes.end() )
    {
        size_t otherNodeId = _OtherNode(nodeId, e);
        success &= fnd0->second.DeleteNeighbor(otherNodeId);
    }
    else
    {
        MacroWarning("Cannot find node0.");
        success = 0;
    }

    return success;
}

size_t Graph::_OtherNode(size_t nodeId, Edge e)
{
    if(nodeId == e.m_id[1])
        return e.m_id[0];
    else if( nodeId == e.m_id[0] )
        return e.m_id[1];
    else
        return cInvalidNodeID;
}

int Graph::_DeleteNode(size_t nodeId)
{
    auto node_fnd = m_Nodes.find(nodeId);
    if( node_fnd == m_Nodes.end() )
        return 0;

    if(node_fnd->second.NeighborCount() == 0)
    {
        m_Nodes.erase( node_fnd );
        return 1;
    }
    else
    {
        MacroWarning("Cannot erase a node that has neighbors.");
        return 0;
    }
}
