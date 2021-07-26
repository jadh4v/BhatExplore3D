#pragma once

#include <vector>
#include <map>
#include <set>
#include "DSObject.h"

namespace sjDS{

/**
    @Class Description: Shallow data-structure for Undirected Graph.
*/
class Graph : public DSObject
{
public:
    class Node;
    class Edge;
    static const size_t cInvalidNodeID = UINT64_MAX;

    Graph();
    int InsertNode(size_t node_id);
    int InsertEdge(size_t node0, size_t node1, double weight=0);
    const Graph::Node* GetNode(size_t id) const;

    /// Iterate over Nodes of the graph. Read-only access.
    void BeginNodeIteration() const;
    const Graph::Node* GetNextNode() const;

    /// Iterate over Edges of the graph. Read-only access.
    void BeginEdgeIteration() const;
    std::pair<Edge,double> GetNextEdge() const;

    std::pair<bool,double> GetEdgeWeight(const Graph::Edge& e) const;

    /// Delete edge entries from edge weight table as well as from the nodes.
    /// This function does NOT delete orphaned nodes. 
    /// Call DeleteNode() explicitly for each node to be deleted.
    int DeleteEdge(const Edge& e);

    /// Delete the edge and merge the two nodes together.
    /// This will preserve the smaller vertex_id and delete the larger id.
    /// All neighbors will be consolidated.
    int CollapseEdge(const Edge& e);

    /// Similar to CollapseEdge() but doesn't require existence of an edge between the two nodes.
    int MergeNodes(size_t keepMe, size_t deleteMe);

    /// Check if an edge already exists.
    bool Exists(const Edge& e) const;

    // Compare for equality:
    bool operator==(const Graph&) const;

    // Structure definition to maintain shallow node information.
    #include "Graph_Node.h"
    // Structure definition to maintain shallow edge information.
    #include "Graph_Edge.hpp"

private:
    Graph::Node* _GetNode(size_t id);
    int _DeleteNeighbor(size_t nodeId, Edge e);
    static size_t _OtherNode(size_t nodeId, Edge e);
    int _DeleteNode(size_t nodeId);

    std::map<size_t, Node> m_Nodes;
    std::map<Edge,double> m_EdgeWeights;
    mutable std::map<size_t, Node>::const_iterator m_NodeIter;
    mutable std::map<Edge, double>::const_iterator m_EdgeIter;
};

}