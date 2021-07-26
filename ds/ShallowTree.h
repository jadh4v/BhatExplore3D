#pragma once

#include <map>
#include <vector>
#include <climits>
#include <QString>
#include "DSObject.h"

namespace sjDS{ 

class ShallowTree : public DSObject
{
public:
    typedef unsigned int nodeId_t;
    static const nodeId_t cRootID;
    static const nodeId_t cInvalidNodeID;

    /// Default Constructor
    ShallowTree();
    /// Constructor
    ShallowTree(const QString& fileName);
    ShallowTree(QDataStream& stream);

    //Insert a parent-child relation into the tree. Create nodes if necessary.
    bool InsertRelation( nodeId_t parent_id, nodeId_t child_id, bool createNodes = true );
    bool EraseNode( nodeId_t node_id );
    bool UpdateNodeId( nodeId_t old_id, nodeId_t new_id );
    nodeId_t GetRoot() const;
    nodeId_t GetParent(nodeId_t node_id) const;
    nodeId_t GetAncestor(nodeId_t node_id) const;
    bool isAncestor(nodeId_t node_id, nodeId_t anc_id) const;
    bool GetChildren(nodeId_t node_id, std::vector<nodeId_t>& children) const;
    int GetLeaves(nodeId_t node_id, std::vector<nodeId_t>& leafDesc) const;
    size_t GetChildCount(nodeId_t node_id) const;
    size_t GetSize() const;
    size_t GetDepth() const;
    nodeId_t GetLargestId() const;
    bool isValid(nodeId_t id) const;
    void BeginIteration() const;
    nodeId_t GetNextId() const;
    void SetPackage(nodeId_t node_id, void* package);
    void* GetPackage(nodeId_t node_id) const;
    std::vector<nodeId_t> GetPathToNode(nodeId_t node_id) const;

    bool HashMatch(const QByteArray& in) const;
    const QByteArray& GetHash() const;

    int Write(const QString& fileName) const;
    int Write(QDataStream& out_stream) const;
    bool operator==(const ShallowTree& T2) const;
    bool operator!=(const ShallowTree& T2) const;

private:
    // Structure definition to maintain shallow node information.
    #include "ShallowTree_node.hpp"

    //===================================
    // Member Variables
    //===================================
    std::map<nodeId_t,node> m_nodes;
    QByteArray m_hash;
    mutable std::map<nodeId_t,node>::const_iterator mu_nodeIter;
    //===================================

    // Find the node and return its pointer.
    node* find_node(nodeId_t node_id);
    const node* find_node(nodeId_t node_id) const;
    bool erase_node(nodeId_t node_id);
    // Find the node and update its parent_id.
    // return false if node was not found.
    bool update_parent(nodeId_t node_id, nodeId_t parent_id );
    int read(const QString& fileName);
    int read(QDataStream& fileName);

};

}
