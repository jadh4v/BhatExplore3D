#pragma once

class Node
{
public:
    Node(size_t id);
    Node() = delete;

    /// Set unique Id of this node.
    //void SetId(size_t id);

    /// Get unique Id of this node.
    size_t Id() const;

    /// Start iterating over node neighbors (ids).
    void BeginNeighborIteration() const;

    /// Return next neighbor id in iteration.
    size_t GetNextNeighbor() const;

    /// Return total number of neighbors for the current node.
    size_t NeighborCount() const;

    /// Confirm that it is a valid constructed node.
    bool IsValid() const;

    /// Insert a neighboring node relationship (edge). This should be done for both nodes involved.
    /// Returns 0 if neighbor already exists.
    /// Returns 1 if neighbor successfully inserted.
    int InsertNeighbor(size_t id);

    int DeleteNeighbor(size_t id);

    void SetFlag(size_t index) const;
    void ClearFlag(size_t index) const;
    bool GetFlag(size_t index) const;
    void ClearAllFlags() const;
    void SetAllFlags() const;

    bool operator==(const Node&) const;
    bool operator!=(const Node&) const;


private:

    /// MEMBER VARIABLES
    /// Unique ID of node. Zero is reserved as invalid / default id.
    size_t m_Id = 0; 
    mutable unsigned char m_flags = 0;

    /// Neighbors (connected thru edges) of the node in the graph structure.
    std::set<size_t> m_Nei; 

    /// Iterate over neighors
    mutable std::set<size_t>::const_iterator m_Iter;

};