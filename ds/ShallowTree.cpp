#include <queue>
#include <QString>
#include <QCryptographicHash>
#include <QFile>
#include <QDataStream>
#include "ShallowTree.h"
#include "core/macros.h"

using std::vector;
using std::queue;
using sjDS::ShallowTree;

const ShallowTree::nodeId_t ShallowTree::cInvalidNodeID = ~0;
const ShallowTree::nodeId_t ShallowTree::cRootID = 0;

ShallowTree::ShallowTree()
{
    // Create default root node:
    node n(cRootID);
    m_nodes.insert( std::make_pair(cRootID, n) );
    BeginIteration();
}

ShallowTree::ShallowTree(const QString& fileName)
{
    if( !read( fileName ) )
        SetInvalidConstruction();

    BeginIteration();
}

ShallowTree::ShallowTree(QDataStream& stream)
{
    read(stream);
}

/*
ShallowTree::ShallowTree(const ShallowTree& C)
{
    m_nodes = C.m_nodes;
    m_hash = C.m_hash;
    BeginIteration();
}

// Assignment Operator
ShallowTree& ShallowTree::operator=(const ShallowTree& C)
{
    m_nodes = C.m_nodes;
    m_hash = C.m_hash;
    BeginIteration();
    return *this;
}
*/

bool ShallowTree::InsertRelation(ShallowTree::nodeId_t parent_id, ShallowTree::nodeId_t child_id,
                                 bool createNodes )
{
    bool ret = true;
    node* p = find_node( parent_id );
    if(p)
    {
        // If parent already exists, insert child.
        ret &= p->InsertChild( child_id );
    }
    else if(createNodes)
    {
        // If parent doesn't exist, create parent and then insert child.
        node n(parent_id);
        ret &= n.InsertChild(child_id );
        m_nodes.insert( std::make_pair(parent_id,n) );
    }
    else
        ret = false; // this is a failure, if nodes were not asked to be created.

    // Update child node if it already exists:
    if( !update_parent(child_id, parent_id) )
    {
        if( createNodes )
        {
            // If child node doesn't exist, insert a new one and update its parent.
            node n(child_id);
            ret &= n.UpdateParent(parent_id);
            m_nodes.insert( std::make_pair(child_id,n) );
        }
        else
            ret = false; // this is a failure, if nodes were not asked to be created.
    }

    // Warn that update of relationship was not successful.
    if( ret == false )
        MacroWarning("Failed to insert relationship.");

    return ret;
}

ShallowTree::node* ShallowTree::find_node(nodeId_t node_id)
{
    auto fnd = m_nodes.find(node_id);
    if( fnd != m_nodes.end() )
        return &(fnd->second);
    else
        return nullptr;
}

const ShallowTree::node* ShallowTree::find_node(nodeId_t node_id) const
{
    auto fnd = m_nodes.find(node_id);
    if( fnd != m_nodes.end() )
        return &(fnd->second);
    else
        return nullptr;
}

bool ShallowTree::erase_node(nodeId_t node_id)
{
    auto fnd = m_nodes.find(node_id);
    if( fnd != m_nodes.end() )
    {
        m_nodes.erase( fnd );
        return true;
    }
    else
        return false;
}

bool ShallowTree::update_parent(ShallowTree::nodeId_t node_id, ShallowTree::nodeId_t parent_id)
{
    node* child = find_node(node_id);
    if( child )
    {
        child->UpdateParent(parent_id);
        /*
        bool overwrite_flag = child->UpdateParent(parent_id);
        if(!overwrite_flag)
            MacroWarning("Some parent_id was overwritten.");
            */

        return true;
    }
    return false;
}

ShallowTree::nodeId_t ShallowTree::GetRoot() const
{
    auto fnd = m_nodes.find( cRootID );
    if( fnd == m_nodes.end() )
    {
        MacroWarning("Root node not found.");
        return cInvalidNodeID;
    }
    else
        return cRootID;
}


ShallowTree::nodeId_t ShallowTree::GetParent(nodeId_t node_id) const
{
    const node* n = find_node(node_id);
    if( n == nullptr )
    {
        MacroWarning("Cannot find requested node.");
        return cInvalidNodeID;
    }

    return n->Parent();
}

ShallowTree::nodeId_t ShallowTree::GetAncestor(nodeId_t node_id) const
{
    const node* n = find_node(node_id);
    if( n == nullptr )
    {
        //MacroWarning("Cannot find requested node.");
        return cInvalidNodeID;
    }

    const node* parent = nullptr;
    while( n )
    {
        nodeId_t p_id = n->Parent();
        parent = n;

        if( p_id == cRootID )
            break;

        n = find_node(p_id);
    }

    if( parent != nullptr && parent->Id() != node_id && parent->Id() != cRootID )
        return parent->Id();
    else
        return cInvalidNodeID;
}

bool ShallowTree::isAncestor(nodeId_t node_id, nodeId_t anc_id) const
{
    const node* n = find_node(node_id);
    if( n == nullptr )
    {
        MacroWarning("Cannot find requested node.");
        return false;
    }

    while( n != nullptr )
    {
        nodeId_t p_id = n->Parent();
        if( p_id == anc_id )
            return true;

        if( p_id == cRootID || p_id == cInvalidNodeID )
            break;

        n = find_node(p_id);
    }

    return false;
}

bool ShallowTree::GetChildren(nodeId_t node_id, vector<nodeId_t>& children) const
{
    const node* n = find_node(node_id);
    if( n == nullptr )
    {
        MacroWarning("Cannot find requested node: " << node_id );
        return false;
    }

    children.clear();
    n->GetChildren( children );
    return true;
}

int ShallowTree::GetLeaves(nodeId_t node_id, std::vector<nodeId_t>& leafDesc) const
{
    leafDesc.reserve(128);
    queue<nodeId_t> Q;
    Q.push(node_id);

    vector<nodeId_t> children;
    while(!Q.empty())
    {
        nodeId_t n = Q.front();
        Q.pop();
        GetChildren( n, children );
        if( children.empty() )
            leafDesc.push_back( n );
        else
        {
            for(auto&& c : children)
                Q.push(c);
        }
    }

    return 1;
}

size_t ShallowTree::GetChildCount(nodeId_t node_id) const
{
    const node* n = find_node(node_id);
    if( n )
        return n->GetChildCount();
    else
    {
        MacroWarning("Cannot find accessed node.");
        return 0;
    }
}

size_t ShallowTree::GetSize() const
{
    return m_nodes.size();
}

ShallowTree::nodeId_t ShallowTree::GetLargestId() const
{
    if( !m_nodes.empty() )
        return m_nodes.rbegin()->first;
    else
        return 0;
}

bool ShallowTree::isValid(nodeId_t id) const
{
    if(id == cInvalidNodeID)
        return false;

    if( m_nodes.find(id) != m_nodes.end() )
        return true;
    else
        return false;
}

bool ShallowTree::EraseNode( nodeId_t node_id )
{
    // First confirm that a valid node ID was passed to erase.
    MacroConfirmOrReturn( isValid( node_id ), false );
    bool ret = true;

    // Get the parent ID and node
    nodeId_t parent_id = GetParent( node_id );
    node* parent = find_node(parent_id);

    // If all info is OK, proceed to delete the node and reconnect
    // its children to the parent node.
    if( isValid(parent_id) && parent != nullptr )
    {
        vector<nodeId_t> children;
        ret &= GetChildren( node_id, children );

        for( auto&& child_id : children )
            ret &= InsertRelation( parent_id, child_id, false );

        ret &= erase_node( node_id );

        parent->EraseChild(node_id);
    }
    else
        ret = false;

    return ret;
}

bool ShallowTree::UpdateNodeId( nodeId_t old_id, nodeId_t new_id )
{
    if( !isValid(old_id) && isValid(new_id) )
    {
        MacroWarning("NodeId update failure.");
        return false;
    }

    // Find the node to update
    node* n = find_node(old_id);
    if( n )
    {
        nodeId_t pid = n->Parent();
        node* p = find_node( pid );

        // Update Parent
        if( p )
        {
            if( !p->InsertChild( new_id ) )
            {
                MacroWarning("Cannot update parent.");
                return false;
            }
            p->EraseChild( old_id );
        }

        node new_node(new_id);
        new_node.UpdateParent( pid );
        vector<nodeId_t> children;
        n->GetChildren( children );
        for( auto cid : children )
        {
            new_node.InsertChild( cid );
            node* c = find_node(cid);
            if( c )
                c->UpdateParent( new_id );
            else
            {
                MacroWarning("Cannot find child node for updating.");
                return false;
            }
        }

        erase_node( old_id );
        m_nodes.insert( std::make_pair( new_id, new_node) );

    }
    else
        return false;

    return true;
}

void ShallowTree::BeginIteration() const
{
    if( m_nodes.empty() )
        mu_nodeIter = m_nodes.end();
    else
        mu_nodeIter = m_nodes.begin();
}

ShallowTree::nodeId_t ShallowTree::GetNextId() const
{
    nodeId_t ret = cInvalidNodeID;

    if( mu_nodeIter != m_nodes.end() )
    {
        ret = mu_nodeIter->first;
        ++mu_nodeIter;
    }

    return ret;
}

void sjDS::ShallowTree::SetPackage(nodeId_t node_id, void* package)
{
    node* N = find_node(node_id);
    if( N )
        N->SetPackage(package);
    else
        MacroWarning("Cannot find node.");
}

void* sjDS::ShallowTree::GetPackage(nodeId_t node_id) const
{
    const node* N = find_node(node_id);
    if( N )
        return N->GetPackage();
    else
    {
        MacroWarning("Cannot find node.");
        return 0;
    }
}

size_t ShallowTree::GetDepth() const
{
    nodeId_t root = GetRoot();
    queue<nodeId_t> Q1;
    Q1.push(root);

    size_t depth = 0;
    vector<nodeId_t> children, next_level;
    while(1)
    {
        while(!Q1.empty())
        {
            nodeId_t nid = Q1.front();
            Q1.pop();

            GetChildren( nid, children );
            next_level.insert( next_level.end(), children.begin(), children.end() );
            children.clear();
        }

        ++depth;
        for( auto& c : next_level )
            Q1.push(c);

        next_level.clear();

        if(Q1.empty())
            break;
    }

    return depth;
}

int ShallowTree::Write(const QString& fileName) const
{
    QFile file(fileName);

    if( !file.open(QIODevice::WriteOnly) )
    {
        MacroWarning("Cannot open file to write: " << fileName.toLatin1().constData());
        return 0;
    }

    QDataStream out_stream( &file );

    Write(out_stream);

    return 1;
}

int ShallowTree::Write(QDataStream& out_stream) const
{
    for( const auto& p : m_nodes )
    {
        const node& n = p.second;
        quint32 node_id   = (quint32)n.Id();
        quint32 parent_id = (quint32)n.Parent();
        quint64 childCount = (quint64)n.GetChildCount();
        quint64 package    = (quint64)n.GetPackage();

        out_stream << node_id;
        out_stream << parent_id;
        out_stream << childCount;
        out_stream << package;

        vector<nodeId_t> children;
        n.GetChildren(children);
        for( auto c : children )
        {
            out_stream << (quint32)c ;
        }
    }

    return 1;
}

bool ShallowTree::operator==(const ShallowTree& T2) const
{
    std::map<nodeId_t,node>::const_iterator iter1 = m_nodes.begin();
    std::map<nodeId_t,node>::const_iterator iter2 = T2.m_nodes.begin();

    while( iter1 != m_nodes.end() && iter2 != T2.m_nodes.end() )
    {
        if( iter1->second != iter2->second )
            break;

        ++iter1;
        ++iter2;
    }

    if( !(iter1 == m_nodes.end() && iter2 == T2.m_nodes.end()) )
        return false;
    else
        return true;
}

bool ShallowTree::operator!=(const ShallowTree& T2) const
{
    return (!(*this==T2));
}

int ShallowTree::read(const QString& fileName)
{
    QFile file(fileName);

    if( !file.open(QIODevice::ReadOnly) )
    {
        MacroWarning("Cannot open file to read: " << fileName.toLatin1().constData());
        return 0;
    }

    QDataStream in_stream( &file );
    read(in_stream);

    file.reset();
    QCryptographicHash hasher(QCryptographicHash::Md5);
    hasher.addData( &file );
    m_hash = hasher.result();

    return 1;
}

int ShallowTree::read(QDataStream& in_stream)
{
    while( !in_stream.atEnd() )
    {
        quint32 node_id = 0;
        quint32 parent_id = 0;
        quint64 childCount = 0;
        quint64 package = 0;

        in_stream >> node_id;
        in_stream >> parent_id;
        in_stream >> childCount;
        in_stream >> package;

        node n(node_id);
        n.UpdateParent( parent_id );
        n.SetPackage((void*)package);

        for( size_t cnt = 0; cnt < childCount; ++cnt )
        {
            quint32 child_id = 0;
            in_stream >> child_id;
            n.InsertChild( (nodeId_t)child_id );
        }

        m_nodes.emplace( node_id, n );
    }

    return 1;
}

bool ShallowTree::HashMatch(const QByteArray& input) const
{
    if( m_hash.isEmpty() )
    {
        MacroWarning("No hash was generated for the loaded ShallowTree.");
        return true;
    }

    if( input.size() != m_hash.size() )
        return false;

    auto  this_iter = m_hash.begin();
    auto input_iter = input.begin();

    while( this_iter != m_hash.end() && input_iter != input.end() )
    {
        if( *this_iter != *input_iter )
            return false;

        ++this_iter;
        ++input_iter;
    }

    return true;
}

const QByteArray& ShallowTree::GetHash() const
{
    return m_hash;
}

std::vector<ShallowTree::nodeId_t> ShallowTree::GetPathToNode(nodeId_t node_id) const
{
    vector<nodeId_t> ret;
    nodeId_t p = node_id;
    while( p != ShallowTree::cRootID )
    {
        if( p == ShallowTree::cInvalidNodeID )
        {
            ret.clear();
            MacroWarning("Cannot compute path to root.");
            break;
        }

        ret.push_back(p);
        p = GetParent(p);
    }

    ret.push_back(ShallowTree::cRootID);
    std::reverse(ret.begin(), ret.end());
    return ret;
}

















