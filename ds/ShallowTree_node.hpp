// Structure definition to maintain shallow node information.
struct node
{
private:
    nodeId_t m_parent = 0;
    nodeId_t m_id = 0;
    std::vector<nodeId_t> m_children;
    void* m_package = nullptr;

public:
    node(nodeId_t id) : m_id(id)
    { }

    nodeId_t Id()     const { return m_id;     }
    nodeId_t Parent() const { return m_parent; }

    // Returns true if child already exists or if child was successfully inserted.
    bool InsertChild(nodeId_t child_id)
    {
        bool fnd = false;
        for( const nodeId_t c : m_children )
        {
            if( c == child_id )
                fnd = true;
        }

        if( !fnd )
            m_children.push_back(child_id);

        //return !fnd;
        return true;
    }

    // Returns false if parent was overwritten.
    bool UpdateParent( nodeId_t parent_id )
    {
        bool ret = m_parent == 0? true : false;
        m_parent = parent_id;
        return ret;
    }

    // Find and erase child from m_children vector.
    bool EraseChild( nodeId_t child_id )
    {
        for(auto iter = m_children.begin(); iter != m_children.end(); ++iter)
        {
            if( *iter == child_id )
            {
                m_children.erase(iter);
                return true;
            }
        }

        return false;
    }

    void GetChildren( std::vector<nodeId_t>& children ) const
    {
        children.insert( children.end(), m_children.begin(), m_children.end() );
    }

    size_t GetChildCount() const
    {
        return m_children.size();
    }

    void SetPackage(void* packagePtr)
    {
        m_package = packagePtr;
    }

    void* GetPackage() const
    {
        return m_package;
    }

    bool operator==(const node& N) const
    {
        if( m_id != N.m_id )
            return false;

        if( m_parent != N.m_parent )
            return false;

        if( GetChildCount() != N.GetChildCount() )
            return false;

        for(size_t i=0; i< m_children.size(); ++i)
        {
            if( m_children[i] != N.m_children[i] )
                return false;
        }

        return true;
    }

    bool operator!=(const node& N) const
    {
        return (!(*this==N));
    }
};
