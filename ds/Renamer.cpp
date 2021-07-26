#include "Renamer.h"
//using sjDS::Renamer;

template<typename T>
bool sjDS::Renamer<T>::Exists(const T & a) const
{
    return( m_entries.count(a) > 0 );
}

template<typename T>
bool sjDS::Renamer<T>::Connected(const T & a, const T & b)
{
    if( a==b )
        return true;

    T a_name = GetName(a);
    T b_name = GetName(b);

    if( a_name == b_name )
        return true;

    return false;
}

template<typename T>
bool sjDS::Renamer<T>::Rename(const T & a, const T & b)
{
    if( a==b || Connected(a,b) )
        return false;

    m_entries.insert( std::make_pair(a,b) );
    return true;
}

template<typename T>
T sjDS::Renamer<T>::GetName(const T & a) 
{
    // Search recursively for the final renamed value.
    std::pair<T,T> last = std::make_pair(0,0);
    auto fnd = m_entries.find(a);
    auto first_entry = fnd;
    while( fnd != m_entries.end())
    {
        last = *fnd;
        fnd = m_entries.find( fnd->second );
    }

    // return the final renamed value.
    if(last.first != last.second)
    {
        first_entry->second = last.second;
        return last.second;
    }

    // return the same name, if not renamed.
    return a;
}



template class sjDS::Renamer<unsigned int>;
template class sjDS::Renamer<size_t>;