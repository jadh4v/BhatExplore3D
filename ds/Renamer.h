#pragma once

//#include <map>
#include <unordered_map>

namespace sjDS{

template<typename T>
class Renamer
{
public:
    /// Check if an entry already exists.
    bool Exists(const T& a) const;
    /// Check if an entry already exists.
    bool Connected(const T& a, const T& b);
    /// Check if an entry already exists, if not, then make an entry.
    /// Checking is necessary since duplicate entries are allowed.
    bool Rename(const T& a, const T& b);

    T GetName(const T& a);

private:

    // DATA MEMBERS
    //std::map<T,T> m_entries;
    std::unordered_map<T,T> m_entries;

};

}