#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include <cstdlib>
#include <iostream>

namespace sjDS {

class BoundingBox
{
public:
    BoundingBox();
    BoundingBox(size_t box[6]);
    void GetDimensions(size_t dim[3]);
    void GetBox(size_t box[6]) const;
    void Expand(const size_t p[3]);
    void ExpandBy(size_t offset);
    bool isValid() const;
    bool Contains(const size_t p[3]) const;
    bool Intersects(const BoundingBox& B) const;
    void Print(std::ostream& s);
    void operator+=(const BoundingBox& B);

private:
    size_t m_min[3], m_max[3];
};

}

#endif
