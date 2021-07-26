#ifndef NODE_H
#define NODE_H

#include "core/macros.h"
#include "DSObject.h"

namespace sjDS{

class Grid;

//#################################################################################################
class GridPoint : public DSObject
{
public:
    typedef uint32_t type_uid;
    typedef uint32_t uint;
    enum NeighborhoodMode{ OrthogonalNeighbors, AllNeighbors };
    static const type_uid cInvalidID;
    static const type_uid cOrthoDisp[6];

    GridPoint( type_uid node_id, const Grid* grid);

    ~GridPoint();

    type_uid id() const { return m_id; }

    void StartNeighborIteration();

    type_uid GetNextNeighborID();

//    type_uid GetLabel() const;

    void SetModeToOrthogonalNeighbors();

    void SetModeToAllNeighbors();

    void SetMode(NeighborhoodMode m);

    int ToIJK(size_t ijk[3]) const;


    GridPoint x_f() const;
    GridPoint x_b() const;
    GridPoint y_f() const;
    GridPoint y_b() const;
    GridPoint z_f() const;
    GridPoint z_b() const;

    GridPoint forward(uint axis) const;
    GridPoint backward(uint axis) const;

    GridPoint& operator++();
    GridPoint& operator+(size_t offset);
    GridPoint& operator+=(size_t offset);

//#################################################################################################
private:

    void init();

    bool is_orthogonal_neighbor(type_uid nei_id) const;


    bool m_nei_iteration;
    enum NeighborhoodMode m_nei_mode;
    type_uid m_id;
    const sjDS::Grid* m_grid;

    //neighbor iteration
    int64_t m_nei_start_ijk[3];
    int m_nei_disp;
    int m_dim[3];
    int m_std_nei_cnt;
    //int m_std_nei_cnt = m_grid->is3D() ? 27 : 9;
};

}

#endif // NODE_H
