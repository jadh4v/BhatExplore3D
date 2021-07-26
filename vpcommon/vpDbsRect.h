#ifndef VPDBSRECT_H
#define VPDBSRECT_H

#include <QGraphicsItem>
#include <QGraphicsRectItem>
#include "core/macros.h"

class vpDragableEnd;
class vpDoubleSlider;

class vpDbsRect : public QGraphicsRectItem
{
public:
    vpDbsRect(const QRectF& rect, vpDoubleSlider* slider, QGraphicsItem* parent=0 );
    MacroSetMember( vpDragableEnd*, m_left, Left )
    MacroSetMember( vpDragableEnd*, m_right, Right )
    void UpdateShape();

protected:
    void mouseMoveEvent(QGraphicsSceneMouseEvent* e);
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

private:
    vpDragableEnd   *m_left, *m_right;
    vpDoubleSlider  *m_slider;
};

#endif // VPDBSRECT_H
