#ifndef VPDRAGABLEEND_H
#define VPDRAGABLEEND_H

#include <QGraphicsLineItem>
#include "core/macros.h"
class vpDoubleSlider;

class vpDragableEnd : public QObject, public QGraphicsRectItem
{
    Q_OBJECT
    Q_INTERFACES( QGraphicsItem )

public:
    vpDragableEnd(const QRectF& rec, vpDoubleSlider* slider, bool isLeft, QGraphicsItem* parent=0);
    int SetPosition(const QPointF& before_pos, const QPointF& after_pos);
    MacroSetMember( vpDragableEnd*, m_other, Other )

protected:
    void mouseMoveEvent(QGraphicsSceneMouseEvent* e);

signals:
    void sign_moved();

private:
    vpDoubleSlider* m_slider;
     vpDragableEnd* m_other;
     bool           m_is_left;

public:
    static const int cLineWidth;
};

#endif // VPDRAGABLEEND_H
