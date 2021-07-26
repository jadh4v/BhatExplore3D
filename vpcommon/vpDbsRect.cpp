#include <QPen>
#include "vpDbsRect.h"
#include "vpDragableEnd.h"
#include "vpDoubleSlider.h"

vpDbsRect::vpDbsRect(const QRectF& rect, vpDoubleSlider* slider, QGraphicsItem* parent ) : QGraphicsRectItem(rect, parent)
{
    setPen(QPen(QColor(Qt::black)));
    //setBrush( QBrush( QColor(Qt::gray), Qt::BrushStyle::SolidPattern ) );
    QLinearGradient gradient(rect.bottomLeft(), rect.bottomRight());
    gradient.setColorAt(0, Qt::black);
    gradient.setColorAt(1, Qt::white);
    setBrush( QBrush(gradient) );
    setFlag(ItemIsSelectable);
    setFlag(ItemIsMovable);
    m_slider = slider;
    // setFlag(ItemSendsGeometryChanges);
    // setCacheMode(DeviceCoordinateCache);
    // setZValue(1);
}



void vpDbsRect::mouseMoveEvent(QGraphicsSceneMouseEvent* e)
{
    QGraphicsItem::mouseMoveEvent(e);
    QPointF after_pos = this->pos();
    setPos( after_pos.x(), 0 );

    // restrict to limits:
    if( after_pos.x() < 0 )
        this->setPos( 0, 0 );
    else if( after_pos.x() + this->rect().width() > m_slider->GetViewWidth() )
        this->setPos( m_slider->GetViewWidth() - this->rect().width(), 0 );

    m_left->setPos( this->pos().x(), 0 );
    m_right->setPos( this->pos().x() + this->rect().width(), 0 );

    m_slider->slot_childMoved();
}

void vpDbsRect::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    QRectF rec = this->rect();
    QLinearGradient gradient(rec.bottomLeft(), rec.bottomRight());
    gradient.setColorAt(0, Qt::black);
    gradient.setColorAt(1, Qt::white);
    setBrush( QBrush(gradient) );
    QGraphicsRectItem::paint( painter, option, widget);
}

void vpDbsRect::UpdateShape()
{
    setRect( 0, 0, m_right->pos().x() - m_left->pos().x(), this->rect().height() );
    setPos(m_left->pos().x(), 0);
}
