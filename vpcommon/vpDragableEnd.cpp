#include <QPen>
#include <QGraphicsScene>
#include "vpDoubleSlider.h"
#include "vpDragableEnd.h"

const int vpDragableEnd::cLineWidth = 10;

vpDragableEnd::vpDragableEnd(const QRectF& rec, vpDoubleSlider* slider,
                             bool isLeft, QGraphicsItem* parent)
    : QGraphicsRectItem(rec, parent)
{
#ifdef _WIN32
    //QPen pen = QPen(QColor(Qt::white));
    //QBrush brush = QBrush(QColor(Qt::blue));
    QPen pen = QPen(QColor(0, 122, 217));
    QBrush brush = QBrush(QColor(0, 122, 217));
#else
    QPen pen(QColor(Qt::white));
    QBrush brush(QColor(Qt::darkGray));
#endif
    //pen.setWidth(cLineWidth);
    //pen.setCapStyle(Qt::RoundCap);
    setPen(pen);
    setBrush(brush);
    setFlag(ItemIsSelectable);
    setFlag(ItemIsMovable);

    m_slider = slider;
    m_is_left = isLeft;

    QObject::connect( this,     SIGNAL(sign_moved()),
                      m_slider, SLOT(slot_childMoved()) );
}

void vpDragableEnd::mouseMoveEvent(QGraphicsSceneMouseEvent* e)
{
    QPointF before_pos = this->pos();
    QGraphicsItem::mouseMoveEvent(e);
    QPointF after_pos = this->pos();

    int ret = this->SetPosition( before_pos, after_pos);
    //after_pos = this->pos();
    if( ret == 1 && m_slider->GetSyncLock() )
    {
        QPointF other_before = m_other->pos();
        qreal dx = after_pos.x() - before_pos.x();
        QPointF other_pt = QPointF(other_before.x()-dx, 0);
        ret = m_other->SetPosition( other_before, other_pt);
        if( ret == 0 )
            this->setPos( before_pos );
    }

    emit(sign_moved());
}

int vpDragableEnd::SetPosition(const QPointF& before_pos, const QPointF& after_pos)
{
    int ret = 1;
    this->setPos( after_pos.x(), before_pos.y() );

    // Restrict movement so that the current end is not dragged
    // beyond the left most edge of drawing window.
    if(this->pos().x() < 0 )
    {
        this->setPos( 0, 0 );
        ret = 0;
    }

    // Restrict movement so that the current end
    // is not dragged beyond the other end.
    if( m_is_left )
    {
        if(this->pos().x() > m_other->pos().x()-cLineWidth)
        {
            this->setPos( m_other->pos().x()-cLineWidth, 0 );
            ret = 0;
        }
    }
    else
    {
        if(this->pos().x() < m_other->pos().x()+cLineWidth)
        {
            this->setPos( m_other->pos().x()+cLineWidth, 0 );
            ret = 0;
        }
    }

    // restrict movement so that the current end
    // is not dragged beyond the right most edge of drawing window.
    if( m_slider )
    {
        qreal scene_x = m_slider->GetViewWidth();
        if( m_is_left )
        {
            if( this->pos().x() > scene_x - cLineWidth )
            {
                this->setPos( scene_x - cLineWidth, 0 );
                ret = 0;
            }
        }
        else
        {
            if( this->pos().x() > scene_x )
            {
                this->setPos( scene_x, 0 );
                ret = 0;
            }
        }
    }

    return ret;
}
