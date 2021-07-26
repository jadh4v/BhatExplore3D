#include <math.h>

#include <QGraphicsSceneMouseEvent>
#include <QColorDialog>
#include <QStyleOption>
#include <QPainter>
#include <QGraphicsView>
#include <QKeyEvent>

#include "vpTFNode.h"
#include "vpTFQuad.h"
#include "core/macros.h"

const int vpTFNode::defaultRadius = 5;
const int vpTFNode::cDefaultMargin = 30;

vpTFNode::vpTFNode( QGraphicsScene* scene, QGraphicsItem* parent )
    : QGraphicsItem( parent )
{
    m_left_quad = m_right_quad = 0;

    // Accept drag-drop events
    setAcceptDrops(true);

    if( scene )
    {
        scene->addItem(this);
        QObject::connect( this, SIGNAL(sign_GraphChanged()),
                          scene, SLOT(slot_GraphChangedByNode()) );
    }

    setFlag(ItemIsSelectable);
    setFlag(ItemIsMovable);
    setFlag(ItemSendsGeometryChanges);
    setCacheMode(DeviceCoordinateCache);
    setZValue(1);

    SetColor( QColor(255,0,0,200) );
}

QRectF vpTFNode::boundingRect() const
{
//    return QRectF( -10, -10, 20, 20 );
    return QRectF(-defaultRadius, -defaultRadius, defaultRadius*2, defaultRadius*2 );
}

QPainterPath vpTFNode::shape() const
{
    QPainterPath path;
//    path.addEllipse(-10, -10, 20, 20);
    path.addEllipse(-defaultRadius, -defaultRadius, defaultRadius*2, defaultRadius*2);
    return path;
}

void vpTFNode::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(widget);

    painter->setRenderHint( QPainter::Antialiasing );

    painter->setPen(Qt::NoPen);
    painter->setBrush(Qt::darkGray);
//    painter->drawEllipse(-7, -7, 20, 20);

    QRadialGradient gradient(-3, -3, 10);
    QColor nodeColor = m_color;
    nodeColor.setAlphaF(1.0f);
//    if (option->state & QStyle::State_Sunken)
    if (option->state & QStyle::State_Selected )
    {
        gradient.setCenter(3, 3);
        gradient.setFocalPoint(3, 3);
        gradient.setColorAt(1, nodeColor.light(120));
        gradient.setColorAt(0, nodeColor.light(120));
    }
    else
    {
        gradient.setColorAt(0, nodeColor.light());
        gradient.setColorAt(1, nodeColor.dark());
    }

//    painter->setBrush(gradient);
    QBrush nodeBrush( nodeColor );
    painter->setBrush( nodeBrush );

    if (option->state & QStyle::State_Selected )
        painter->setPen(QPen(Qt::red, 2.0));
    else
        painter->setPen(QPen(Qt::black, 0));

//    painter->drawEllipse(-10, -10, 20, 20);
    painter->drawEllipse(-defaultRadius, -defaultRadius, defaultRadius*2, defaultRadius*2);

}

void vpTFNode::SetMovable(bool val)
{
    setFlag(ItemIsMovable, val);
}

QColor vpTFNode::GetColor() const
{
    return m_color;
}

void vpTFNode::SetColor(QColor color)
{
    m_color = color;
    if( scene() )
    {
        float yp = sqrt(color.alphaF()) * (scene()->height() - cDefaultMargin);
        setPos( x(), yp + cDefaultMargin );
        update();
    }
    else
    {
        MacroWarning("Scene not set.");
    }
}

void vpTFNode::SetLeftQuad(vpTFQuad *quad)
{
    m_left_quad = quad;
}

void vpTFNode::SetRightQuad(vpTFQuad *quad)
{
    m_right_quad = quad;
}

bool vpTFNode::ReplaceLeftQuadNode( vpTFNode* newNode )
{
    if( m_left_quad )
    {
        m_left_quad->ReplaceLeftNode(newNode);
        return true;
    }
    else
    {
        MacroWarning("No left quad found.");
        return false;
    }
}

bool vpTFNode::ReplaceRightQuadNode( vpTFNode* newNode )
{
    if( m_right_quad )
    {
        m_right_quad->ReplaceRightNode(newNode);
        return true;
    }
    else
    {
        MacroWarning("No right quad found.");
        return false;
    }
}

QVariant vpTFNode::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value)
{
    if(change == ItemPositionHasChanged)
    {
        if( m_left_quad  )  m_left_quad->adjust();
        if( m_right_quad )  m_right_quad->adjust();
    }

    return QGraphicsItem::itemChange(change, value);
}

void vpTFNode::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event)
{
    QColor color = QColorDialog::getColor( m_color, 0,
                                           "Pick Node Color",
                                           QColorDialog::ShowAlphaChannel | QColorDialog::DontUseNativeDialog );

    if(color.isValid())
        SetColor(color);
    update();
    emit sign_GraphChanged();
    event->accept();
}

void vpTFNode::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    update();
    QGraphicsItem::mousePressEvent(event);
}

void vpTFNode::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    update();
    QGraphicsItem::mouseReleaseEvent(event);
    emit sign_GraphChanged();
}

void vpTFNode::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    QGraphicsItem::mouseMoveEvent(event);

    if( x() < cDefaultMargin )
        setPos( cDefaultMargin, y() );
    else if( x() > scene()->width() )
        setPos( scene()->width(), y() );

    if( y() < cDefaultMargin )
        setPos( x(), cDefaultMargin );
    else if( y() > scene()->height() )
        setPos( x(), scene()->height() );

    CheckCollision();

    qreal totalHeight = scene()->height() - cDefaultMargin;
    qreal dy = y() - cDefaultMargin;
    dy = dy < 0? 0 : dy;
    m_color.setAlphaF( pow( dy / totalHeight, 2.0f) );

    //emit sign_GraphChanged();
}

void vpTFNode::keyPressEvent(QKeyEvent *event)
{
    QGraphicsItem::keyPressEvent(event);

    if( isSelected() && event->key() == Qt::Key_Delete )
        PrepareDelete();
}

void vpTFNode::dragLeaveEvent(QGraphicsSceneDragDropEvent *event)
{
    QGraphicsItem::dragLeaveEvent(event);

    //emit sign_GraphChanged();
}

void vpTFNode::PrepareDelete()
{
    if( !m_left_quad || !m_right_quad)
        return;

    vpTFNode* rNode = m_right_quad->GetRightNode();

    m_left_quad->ReplaceRightNode(rNode);

    // delete one quad
    m_right_quad->ReplaceLeftNode(0);
    m_right_quad->ReplaceRightNode(0);
    scene()->removeItem(m_right_quad);
//    delete m_right_quad;
}

void vpTFNode::CheckCollision()
{
    if( m_left_quad )
    {
        vpTFNode* lnode = m_left_quad->GetLeftNode();
        CheckCollision(lnode, 1);
    }
    if( m_right_quad )
    {
        vpTFNode* rnode = m_right_quad->GetRightNode();
        CheckCollision(rnode, 2);
    }
}

void vpTFNode::CheckCollision( vpTFNode* otherNode, int side )
{
    if( !otherNode )
        return;

    if(side == 1)
    {
        if( x() < otherNode->x() )
            setPos( otherNode->x(), y() );
    }
    else if(side == 2)
    {
        if( x() > otherNode->x() )
            setPos( otherNode->x(), y() );
    }
}
