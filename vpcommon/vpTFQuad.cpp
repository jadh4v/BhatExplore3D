#include <QGraphicsScene>
#include <QPainter>
#include <QStyle>
#include <QWidget>

#include "vpTFNode.h"
#include "vpTFQuad.h"
#include "core/macros.h"

#include <cassert>


vpTFQuad::vpTFQuad(vpTFNode* leftNode, vpTFNode* rightNode)
{
    setAcceptedMouseButtons(0);
    m_left  = leftNode;
    m_right = rightNode;

    m_left->SetRightQuad(this);
    m_right->SetLeftQuad(this);

    setZValue(0);
    adjust();
    this->update();
}

void vpTFQuad::adjust()
{
    prepareGeometryChange();
    m_leftPoint  = m_left->pos();
    m_rightPoint = m_right->pos();
}

bool vpTFQuad::HasNode(vpTFNode *node) const
{
    if( m_left == node || m_right == node )
        return true;
    else
        return false;
}

void vpTFQuad::ReplaceLeftNode(vpTFNode *newNode)
{
    m_left = newNode;
    if( newNode )
    {
        newNode->SetRightQuad(this);
        this->adjust();
        this->update();
        newNode->update();
    }
}

void vpTFQuad::ReplaceRightNode(vpTFNode *newNode)
{
    m_right = newNode;
    if( newNode )
    {
        newNode->SetLeftQuad(this);
        this->adjust();
        this->update();
        newNode->update();
    }
}

vpTFNode* vpTFQuad::GetLeftNode() const
{
    return m_left;
}

vpTFNode* vpTFQuad::GetRightNode() const
{
    return m_right;
}

/*
bool vpTFQuad::ReplaceNode(vpTFNode *oldNode, vpTFNode *newNode)
{
    if( m_left == oldNode )
    {
        m_left = newNode;
        m_left->AddQuad(this);
        return true;
    }
    else if( m_right == oldNode )
    {
        m_right = newNode;
        m_right->AddQuad(this);
        return true;
    }

    return false;
}

int vpTFQuad::GetOtherNode(vpTFNode *node, vpTFNode* &otherNode ) const
{
    if( m_left  == node )
    {
        otherNode = m_right;
        return 2;
    }
    if( m_right == node )
    {
        otherNode = m_left;
        return 1;
    }
    return 0;
}
*/

QRectF vpTFQuad::boundingRect() const
{
    if (!m_left || !m_right)
        return QRectF();

    qreal w = qAbs( m_leftPoint.x()-m_rightPoint.x() );
    qreal h = qMax( m_leftPoint.y(), m_rightPoint.y() );

    QRectF rect( m_leftPoint.x(), 0, w, h );
    return rect;
}

void vpTFQuad::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(widget);
    Q_UNUSED(option);

    if( !m_left || !m_right || !scene() )
        return;

    painter->setRenderHint( QPainter::Antialiasing );

    QRectF rect = boundingRect();
    QLinearGradient gradient( rect.topLeft(), rect.topRight() );
    gradient.setInterpolationMode(QLinearGradient::ColorInterpolation);

    QColor startColor = m_left ->GetColor();
    QColor endColor   = m_right->GetColor();

//    qreal totalHeight = scene()->height();
//    startColor.setAlpha( int(255*m_leftPoint.y()  / totalHeight) );
//    endColor  .setAlpha( int(255*m_rightPoint.y() / totalHeight) );

    gradient.setColorAt( 0, startColor );
    gradient.setColorAt( 1, endColor );

    painter->setPen(QPen(Qt::black, 0));
    painter->setBrush(gradient);

    QPoint points[4];
    points[0] = QPoint( m_leftPoint.x(),  vpTFNode::cDefaultMargin );
    points[1] = QPoint( m_rightPoint.x(), vpTFNode::cDefaultMargin );
    points[2] = QPoint( m_rightPoint.x(), m_rightPoint.y() );
    points[3] = QPoint( m_leftPoint.x(),  m_leftPoint.y()  );

    painter->drawConvexPolygon( points, 4 );
}

