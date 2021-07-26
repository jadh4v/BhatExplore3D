#include <math.h>

#include <QPainter>
#include <QGraphicsSceneMouseEvent>
#include <QKeyEvent>

#include "vpTransferFunctionEditor.h"
#include "vpTransferFunction.h"
#include "vpTFNode.h"
#include "vpTFQuad.h"
#include "vpTFScene.h"

#include "core/macros.h"

//const int vpTFScene::scene_width  = 975;
const int vpTFScene::scene_width  = 675;
const int vpTFScene::scene_height = 200;

typedef std::vector<vpTransferFunction::TFPoint>::const_iterator TFIter;

vpTFScene::vpTFScene( const vpTransferFunction* seed_tf, QObject* parent = 0 )
    : QGraphicsScene(parent)
{
    setSceneRect( 0, 0, vpTFScene::scene_width, vpTFScene::scene_height );

    if( seed_tf )
    {
        Initialize(seed_tf);
    }
    else
    {
        vpTFNode* first_node = create_first_node();

        vpTFNode* node1 = new vpTFNode(this);
        node1->setPos( 100, 100 );
//        this->addItem(node1);
        m_nodes << node1;

        vpTFNode* last_node = create_last_node();

        vpTFQuad* quad1 = new vpTFQuad(first_node, node1);
        this->addItem(quad1);

        vpTFQuad* quad2 = new vpTFQuad(node1, last_node);
        this->addItem(quad2);
    }

    //---------------------------
    // Draw Axes
    //---------------------------
    QPen pen;
    pen.setWidth(2);
    QBrush brush( Qt::black, Qt::SolidPattern);

    //---------------------------
    // Draw X-Axes
    //---------------------------
    int margin = vpTFNode::cDefaultMargin;

    // draw horizontal line
    QGraphicsLineItem* x_axis  = new QGraphicsLineItem( margin, margin, this->width(), margin );
    x_axis->setPen(pen);
    this->addItem( x_axis );

    // draw arrow head
    QPolygon arrow;
    arrow.push_back( QPoint(this->width(), margin) );
    arrow.push_back( QPoint(this->width()-10, margin+5) );
    arrow.push_back( QPoint(this->width()-10, margin-5) );
    QGraphicsPolygonItem* arrow_head_X = new QGraphicsPolygonItem( arrow );
    arrow_head_X->setBrush(brush);
    this->addItem(arrow_head_X);

    //---------------------------
    // Draw Y-Axis
    //---------------------------
    // draw vertical line
    QGraphicsLineItem* y_axis  = new QGraphicsLineItem( margin, margin, margin, this->height() );
    y_axis->setPen(pen);
    this->addItem( y_axis );

    // draw arrow head
    arrow.clear();
    arrow.push_back( QPoint( margin,   this->height()    ) );
    arrow.push_back( QPoint( margin-5, this->height()-10 ) );
    arrow.push_back( QPoint( margin+5, this->height()-10 ) );
    QGraphicsPolygonItem* arrow_head_Y = new QGraphicsPolygonItem( arrow );
    arrow_head_Y->setBrush(brush);
    this->addItem(arrow_head_Y);

    //---------------------------
    // Axis Labels
    //---------------------------
    // x-axis label
    QGraphicsTextItem* x_label = new QGraphicsTextItem("Intensity");
    x_label->setPos( (scene_width-margin)/2 + margin, margin );
    QTransform t;
    t.scale( 1, -1);
    x_label->setTransform(t);
    this->addItem( x_label );

    // y-axis label
    QGraphicsTextItem* y_label = new QGraphicsTextItem("Opacity");
    int dw = (int)y_label->boundingRect().width();
    y_label->setPos( 0, (scene_height-margin)/2 + margin - dw/2 );
    t.rotate(-90);
    y_label->setTransform(t);
    this->addItem( y_label );
}

void vpTFScene::Initialize( const vpTransferFunction* seed_tf )
{
    vpTFNode* first_node = create_first_node();

    for( TFIter iter = seed_tf->Begin(); iter != seed_tf->End(); iter++ )
    {
        if( iter->x <= 0 )
            continue;
        if( iter->x >= 1 )
            continue;

        vpTFNode* node = new vpTFNode(this);
        int margin = vpTFNode::cDefaultMargin;
        float x = iter->x * (scene_width - margin) + margin;
//        float y = iter->color.a * scene_height;
        float y = sqrt(iter->color.a) * (scene_height - margin) + margin;
        node->setPos( x, y );

//        node->SetColor( QColor(iter->color.r, iter->color.g, iter->color.b, iter->color.a) );
        QColor currColor;
        currColor.setRedF  ( iter->color.r );
        currColor.setGreenF( iter->color.g );
        currColor.setBlueF ( iter->color.b );
        currColor.setAlphaF( iter->color.a );
        node->SetColor(currColor);

        m_nodes << node;

        vpTFQuad* quad = new vpTFQuad( first_node, node );
        this->addItem(quad);

        first_node = node;
    }

    vpTFNode* last_node = create_last_node();
    vpTFQuad* quad = new vpTFQuad(first_node, last_node);
    this->addItem(quad);
}

void vpTFScene::drawBackground(QPainter *painter, const QRectF &rect)
{
    int margin = vpTFNode::cDefaultMargin;
    QColor bkColor(240,240,255);
    QBrush bkBrush( bkColor, Qt::HorPattern );
    setBackgroundBrush( bkBrush );

    QGraphicsScene::drawBackground( painter, rect );

    QColor histColor(150,150,150,155);
    QBrush histBrush( histColor );
    painter->setBrush(histBrush);
    painter->setPen(Qt::NoPen);
    const float delx = (float)(width()-margin) / (float)m_histogram.size();
    float xp = margin;
    for(size_t i=0; i < m_histogram.size(); i++)
    {
        float h = (height()-margin)*sqrt(m_histogram[i]);
        QRectF rect(xp, margin, delx, h);
        painter->drawRect(rect);
        xp += delx;
    }
}

vpTFNode *vpTFScene::create_first_node()
{
    vpTFNode* first_node = new vpTFNode(this);
//    this->addItem(first_node);
    first_node->setPos( vpTFNode::cDefaultMargin, vpTFNode::cDefaultMargin );
    first_node->SetColor( QColor(0,0,0,0) );
    first_node->SetMovable(false);
    first_node->setVisible(false);
    m_nodes << first_node;

    return first_node;
}

vpTFNode *vpTFScene::create_last_node()
{
    vpTFNode* last_node = new vpTFNode(this);
//    this->addItem(last_node);
    last_node->setPos( scene_width, vpTFNode::cDefaultMargin );
    last_node->SetColor( QColor(0,0,0,0) );
    last_node->SetMovable(false);
    last_node->setVisible(false);
    m_nodes << last_node;

    return last_node;
}

void vpTFScene::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event)
{
    QGraphicsScene::mouseDoubleClickEvent(event);
    if( !event->isAccepted() )
    {
        QPointF pos = event->scenePos();
        this->AddNode( pos );
    }
}

void vpTFScene::keyPressEvent(QKeyEvent *event)
{
    QGraphicsScene::keyPressEvent(event);

    if( !event->isAccepted() )
    {
        if( event->type() == QKeyEvent::KeyPress && event->key() == Qt::Key_Delete )
        {
            MacroMessage("Delete Key Pressed.");
            DeleteNodeEvent();
        }
    }
}

QColor vpTFScene::interpolate(vpTFNode* back, vpTFNode* front, float x )
{
    QColor ret;
    if( !back || !front )
        return ret;

    float x0 = back->x();
    float x1 = front->x();
    QColor color0 = back->GetColor();
    QColor color1 = front->GetColor();

    if( x < x0 || x1 < x )
        return ret;

    float w = x1 - x0;
    float d = x  - x0;

    float factor = d/w;

    ret.setRedF  ( color0.redF()   + (color1.redF()   - color0.redF())   * factor );
    ret.setGreenF( color0.greenF() + (color1.greenF() - color0.greenF()) * factor );
    ret.setBlueF ( color0.blueF()  + (color1.blueF()  - color0.blueF())  * factor );
    ret.setAlphaF( color0.alphaF() + (color1.alphaF() - color0.alphaF()) * factor );

    return ret;
}

void vpTFScene::AddNode(QPointF pos)
{
    if( pos.x() <= 0 || pos.x() >= scene_width )
        return;

    vpTFNode* newNode = new vpTFNode(this);
    newNode->setPos( pos );

    int front_node_pos = 0;
    for(int i=0; i < m_nodes.size(); i++)
    {
        const vpTFNode* currNode = m_nodes.at(i);
        if( newNode->x() < currNode->x() )
        {
            front_node_pos = i;
            break;
        }
    }

    if( front_node_pos > 0)
    {
        vpTFNode* back  = m_nodes.at( front_node_pos-1 );
        vpTFNode* front = m_nodes.at( front_node_pos   );

        QColor newColor = interpolate( back, front, pos.x() );
        newNode->SetColor(newColor);

        back->ReplaceRightQuadNode( newNode );

        vpTFQuad* newQuad = new vpTFQuad( newNode, front );
        this->addItem(newQuad);

        m_nodes.insert( front_node_pos, newNode );
    }
}

/**
 * @brief vpTFScene::GetTransferFunction
 *        The function will generate a transfer function based on the edited graph so far.
 * @param tfunc Transfer function return object passed by reference.
 * @return 0 - failure, 1 - success
 */
int vpTFScene::GetTransferFunction( vpTransferFunction &tfunc ) const
{
    for( int i=0; i < m_nodes.size(); i++)
    {
        vpTransferFunction::TFPoint tf_point;
        const vpTFNode* currNode = m_nodes.at(i);
        if( !currNode )
        {
            MacroWarning("Null node encountered.");
            return 0;
        }
        int margin = vpTFNode::cDefaultMargin;
        tf_point.x = (currNode->pos().x() - margin) / (scene_width - margin);

        QColor currColor = currNode->GetColor();
        tf_point.color.r = currColor.redF();
        tf_point.color.g = currColor.greenF();
        tf_point.color.b = currColor.blueF();
        tf_point.color.a = currColor.alphaF();

        tfunc.AddPoint(tf_point);
    }

    return 1;
}

/**
 * @brief vpTFScene::DeleteNodeEvent
 * Find selected node and remove from scene.
 */
void vpTFScene::DeleteNodeEvent()
{
    for( int i=0; i < m_nodes.size(); i++)
    {
        vpTFNode* currNode = m_nodes.at(i);
        if( currNode->isSelected() )
        {
            m_nodes.removeAt(i);
            currNode->PrepareDelete();
            this->removeItem(currNode);
        }
    }
}

void vpTFScene::slot_GraphChangedByNode()
{
    emit sign_GraphChanged();
}

void vpTFScene::SetHistogram( const std::vector<float>& hist )
{
    m_histogram = hist;
}






















