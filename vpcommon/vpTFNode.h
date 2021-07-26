#ifndef VPTFNODE_H
#define VPTFNODE_H

#include <QObject>
#include <QGraphicsItem>

QT_BEGIN_NAMESPACE
class QGraphicsSceneMouseEvent;
QT_END_NAMESPACE

class QGraphicsView;
class vpTFQuad;

class vpTFNode : public QObject, public QGraphicsItem
{
    Q_OBJECT
    Q_INTERFACES( QGraphicsItem )
private:
//    QList<vpTFQuad*> m_quadList;
    vpTFQuad *m_left_quad;
    vpTFQuad *m_right_quad;
    QColor m_color;

    void CheckCollision();

    void CheckCollision(vpTFNode *otherNode, int side);

public:
    static const int defaultRadius;
    static const int cDefaultMargin;

    vpTFNode( QGraphicsScene* scene, QGraphicsItem* parent=0 );

    enum { Type = UserType + 1 };
    int type() const { return Type; }

    QRectF boundingRect() const;

    QPainterPath shape() const;

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

    void SetMovable(bool val);

    QColor GetColor() const;

    void SetColor(QColor color);

    void SetLeftQuad(vpTFQuad* quad);

    void SetRightQuad(vpTFQuad* quad);

    bool ReplaceLeftQuadNode(vpTFNode *newNode);

    bool ReplaceRightQuadNode(vpTFNode *newNode);

    void PrepareDelete();

protected:

    QVariant itemChange(GraphicsItemChange change, const QVariant &value);

    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event);

    void mousePressEvent(QGraphicsSceneMouseEvent *event);

    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);

    void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

    void keyPressEvent(QKeyEvent *event);

    void dragLeaveEvent(QGraphicsSceneDragDropEvent *event);

signals:
    void sign_GraphChanged();

};

#endif // VPTFNODE_H
