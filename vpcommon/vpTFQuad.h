#ifndef VPTFQUAD_H
#define VPTFQUAD_H

#include <QGraphicsItem>

class vpTFNode;

class vpTFQuad : public QGraphicsItem
{
private:
    vpTFNode       *m_left, *m_right;
    QPointF m_leftPoint;
    QPointF m_rightPoint;

public:
    vpTFQuad(vpTFNode* leftNode, vpTFNode* rightNode);

    enum { Type = UserType + 2 };
    int type() const { return Type; }

    void adjust();

    bool HasNode( vpTFNode* node ) const;

    void ReplaceLeftNode( vpTFNode* newNode );

    void ReplaceRightNode( vpTFNode* newNode );

    vpTFNode* GetLeftNode() const;

    vpTFNode* GetRightNode() const;

//    int GetOtherNode(vpTFNode* node , vpTFNode* &otherNode) const;

protected:
    QRectF boundingRect() const;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

};

#endif // VPTFQUAD_H
