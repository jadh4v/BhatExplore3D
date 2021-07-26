#ifndef VPTFSCENE_H
#define VPTFSCENE_H

#include <QGraphicsScene>

class vpTFNode;
class vpTransferFunction;

class vpTFScene : public QGraphicsScene
{
    Q_OBJECT

private:
    QList<vpTFNode*>    m_nodes;
    std::vector<float>  m_histogram;

    void Initialize(const vpTransferFunction *seed_tf);

    vpTFNode* create_first_node();

    vpTFNode *create_last_node();

    static QColor interpolate(vpTFNode *back, vpTFNode *front, float x);

public:
    static const int scene_width, scene_height;

    vpTFScene( const vpTransferFunction* seed_tf, QObject* parent );

    void AddNode(QPointF pos);

    int GetTransferFunction( vpTransferFunction& tfunc ) const;

    void SetHistogram(const std::vector<float> &hist);

protected:

    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event);

    void keyPressEvent(QKeyEvent *event);

    void drawBackground(QPainter *painter, const QRectF &rect);

public slots:
    void DeleteNodeEvent();
    void slot_GraphChangedByNode();

signals:
    void sign_GraphChanged();

};


#endif // VPTFSCENE_H
