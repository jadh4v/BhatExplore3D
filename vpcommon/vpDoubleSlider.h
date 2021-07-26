#ifndef VPDOUBLESLIDER_H
#define VPDOUBLESLIDER_H

#include <QWidget>
#include "core/macros.h"
#include "vpCommon.h"

class QGraphicsScene;
class QGraphicsView;
class QGraphicsRectItem;
class vpDbsRect;
class vpDragableEnd;

class LIBRARY_API vpDoubleSlider : public QWidget
{
    Q_OBJECT
public:
    vpDoubleSlider( QWidget* parent=0 );
    qreal GetViewWidth();
    qreal GetViewHeight();
    void UpdateShape();
    void SetScalarRange(double range[2]);
    void SetColorWindow(int center, int width);
    void GetColorWindow(int& center, int& width);
    MacroGetSetMember( bool, m_lock, SyncLock )

protected:
    void resizeEvent(QResizeEvent* e);

public slots:
    void slot_childMoved();

signals:
    void sign_updated(int center, int width);

private:
    void init();
    void clamp();
    int   ToAbsCenter(float normalized_center);
    int   ToAbsWidth(float normalized_width);
    float ToNormCenter(int absolute_center);
    float ToNormWidth(int absolute_width);

    QGraphicsView       *m_view;
    QGraphicsScene      *m_scene;
    vpDbsRect           *m_rect;
    vpDragableEnd       *m_left, *m_right;
    qreal               m_norm_left_pos;
    qreal               m_norm_right_pos;
    bool                m_lock;
    double              m_scalar_range[2];
};

#endif // VPDOUBLESLIDER_H
