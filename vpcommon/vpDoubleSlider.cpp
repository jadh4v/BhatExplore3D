#include <QResizeEvent>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsRectItem>
#include <QLayout>
#include <QStyle>

#include "vpDoubleSlider.h"
#include "vpDragableEnd.h"
#include "vpDbsRect.h"
#include "core/macros.h"

void vpDoubleSlider::init()
{
    m_left = m_right = NULL;
    m_rect = NULL;
    m_scene = NULL;
    m_view = NULL;

    m_lock = false;
    m_norm_left_pos = 0.3;
    m_norm_right_pos = 0.7;
}

vpDoubleSlider::vpDoubleSlider( QWidget* parent ) : QWidget(parent)
{
    init();

    this->setBaseSize(100,50);
    this->setMaximumHeight(50);
    this->setMinimumHeight(50);
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    m_scene = new QGraphicsScene(this);
    m_view  = new QGraphicsView(m_scene);

    QVBoxLayout* layout = new QVBoxLayout(this);
    this->setLayout( layout );
    layout->addWidget( m_view  );

    m_view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_view->centerOn(0.5,0.5);
    m_view->setSceneRect( 0, 0, GetViewWidth(), GetViewHeight() );
    m_view->scale(1,-1);
    int bk_grey = 200;
    m_view->setBackgroundBrush(QBrush(QColor(bk_grey,bk_grey,bk_grey)));

    m_norm_left_pos  = 0.3;
    m_norm_right_pos = 0.7;

    int w = GetViewWidth();
    int h = GetViewHeight();
    QRectF rectf( 0, 0, (m_norm_right_pos - m_norm_left_pos)*w, h );
    m_rect = new vpDbsRect(rectf, this);
    m_scene->addItem(m_rect);
    m_rect->setPos( m_norm_left_pos*w, 0 );

    qreal ht = 30;
    //QLineF left ( vpDragableEnd::cLineWidth/2.0, 0, vpDragableEnd::cLineWidth/2.0, ht );
    QRectF left ( 0, 1, vpDragableEnd::cLineWidth, ht-1 );
    m_left = new vpDragableEnd(left, this, true);
    m_scene->addItem(m_left);
    m_left->setPos( m_norm_left_pos*w, 0 );

    //QLineF right( -vpDragableEnd::cLineWidth/2.0, 0, -vpDragableEnd::cLineWidth/2.0, ht );
    QRectF right( -vpDragableEnd::cLineWidth, 1, vpDragableEnd::cLineWidth+1, ht-1 );
    m_right = new vpDragableEnd(right, this, false);
    m_right->setPos( m_norm_right_pos*w, 0 );
    m_scene->addItem(m_right);

    m_left->SetOther(m_right);
    m_right->SetOther(m_left);

    m_rect->SetLeft(m_left);
    m_rect->SetRight(m_right);
}

void vpDoubleSlider::resizeEvent(QResizeEvent*)
{
    //MacroConfirm(e);
    UpdateShape();
}

void vpDoubleSlider::UpdateShape()
{
    qreal w = GetViewWidth();
    qreal h = GetViewHeight();
    m_view->setSceneRect( 0, 0, w, h );
    m_view->centerOn(0.5,0.5);

    m_left->setPos( m_norm_left_pos * w, 0 );
    m_right->setPos( m_norm_right_pos * w, 0 );

    m_rect->UpdateShape();
}


void vpDoubleSlider::slot_childMoved()
{
    m_rect->UpdateShape();

    m_norm_left_pos  = m_left->pos().x()  / (qreal)GetViewWidth();
    m_norm_right_pos = m_right->pos().x() / (qreal)GetViewWidth();

    clamp();

    // std::cout << "m_slider width = " << width() << ", scene width = " << m_view->width() << std::endl;
    // std::cout << "m_norm_left_pos = " << m_norm_left_pos << " m_norm_right_pos = " << m_norm_right_pos << std::endl;

    float norm_width  = m_norm_right_pos - m_norm_left_pos;
    float norm_center = m_norm_left_pos + norm_width/2.0f;

    int center = ToAbsCenter(norm_center);
    int width = ToAbsWidth(norm_width);

    emit sign_updated(center, width);
}

void vpDoubleSlider::clamp()
{
    m_norm_left_pos = m_norm_left_pos > 1? 1 : m_norm_left_pos;
    m_norm_right_pos = m_norm_right_pos > 1? 1 : m_norm_right_pos;

    m_norm_left_pos = m_norm_left_pos < 0? 0 : m_norm_left_pos;
    m_norm_right_pos = m_norm_right_pos < 0? 0 : m_norm_right_pos;

}

qreal vpDoubleSlider::GetViewWidth()
{
    MacroConfirmOrReturn(m_view,0);
    return (qreal)m_view->width();
}

qreal vpDoubleSlider::GetViewHeight()
{
    MacroConfirmOrReturn(m_view,0);
    return (qreal)m_view->height();
}

void vpDoubleSlider::GetColorWindow(int& center, int& width)
{
    m_rect->UpdateShape();

    m_norm_left_pos  = m_left->pos().x()  / (qreal)GetViewWidth();
    m_norm_right_pos = m_right->pos().x() / (qreal)GetViewWidth();

    clamp();

    float norm_width  = m_norm_right_pos - m_norm_left_pos;
    float norm_center = m_norm_left_pos + norm_width/2.0f;

    center = ToAbsCenter(norm_center);
    width  = ToAbsWidth(norm_width);
}

void vpDoubleSlider::SetColorWindow(int absCenter, int absWidth)
{
    float norm_center = ToNormCenter( absCenter );
    float norm_width  = ToNormWidth( absWidth  );

    m_norm_left_pos = norm_center - norm_width/2.0f;
    m_norm_right_pos = norm_center + norm_width/2.0f;

    clamp();
    UpdateShape();
    update();
}

void vpDoubleSlider::SetScalarRange(double range[2])
{
    m_scalar_range[0] = range[0];
    m_scalar_range[1] = range[1];

    // reset order if necessary
    if( m_scalar_range[1] < m_scalar_range[0])
        std::swap(m_scalar_range[0], m_scalar_range[1]);
}

float vpDoubleSlider::ToNormCenter(int absolute_center)
{
    float ret = float(absolute_center - m_scalar_range[0]) / float(m_scalar_range[1] - m_scalar_range[0]);
    return ret;
}

int vpDoubleSlider::ToAbsCenter(float normalized_center)
{
    int ret = (int)(normalized_center*(m_scalar_range[1]-m_scalar_range[0]) + m_scalar_range[0]);
    return ret;
}

float vpDoubleSlider::ToNormWidth(int absolute_width)
{
    float ret = float(absolute_width) / float(m_scalar_range[1] - m_scalar_range[0]);
    return ret;
}

int vpDoubleSlider::ToAbsWidth(float normalized_width)
{
    int ret = (int)(normalized_width * (m_scalar_range[1]-m_scalar_range[0]));
    return ret;
}



















