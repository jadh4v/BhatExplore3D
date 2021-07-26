#include <QPushButton>
#include <QLayout>
#include <QGraphicsView>
#include <QGraphicsScene>

#include "vpTFScene.h"
#include "vpTFNode.h"
#include "vpTFQuad.h"
#include "vpTransferFunction.h"
#include "vpTransferFunctionEditor.h"

#include "core/macros.h"

//const int vpTransferFunctionEditor::cwidth       = 1000;
//const int vpTransferFunctionEditor::cheight      = 300;
const int vpTransferFunctionEditor::cwidth       = 700;
const int vpTransferFunctionEditor::cheight      = 220;

void vpTransferFunctionEditor::init()
{
    this->m_view = nullptr;
    this->m_scene = nullptr;
    this->m_layout = nullptr;
    this->m_reference_range.first = -1024;
    this->m_reference_range.second = 3072;
}

vpTransferFunctionEditor::vpTransferFunctionEditor( const vpTransferFunction* seed_tf, QWidget *parent) :
    QFrame(parent)
{
    //setModal(false);
    setWindowTitle("Transfer Function Editor");

    resize( cwidth, cheight );
    setMaximumSize( cwidth, cheight );
    setMinimumSize( 300, cheight );

    QVBoxLayout* m_layout = new QVBoxLayout(this);
    this->setLayout( m_layout );

    m_scene = new vpTFScene(seed_tf, this);
    if( seed_tf )
        m_reference_range = seed_tf->GetReferenceRange();
    QColor bkColor(240,240,255);
    QBrush bkBrush( bkColor, Qt::HorPattern );
    m_scene->setBackgroundBrush( bkBrush );

    QObject::connect( m_scene, SIGNAL(sign_GraphChanged()), this, SLOT(slot_GraphChangedByNode()) );

    m_view  = new QGraphicsView(m_scene);
    m_layout->addWidget( m_view  );

    m_view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_view->centerOn(0,0);
    m_view->setSceneRect( 0, 0, vpTFScene::scene_width, vpTFScene::scene_height );
    m_view->scale(1,-1);
}

vpTransferFunctionEditor::~vpTransferFunctionEditor()
{
    MacroDelete(m_scene);
    MacroDelete(m_view);
    // MacroDelete(m_ok_button);
    // MacroDelete(m_layout);
}

void vpTransferFunctionEditor::GetTransferFunction( vpTransferFunction& tfunc )
{
    m_scene->GetTransferFunction( tfunc );
    tfunc.SetReferenceRange(m_reference_range);
}

/**
 * @brief vpTransferFunctionEditor::SetHistogram
 *          Takes normalized values (by maximum). Each value represents a bin. Maximum value
 *          will be represented as 75% of the height of the scene.
 * @param hist Values should be between 0 and 1.
 */
void vpTransferFunctionEditor::SetHistogram(const std::vector<float> &hist)
{
    m_scene->SetHistogram( hist );
}

/*
vpTransferFunction vpTransferFunctionEditor::EditTransferFunctionDialog(
                                                    const vpTransferFunction *tfunc_ref,
                                                    const std::vector<float>& histogram,
                                                    QWidget* mainWindow  )
{
    vpTransferFunctionEditor* editor = new vpTransferFunctionEditor( tfunc_ref, mainWindow );
    editor->SetHistogram( histogram );
    editor->exec();

    vpTransferFunction tf_new;
    editor->GetTransferFunction(tf_new);

    MacroDelete(editor);
    return tf_new;
}
*/

void vpTransferFunctionEditor::slot_GraphChangedByNode()
{
    vpTransferFunction* tfunc = new vpTransferFunction(m_reference_range);
    GetTransferFunction( *tfunc );
    //std::cout << "reference range: " << m_reference_range.first << ", " << m_reference_range.second << std::endl;
    emit sign_TransferFunctionUpdated( tfunc );
}



















