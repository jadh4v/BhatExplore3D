#ifndef VPTRANSFERFUNCTIONEDITOR_H
#define VPTRANSFERFUNCTIONEDITOR_H

//#include <QDialog>
#include <QFrame>
#include "vpCommon.h"

class QGraphicsView;
class QGraphicsScene;
class QPushButton;
class QVBoxLayout;
class vpTFNode;
class vpTFQuad;
class vpTFScene;
class vpTransferFunction;
class vpVolumeViewer;

class LIBRARY_API vpTransferFunctionEditor : public QFrame
{
    Q_OBJECT
public:
    static const int cwidth; 
	static const int cheight; 
	static const int scene_width; 
	static const int scene_height;

    explicit vpTransferFunctionEditor( const vpTransferFunction* seed_tf, QWidget *parent = 0);
    ~vpTransferFunctionEditor();

    void AddNode( QPointF pos );

    void GetTransferFunction(vpTransferFunction &tfunc);

    void SetHistogram( const std::vector<float>& hist );

    /*
    static vpTransferFunction EditTransferFunctionDialog( const vpTransferFunction* tfunc_ref,
                                                          const std::vector<float> &histogram,
                                                          QWidget* mainWindow );
                                                          */

public slots:
    void slot_GraphChangedByNode();

signals:
    void sign_TransferFunctionUpdated( const vpTransferFunction* tfunc );

private:

    void init();

    QGraphicsView       *m_view;
    vpTFScene           *m_scene;
    QVBoxLayout         *m_layout;
    std::pair<double,double> m_reference_range;
};

#endif // VPTRANSFERFUNCTIONEDITOR_H
