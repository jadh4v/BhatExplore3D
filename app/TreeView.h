#pragma once
#include <map>
#include <QColor>
#include <QWidget>

class QPushButton;
class QTreeWidget;
class QTreeWidgetItem;
class Slider;

class TreeView : public QWidget
{
    Q_OBJECT
public:
    TreeView(QWidget* parent=nullptr);
    void AddItem(size_t id);
    int GetSelectedItem() const;

signals:
    void sign_colorChanged(size_t id, int r, int g, int b, int a);
    void sign_nodeChanged(size_t id);

private:
    QTreeWidget* m_Tree = 0;
    Slider* m_Slider = 0;
    QPushButton* m_Color = 0;
    struct Prop {
        QColor c = QColor(100,100,100);
        uchar opacity = 128;
    };
    std::map<size_t,Prop> m_Items;
    
private slots:
    void slot_PickColor();
    void slot_OpacityChanged(int);
    //void slot_RowChanged(int);
    void slot_ItemChanged(QTreeWidgetItem*, QTreeWidgetItem*);
};
