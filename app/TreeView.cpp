#include <QBoxLayout>
#include <QColorDialog>
#include <QPushButton>
#include <QTreeWidget>
#include <QLabel>
#include "core/macros.h"
#include "Slider.h"
#include "TreeView.h"

TreeView::TreeView(QWidget* parent) : QWidget(parent)
{
    this->setLayout(new QVBoxLayout());
    m_Tree = new QTreeWidget();
    m_Tree->setColumnCount(1);
    m_Tree->setHeaderLabel("Exploration Tree");
    m_Slider = new Slider("Opacity", Qt::Horizontal, 0, 255);
    m_Color = new QPushButton("Color");

    this->layout()->addWidget(m_Tree);
    this->layout()->addWidget(m_Slider);
    this->layout()->addWidget(m_Color);

    QObject::connect(m_Color, SIGNAL(clicked()), this, SLOT(slot_PickColor()));
    QObject::connect(m_Slider, SIGNAL(sign_valueChanged(int)), this, SLOT(slot_OpacityChanged(int)));
    //QObject::connect(m_Tree, SIGNAL(currentRowChanged(int)), this, SLOT(slot_RowChanged(int)));
    QObject::connect(m_Tree, SIGNAL(currentItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)), this, SLOT(slot_ItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)));
}

void TreeView::AddItem(size_t id)
{
    Prop p;
    m_Items[id] = p;
    QTreeWidgetItem* item = new QTreeWidgetItem(static_cast<QTreeWidget*>(nullptr), QStringList(QString::number(id)));
    item->setChildIndicatorPolicy(QTreeWidgetItem::DontShowIndicatorWhenChildless);
    if (m_Tree->currentItem())
        m_Tree->currentItem()->addChild(item);
    else
    {
        m_Tree->insertTopLevelItem(0, item);
        m_Tree->setCurrentItem(item);
    }
}

void TreeView::slot_PickColor()
{
    /*
    if (m_Tree->count() == 0)
        return;
    size_t id = (size_t)m_Tree->currentRow();
    */
    MacroConfirm(m_Tree->currentItem());
    auto item = m_Tree->currentItem();
    size_t id = (size_t)item->text(0).toUInt();

    if (id < m_Items.size())
    {
        Prop& prop = m_Items.at(id);
        QColor updatedColor = QColorDialog::getColor(prop.c, 0, "Pick Color");
        prop.c = updatedColor;
        emit(sign_colorChanged(id, updatedColor.red(), updatedColor.green(), updatedColor.blue(), prop.opacity));
    }
}

void TreeView::slot_OpacityChanged(int opacity)
{
    /*
    if (m_Tree->count() == 0)
        return;
    size_t id = (size_t)m_Tree->currentRow();
    */
    MacroConfirm(m_Tree->currentItem());
    auto item = m_Tree->currentItem();
    size_t id = (size_t)item->text(0).toUInt();

    if (id < m_Items.size())
    {
        Prop& prop = m_Items.at(id);
        prop.opacity = opacity;
        emit(sign_colorChanged(id, prop.c.red(), prop.c.green(), prop.c.blue(), prop.opacity));
    }
}

void TreeView::slot_ItemChanged(QTreeWidgetItem* current, QTreeWidgetItem* previous)
{
    MacroConfirm(current);
    size_t id = (size_t)current->text(0).toUInt();

    if (id < m_Items.size())
    {
        Prop& prop = m_Items.at(id);
        //emit(sign_colorChanged(id, prop.c.red(), prop.c.green(), prop.c.blue(), prop.opacity));
        m_Slider->SetValue(prop.opacity);
        emit(sign_nodeChanged(id));
    }
}


int TreeView::GetSelectedItem() const
{
    return m_Tree->currentItem()->text(0).toInt();
    //return m_Tree->currentIndex().row();
}
