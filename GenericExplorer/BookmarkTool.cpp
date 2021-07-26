#include <set>
#include <QApplication>
#include <QMessageBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QListView>
//#include <QListWidget>
#include <QTableView>
#include <QAbstractListModel>
//#include <QPushButton>
#include <QToolButton>
//#include <QStringListModel>
#include <QHeaderView>
#include <QGroupBox>
#include "BookmarkTool.h"
#include "BookmarkListModel.h"
#include "DialogOpticalProperties.h"
#include "Graphics/VolumeRenderer.h"
#include "DS/VoxelRow.h"
#include "DS/Image.h"
#include "VolumeStats.h"
#include "macros.h"

using std::set;
using std::vector;
using sjDS::VoxelRow;

const float BookmarkTool::cOpacityRange[2] = {0.0f, 0.2f};
const float BookmarkTool::cDefaultPersistenceThreshold = 0.01f;

void BookmarkTool::init()
{
    m_add       = nullptr;
    m_remove    = nullptr;
    m_save       = nullptr;
    m_load    = nullptr;
    m_bookListModel = nullptr;
    m_bookListView  = nullptr;
    m_volume = nullptr;
}

BookmarkTool::~BookmarkTool()
{
    MacroDelete( m_add );
    MacroDelete( m_remove );
    MacroDelete( m_save );
    MacroDelete( m_load );
    MacroDelete( m_bookListView );
    MacroDelete( m_bookListModel );
}

BookmarkTool::BookmarkTool(QWidget* parent) : QGroupBox(parent)
{
    QVBoxLayout* layout = new QVBoxLayout;
    setLayout(layout);

    size_t requiredSize = 0;
    getenv_s(&requiredSize, 0, 0, "ICONS_DIR");
    vector<char> icons_env_path(requiredSize);
    getenv_s(&requiredSize, icons_env_path.data(), requiredSize, "ICONS_DIR");
    QString path_icons(icons_env_path.data());


    m_add = new QToolButton(this);
    m_add->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    m_add->setIcon(QIcon(path_icons + "/add.png"));
    m_add->setText("Add");
    m_remove = new QToolButton(this);
    m_remove->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    m_remove->setIcon(QIcon(path_icons + "/dust-bin.png"));
    m_remove->setText("Remove");
    m_save = new QToolButton(this);
    m_save->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    m_save->setIcon(QIcon(path_icons + "/floppy-disk.png"));
    m_save->setText("Save");
    m_load = new QToolButton(this);
    m_load->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    m_load->setIcon(QIcon(path_icons + "/load.png"));
    m_load->setText("Load");
    m_merge = new QToolButton(this);
    m_merge->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    m_merge->setIcon(QIcon(path_icons + "/merge.png"));
    m_merge->setText("Merge");

    m_add->setFixedSize(60, 50);
    m_remove->setFixedSize(60, 50);
    m_save->setFixedSize(60, 50);
    m_load->setFixedSize(60, 50);
    m_merge->setFixedSize(60, 50);

    m_bookListView = new QTableView(this);
    m_bookListView->setSizePolicy( QSizePolicy::Expanding, QSizePolicy::Expanding );
    //m_bookListView->setSizePolicy( QSizePolicy::Fixed, QSizePolicy::Fixed );
    //m_bookListView->setFixedSize(230,550);
    //m_bookListView->setFixedSize(230,400);
    m_bookListModel = new BookmarkListModel(this);
    m_bookListView->setModel(m_bookListModel);
    m_bookListView->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    //m_bookListView->setFixedHeight(340);

    QHBoxLayout* my_lists = new QHBoxLayout;
    my_lists->addWidget(m_bookListView);

    QGridLayout* button_grid = new QGridLayout;
    button_grid->addWidget(m_add,0,0);
    button_grid->addWidget(m_remove,0,1);
    //button_grid->addWidget(m_edit,0,2);
    button_grid->addWidget(m_save,1,0);
    button_grid->addWidget(m_load,1,1);
    button_grid->addWidget(m_merge,1,2);

    layout->addItem( button_grid );
    //QSpacerItem* spacer = new QSpacerItem( 10, 70, QSizePolicy::Preferred, QSizePolicy::Expanding );
    //layout->addItem( spacer );
    layout->addItem( my_lists );
    this->setTitle("Bookmark Controls");

    //vpTransferFunction tf;
    //m_colorDialog = new DialogOpticalProperties( tf, this, Qt::Dialog );


    QObject::connect( m_add,    SIGNAL(clicked(bool)), this, SIGNAL(sign_addTriggered()) );
    QObject::connect( m_remove, SIGNAL(clicked(bool)), this, SLOT(slot_removeClicked()) );
    QObject::connect( m_save,   SIGNAL(clicked(bool)), this, SLOT(slot_saveClicked()) );
    QObject::connect( m_load,   SIGNAL(clicked(bool)), this, SLOT(slot_loadClicked()) );
    QObject::connect( m_merge,  SIGNAL(clicked(bool)), this, SLOT(slot_mergeClicked()) );

    QObject::connect( m_bookListModel, SIGNAL(sign_checkStateChanged(QModelIndex)), this, SLOT(slot_bookChecked(QModelIndex)) );

    QItemSelectionModel* selModel = m_bookListView->selectionModel();
    QObject::connect( selModel, SIGNAL(currentChanged(QModelIndex,QModelIndex)), this, SLOT(slot_currentChanged(QModelIndex,QModelIndex)) );
}

Bookmark* BookmarkTool::AddBookmark(const vector<VoxelRow> &region)
{
    Bookmark* b = m_bookListModel->AddBookmark( region );
    VolumeStats vs(*m_volume);
    vs.ComputeHistogram( b->GetRegion(), cDefaultBinSize );
    float opacityRange[] = {cOpacityRange[0], cOpacityRange[1] };
    //std::pair<double, double> ref_range(0.0, 1024.0);
    std::pair<double, double> ref_range(m_ref_range[0], m_ref_range[1]);
    b->SetTF( vpTransferFunction::AutoCompute(vs.GetHistogram(), opacityRange, ref_range, cDefaultMaxFeatureCount, cDefaultPersistenceThreshold) );
    return b;
}

Bookmark* BookmarkTool::AddBookmark(vector<VoxelRow>&& region)
{
    Bookmark* b = m_bookListModel->AddBookmark( region );
    VolumeStats vs(*m_volume);
    vs.ComputeHistogram( b->GetRegion(), cDefaultBinSize );
    float opacityRange[] = {cOpacityRange[0], cOpacityRange[1] };
    std::pair<double, double> ref_range(0.0, 1024.0);
    b->SetTF( vpTransferFunction::AutoCompute(vs.GetHistogram(), opacityRange, ref_range, cDefaultMaxFeatureCount, cDefaultPersistenceThreshold) );
    return b;
}

void BookmarkTool::AddBookmark(Bookmark* b)
{
    m_bookListModel->AddBookmark( b );
}

void BookmarkTool::Clear()
{
    m_bookListView->selectAll();
    slot_removeClicked();
}

void BookmarkTool::UncheckAll()
{
    const vector<Bookmark*>& bookmarks = m_bookListModel->GetAllBookmarks();

    for( Bookmark* b : bookmarks)
        b->SetChecked(false);
}

std::vector<Bookmark*> BookmarkTool::GetCheckedBookmarks() const
{
    return m_bookListModel->GetCheckedBookmarks();
}

void BookmarkTool::slot_removeClicked()
{
    int ret = QMessageBox::warning( this, "Delete Bookmarks?", "Are you sure?", QMessageBox::Yes, QMessageBox::No );
    if( ret == QMessageBox::No )
        return;

    set<uint> regs;
    QItemSelectionModel* selModel = m_bookListView->selectionModel();
    QModelIndexList sel = selModel->selectedRows();

    int disp = 0;
    for( auto i = sel.begin(); i != sel.end(); ++i )
    {
        int pos = i->row() - disp;
        if( pos >= m_bookListModel->rowCount() )
            continue;

        Bookmark& b = m_bookListModel->GetBookmark(pos);
        auto v = b.GetRegion();
        m_bookListModel->RemoveBookmark( pos );
        ++disp;
    }

    // Update rendering.
    update_rendering();
}

void BookmarkTool::slot_mergeClicked()
{
    int ret = QMessageBox::warning( this, "Merge Selected Bookmarks?", "Are you sure?", QMessageBox::Yes, QMessageBox::No );
    if( ret == QMessageBox::No )
        return;

    set<uint> regs;
    QItemSelectionModel* selModel = m_bookListView->selectionModel();
    QModelIndexList sel = selModel->selectedRows();

    std::vector<sjDS::VoxelRow> voxRows;
    for( auto i = sel.begin(); i != sel.end(); ++i )
    {
        int pos = i->row();
        if( pos >= m_bookListModel->rowCount() )
            continue;

        Bookmark& b = m_bookListModel->GetBookmark(pos);
        auto v = b.GetRegion();
        voxRows.insert(voxRows.end(), v.begin(), v.end());
    }
    this->AddBookmark(new Bookmark(voxRows, QString("Merged")));

    // Update rendering.
    update_rendering();
}

void BookmarkTool::slot_bookChecked(const QModelIndex& index)
{
    if(!index.isValid())
        return;

    Bookmark& b= m_bookListModel->GetBookmark(index.row());

    update_rendering();
}

void BookmarkTool::update_rendering()
{
    const vector<Bookmark*>& books = m_bookListModel->GetAllBookmarks();
    emit sign_updateRendering(books);
}

void BookmarkTool::slot_reRootTreeToRegion(const QModelIndex& index)
{
    if( index.isValid() )
    {
        if( QApplication::mouseButtons() == Qt::MidButton )
        {
            QVariant d = index.data();
            uint id = d.toUInt();
            emit sign_reRootRegion(id);
        }
    }
}

void BookmarkTool::slot_saveClicked()
{
    emit sign_saveMe( m_bookListModel->GetAllBookmarks() );
}

void BookmarkTool::slot_loadClicked()
{
    emit sign_loadMe(this);
}

void BookmarkTool::slot_tfupdate(const vpTransferFunction* tfunc)
{
    QItemSelectionModel* model = m_bookListView->selectionModel();
    QModelIndexList sel = model->selectedIndexes();
    if( sel.isEmpty() )
        return;

    //vpTransferFunction tf = m_colorDialog->GetTF();
    for(int i=0; i < sel.size(); ++i)
    {
        Bookmark& b_i = m_bookListModel->GetBookmark(sel[i].row());
        b_i.SetTF(*tfunc);
        b_i.SetMode( VolumeRenderer::RENMODE_DEFAULT );
    }
    update_rendering();
}

void BookmarkTool::SetVolume(const sjDS::Image* vol)
{
    m_volume = vol;
    std::vector<VoxelRow> region;
    VoxelRow r(0, sjDS::voxelNo_t(m_volume->GetArraySize()-1));
    region.push_back(r);
    //this->AddBookmark(new Bookmark(region, QString("Full Volume")));
    auto b = this->AddBookmark(region);
    b->SetName("Full Volume");
}

void BookmarkTool::slot_currentChanged(QModelIndex current, QModelIndex previous)
{
    current.row();
    const auto& b = m_bookListModel->GetBookmark(current.row());
    auto tf = b.GetTF();
    emit(sign_bookmarkSelected(tf));
}
