#pragma once

#include <vector>
#include <QFrame>
#include <QGroupBox>
#include <QItemSelection>
#include "Bookmark.h"

class QToolButton;
//class QPushButton;
//class QListWidget;
class QTableView;
class BookmarkListModel;
class DialogOpticalProperties;

namespace sjDS{
class Image;
}

namespace sjDS{
    class VoxelRow;
}

class BookmarkTool : public QGroupBox
{
    Q_OBJECT
public:
    static const uint cDefaultBinSize = 32;
    static const float cOpacityRange[2];
    static const int cDefaultMaxFeatureCount = 11;
    static const float cDefaultPersistenceThreshold;
    BookmarkTool(QWidget* parent=0);
    virtual ~BookmarkTool();

    Bookmark* AddBookmark(const std::vector<sjDS::VoxelRow>& region);
    Bookmark* AddBookmark(std::vector<sjDS::VoxelRow>&& region);
    void AddBookmark(Bookmark* b);
    void Clear();
    void UncheckAll();
    std::vector<Bookmark*> GetCheckedBookmarks() const;
    //MacroSetMember( const sjDS::Image*, m_volume, Volume )
    void SetVolume(const sjDS::Image* vol);
    void SetRefRange(double rmin, double rmax) { m_ref_range[0] = rmin; m_ref_range[1] = rmax; }
    MacroSetMember(DialogOpticalProperties*, m_colorDialog, TFEditorWidget)

private:
    void init();
    void update_rendering();

    const sjDS::Image* m_volume =nullptr;
    QTableView*   m_bookListView =nullptr;
    BookmarkListModel* m_bookListModel =nullptr;
    QToolButton* m_add =nullptr;
    QToolButton* m_remove =nullptr;
    QToolButton* m_save =nullptr;
    QToolButton* m_load =nullptr;
    QToolButton* m_merge =nullptr;
    DialogOpticalProperties* m_colorDialog = nullptr;
    double m_ref_range[2] = { 0, 1024 };

private slots:
    void slot_removeClicked();
    void slot_saveClicked();
    void slot_loadClicked();
    void slot_mergeClicked();
    void slot_bookChecked(const QModelIndex& index);
    void slot_reRootTreeToRegion(const QModelIndex& index);
    void slot_currentChanged(QModelIndex, QModelIndex);

public slots:
    void slot_tfupdate(const vpTransferFunction* tfunc);

signals:
    void sign_addTriggered( );
    void sign_updateRendering(const std::vector<Bookmark*>& bookmarks);
    void sign_reRootRegion( uint id );
    void sign_saveMe(const std::vector<Bookmark*>& all_bookmarks);
    void sign_loadMe(BookmarkTool*);
    void sign_bookmarkSelected(const vpTransferFunction&);
};

