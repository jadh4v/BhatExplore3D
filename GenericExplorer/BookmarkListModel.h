#pragma once

#include <vector>
#include <QAbstractListModel>
//#include <QAbstractTableModel>
#include "Bookmark.h"

namespace sjDS{
    class VoxelRow;
}

class BookmarkListModel : public QAbstractListModel
//class BookmarkListModel : public QAbstractTre
{
    Q_OBJECT
public:
    BookmarkListModel(QObject *parent=0);

    virtual int rowCount(const QModelIndex &parent = QModelIndex()) const;

    virtual int columnCount(const QModelIndex &parent) const;

    virtual QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;

    virtual QVariant headerData(int section, Qt::Orientation orientation, int role) const;

    Bookmark* AddBookmark(const std::vector<sjDS::VoxelRow>& region);
    Bookmark* AddBookmark(std::vector<sjDS::VoxelRow>&& region);
    void AddBookmark(Bookmark* b);
    void RemoveBookmark( int rowNumber );

    Qt::ItemFlags flags(const QModelIndex &index) const;

    bool setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole);

    Bookmark& GetBookmark(int idx) { return *(m_list.at(idx)); }

    const std::vector<Bookmark*>& GetAllBookmarks() const;

    std::vector<Bookmark*> GetCheckedBookmarks() const;

private:
    std::vector<Bookmark*> m_list;

signals:
    void sign_checkStateChanged( const QModelIndex& index );

};

