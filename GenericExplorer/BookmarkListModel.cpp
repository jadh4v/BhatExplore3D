#include <QModelIndex>
#include "BookmarkListModel.h"
#include "Bookmark.h"
#include "DS/VoxelRow.h"
#include "macros.h"

using std::vector;
using sjDS::VoxelRow;

BookmarkListModel::BookmarkListModel(QObject *parent) : QAbstractListModel(parent)
{
}

QVariant BookmarkListModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if( role == Qt::DisplayRole && orientation == Qt::Horizontal )
        return QString("Bookmarks");
    else
        return QVariant();
}

int BookmarkListModel::rowCount(const QModelIndex &parent) const
{
    return (int)m_list.size();
}

int BookmarkListModel::columnCount(const QModelIndex &parent) const
{
    return 1;
}

QVariant BookmarkListModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    if( index.row() >= (int)m_list.size() )
        return QVariant();

    const Bookmark& item = *(m_list.at(index.row()));

    if( role == Qt::CheckStateRole && index.column()==0 )
        return static_cast<int>( item.IsChecked() ? Qt::Checked : Qt::Unchecked );

    if ( role == Qt::DisplayRole )
        return item.GetName();
    else
        return QVariant();

}

Bookmark* BookmarkListModel::AddBookmark(const std::vector<VoxelRow>& region)
{
    QModelIndex i = this->index( this->rowCount() );
    this->beginInsertRows( i, this->rowCount(), this->rowCount() );
    m_list.push_back(new Bookmark(region,QString::number(m_list.size())));
    this->endInsertRows();
    return m_list.back();
}

Bookmark* BookmarkListModel::AddBookmark(std::vector<VoxelRow>&& region)
{
    QModelIndex i = this->index( this->rowCount() );
    this->beginInsertRows( i, this->rowCount(), this->rowCount() );
    m_list.push_back(new Bookmark( region, QString::number(m_list.size())) );
    this->endInsertRows();
    return m_list.back();
}


void BookmarkListModel::AddBookmark(Bookmark* b)
{
    if( !b )
        return;

    QModelIndex i = this->index( this->rowCount() );
    this->beginInsertRows( i, this->rowCount(), this->rowCount() );
    m_list.push_back( b );
    this->endInsertRows();
}

bool BookmarkListModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if (!index.isValid())
        return false;

    if( index.row() >= (int)m_list.size() )
        return false;

    Bookmark& item = * m_list.at(index.row());

    if( role == Qt::CheckStateRole && index.column()==0)
    {
        item.SetChecked( value.toBool() );
        emit dataChanged( index, index );
        emit sign_checkStateChanged( index );
        return true;
    }
    else if( role == Qt::EditRole && index.column() == 0 )
    {
        item.SetName(QString(value.toString()));
        emit dataChanged( index, index );
        return true;
    }

    return false;
}

Qt::ItemFlags BookmarkListModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return 0;

    Qt::ItemFlags flags = QAbstractListModel::flags(index);
    {
        flags |= Qt::ItemIsSelectable;
        flags |= Qt::ItemIsEditable;
    }

    if( index.column() == 0 )
        flags |= Qt::ItemIsUserCheckable;

    //return QAbstractItemModel::flags(index);
    return flags;
}

void BookmarkListModel::RemoveBookmark(int rowNumber)
{
    QModelIndex i = this->index( this->rowCount() );
    this->beginRemoveRows( i, rowNumber, rowNumber );

    int cnt = 0;
    for( auto iter = m_list.begin(); iter != m_list.end(); ++iter, ++cnt )
    {
        if( cnt == rowNumber )
        {
            MacroDelete( *iter );
            m_list.erase( iter );
            break;
        }
    }

    this->endRemoveRows();
}

const std::vector<Bookmark*>& BookmarkListModel::GetAllBookmarks() const
{
    return m_list;
}

std::vector<Bookmark*> BookmarkListModel::GetCheckedBookmarks() const
{
    std::vector<Bookmark*> ret;
    for( auto& b : m_list )
    {
        if( b->IsChecked() )
            ret.push_back(b);
    }

    return ret;
}
















