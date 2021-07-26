#pragma once

#include <set>
#include <vector>
#include <QWidget>
#include <vtkImageData.h>
#include <vtkSmartPointer.h>

#include "Graphics/TextureContext.h"
#include "DS/Image.h"
#include "DS/CompressedSegmentation.h"

class QSlider;
class AxialView;
class VolumeRenderer;
class BookmarkTool;
class Bookmark;

namespace sjDS{
    class Image;
    class VoxelRow;
    class Histogram;
    class ImageSegmentation;
}

/// Constructs and handles Merger user-interface.
class VolumeVisualizer : public QWidget
{
    Q_OBJECT
public:
    VolumeVisualizer(QWidget* parent=nullptr);
    ~VolumeVisualizer();
    //void SetImage(const sjDS::Image& mainVolume );
    void SetImage(vtkSmartPointer<vtkImageData> mainVolume);
    void SetSeg(const sjDS::CompressedSegmentation& seg) { m_Seg = seg; }
    void ClipVolume(float value[6]);

protected:
    //virtual void resizeEvent(QResizeEvent *event);
    virtual void showEvent(QShowEvent* event);

private:

    void _CreateVolumeTexture_RGBA();
    void _CreateVolumeTexture_GrayScale();
    void _ConstructOpticalProperties(size_t numOfSegments);
    void _ConstructOpticalProperties(const std::vector<std::vector<uint>>& segTFs, const std::vector<std::pair<uint,uint>>& segRanges);
    void _UpdateRenderedSegmentation(const std::set<size_t>& voxels, uint value);
    void _UpdateRenderedSegmentation(const std::vector<sjDS::VoxelRow>& voxels, uint value);
    void _UpdateRenderedSegmentationAccumulative(const std::vector<uint>& superVoxels, uint& highestLabelCount);
    void _UpdateRenderedSegmentationAccumulative(const std::vector<sjDS::VoxelRow>& voxRows, uint& highestLabelCount);
    int _GetScalarRange(type_uint range[2], const std::vector<sjDS::VoxelRow>& region) const;

    QWidget*          m_Central               = nullptr;
    TextureContext*   m_Context               = nullptr;
    QSlider*          m_AxialSlider           = nullptr;
    AxialView*        m_AxialView             = nullptr;
    VolumeRenderer*   m_VolRen                = nullptr;

    sjDS::Image       m_Optical;
    sjDS::Image*      m_RenderedSegmentation  = nullptr;
    sjDS::CompressedSegmentation m_Seg;
    //const sjDS::Image* m_OriginalImage = nullptr;
    const sjDS::Histogram* m_Histogram = nullptr;
    vtkSmartPointer<vtkImageData> m_OriginalImage = nullptr;
    TextureHandle     m_VolTex;

private slots:
    //void slot_AddCurrentToBookmark();
    void slot_UpdateRendering(const std::vector<Bookmark*>& bookmarks);
    void slot_saveBookmarks(const std::vector<Bookmark*>& all_bookmarks);
    void slot_loadBookmarks(BookmarkTool* bookmarkTool);
    void slot_dice(bool);

signals:
    void sign_makeSnapshot(int);

};
