#pragma once

#include "vpCanvas.h"
#include "vpStructures.h"
#include "ds/BitImage.h"

typedef TextureHandle Texh;

namespace sjDS{
class Grid;
}

//##################################################################################################
class ContinuousSliceView : public vpCanvas
{
Q_OBJECT

public:
    ContinuousSliceView(QWidget *parent = 0);
    ~ContinuousSliceView();

    /// Set the opengl texture ID for the volume to render.
    void SetVolumeTextureArray( uint handle );

    /// Set the opengl texture ID for the segmentation to render.
    void SetSegmentTextureArray( uint handle );

    /// Set the opengl texture ID for the segmentation to render.
    void SetGradientTextureArray( uint handle );

    /// Set the opengl texture ID for the Optical Properties of each segment.
    void SetOpticalTexture( uint handle );

    /// Set the dicom color window settings if the input volume is a dicom image.
    /// First value  (window[0]) is the center of the window.
    /// Second value (window[1]) is the width  of the window.
    /// All values are clapped to this window.
    /// Contrast and brightness is determined by these settings.
    void SetDicomColorWindow( float window[2] );
    void SetSpacing(const double spacing[2]);
    /// Set a grid object that describes the input volume grid.
    void SetGrid(const sjDS::Grid* g);
    /// Maximum scalar value of in the volume dataset.
    MacroSetMember(int, m_voltex_maxvalue, VolumeTextureMaxValue)
    /// Set current slice of the volume to be displayed as image.
    MacroSetMember(float, m_current_slice, CurrentSlice)
    MacroSetMember(bool, m_repainting_flag, EnableRepainting)

    const sjDS::BitImage& SelectionMask() const { return m_selectionMask; }

    void ResetScribbleChanged() { m_ScribbleChanged = false; }
    bool IsScribbleChanged() const { return m_ScribbleChanged; }
    void SetSliceShaderProgram(QString vertex_shader, QString fragment_shader);
    void ToggleInterpolation();

//##################################################################################################
protected:
    virtual void initializeGL();
    virtual void wheelEvent(QWheelEvent* event);
    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent(QMouseEvent* event);
    virtual void keyPressEvent(QKeyEvent *event);

private:
    void initialize();
    /// Main opengl draw routine. Draw 3D stuff by overriding this method.
    virtual void draw_gl();
    /// Override to draw text etc over the 3D GL drawing drawn by draw_gl().
    virtual void draw_over(QPainter* painter);
    void reconstruct_slice_plane();
    void update_model_matrix();
    void slice_scaling();
    int get_slice_aspect_ratio(double ratio[2]) const;
    void _FindClickedVoxel(QMouseEvent* event);
    int _CreateSelectionTexture();
    int _UpdateSelectionTexture(bool currentSlice);
    void _AdjustColorWindow(const QPoint & delta);
    void _Zoom(const QPoint & delta);
    void _Pan(const QPoint & delta);

    //###################################
    static const GLfloat cSlice[12];

    float       m_color_window[2];  /**< Window settings for dicom images */
    int         m_voltex_maxvalue;  /**< max intensity value in volume tex */
    int         m_pick;             /**< flag for indicating selection mode */
    int         m_segoutline;       /**< flag - show/hide segmentation outlines */
    float       m_current_slice;    /**< current slice of the volume being displaced as an image */
    int         m_interpolation_mode; /**< flag for indicating interpolation mode to be used for slice view */
    float       m_scaling_factor;   /**< accumulated scaling based on mouse action */
    GLfloat     m_displace[2];      /**< accumulated displacement / panning based on mouse action */
    bool        m_repainting_flag;
    bool        m_scribbling;
    bool        m_ScribbleChanged;
    double      m_spacing[2];

    const sjDS::Grid* m_grid;             /**< Grid object for describing the input volume */
    sjDS::BitImage m_selectionMask;

    QPoint      m_mouse_prevpos;    /**< record previous mouse position for mouse move events. */
    GLfloat     m_selected_point[3];   /**< Draw a selected point using a mouse. */

    GLfloat     m_quad_vertices[12],   /**< vertex points for slice plane  */
                m_quad_colors[12];     /**< vertex points for slice colors */

    GLuint      m_program_slice;
    QString     m_path_to_slice_program[2] = {"../common/Graphics/slice.vert.glsl","../common/Graphics/slice.frag.glsl"};

    GLuint      m_vbo_verts,           /**< vertex coordinates for slice-plane. */
                m_vbo_colors;          /**< VBO for colors buffer */

    GLint       m_attrib_verts,        /**< shader binding for vertex coordinates.     */
                m_uniform_pick,        /**< shader binding for selection mode flag.    */
                m_attrib_colors,       /**< shader binding for texture coordinates.    */
                m_uniform_window,      /**< shader binding for dicom window settings.  */
                m_uniform_maxvalue,    /**< shader binding for max image scalar value. */
                m_uniform_slice_number,/**< current slice number sent to glsl program. */
                m_uniform_segoutline,  /**< shader binding for seg-outline enable flag.*/
                m_uniform_interpolation;

    Texh        m_volume;             /**< 3D texture handler. Set from outside.      */
    Texh        m_segment;            /**< Segmentation texture handler. Set from outside. */
    Texh        m_gradient;           /**< Segmentation texture handler. Set from outside. */
    Texh        m_optical;            /**< Optical properties texture handler. Set from outside. */
    Texh        m_selection;          /**< Selection mask texture handler. Constructed inside. */

public slots:
    void slot_updateSlice(int slice_number);
    void slot_EnableScribbling(bool);
    void slot_ClearScribbling(bool);

signals:
    void sign_sliceChanged(int slice_number);

};
