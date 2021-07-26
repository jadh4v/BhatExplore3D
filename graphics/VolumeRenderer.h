/**
 * @brief The VolumeRenderer class
 *          This class implements opengl volume rendering using GLSL
 *          and fragment and vertex shaders. It uses the cube back
 *          and front face rendering to compute rays.
 */
#ifndef VOLUMERENDERER_H
#define VOLUMERENDERER_H
#include <vector>
#include <QObject>
#include <QWidget>
#include <QGLWidget>
#include "core/macros.h"
#include "vpStructures.h"
#include "vpTypes.h"
#include "vpCanvas.h"

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class QByteArray;
class vpVisualState;
namespace sjDS{
class Grid;
}

class VolumeRenderer : public vpCanvas
{
Q_OBJECT

public:
    enum enumRenderMode{RENMODE_DEFAULT=0, RENMODE_SURFACE};

    struct Camera
    {
    public:
        double      up[3];
        double      focus[3];
        double      position[3];
        glm::quat   rot_quat;

        Camera()
        {
                  up[0] = 0;       up[1] = 1;       up[2] = 0;
               focus[0] = 0;    focus[1] = 0;    focus[2] = 0;
            position[0] = 0; position[1] = 0; position[2] = 1;
        }

    };


    VolumeRenderer(QWidget *parent = 0);
    virtual ~VolumeRenderer();

    void RecordWindowSize( int width, int height );
    void SetVolumeTextureScaleOffset( int scale, int offset );
    void SetRayCastingStepSize(float input_value);
    void SetDisplayBoundingBox ( size_t dim[6] );
    void SetVoxelSpacing( const double spacing[3] );
    void SetVolumeDimensions( size_t dim[3] );
    void SetGrid(const sjDS::Grid* grid);
    void SetModeToIgnoreScalars();
    void SetModeToConsiderScalars();
    const sjDS::Grid* GetGrid() const { return m_grid; }

    void GetModelBox(float box[3]);
    int GetNormalizedDimensions(float norm_dim[]) const;
    int GetNormalizedDisplayBox(float norm_box[]) const;
    void GetDisplayBoundingBox(size_t corners[6]) const;

    MacroSetMember(uint, m_volume.opengl_tex, VolumeTexture)
    MacroSetMember(uint, m_optical.opengl_tex, OpticalTexture)
    MacroSetMember(uint, m_volume_slices.opengl_tex, VolumeTextureArray)
    MacroSetMember(uint, m_seg_slices.opengl_tex, SegmentTextureArray)
    MacroSetMember(uint, m_seg.opengl_tex, SegmentTexture)
    MacroSetMember(bool, m_repainting_flag, EnableRepainting)
    void GetDisplayDimensions(uint dim[3]) const;
    MacroGetMember( const float*, m_camera_matrix, CameraMatrix)
    int GrabFrame(void** data);
    void SetNumberOfSegments(size_t segNumber);
    void SetRenderMode(uint label, enumRenderMode mode );
    void SetSegmentRange(uint label, uint range[2]);

    // Change default vertex and fragment shader for the color cube rendering program.
    void SetCubeProgramPath( QString vertex_shader, QString fragment_shader );
    // Change default vertex and fragment shader for the ray-casting program.
    void SetRayProgramPath( QString vertex_shader, QString fragment_shader );

    // Set Camera Parameters externally.
    void SetCamera(const VolumeRenderer::Camera& cam);
    MacroGetMember( const VolumeRenderer::Camera&, m_camera, Camera )

//##################################################################################################
protected:
    virtual void initializeGL();
    virtual void resizeGL( int w, int h );
    virtual void mouseReleaseEvent(QMouseEvent*);
    virtual void mouseMoveEvent(QMouseEvent*);

//##################################################################################################
private: // functions

    /// Main opengl draw routine. Draw 3D stuff by overriding this method.
    virtual void draw_gl();

    /// Override to draw text etc over the 3D GL drawing drawn by draw_gl().
    virtual void over_draw();

    void initialize();

    // Read an input file to a char array
    static QByteArray file_read( const char *filename );

    void init_frame_buffers();

    void render_buffer_to_screen();

    void draw_fullscreen_quad();

    void reshape_ortho(int w, int h);

    void render_backface();

    void raycasting_pass();

    GLuint create_tftexture( size_t width, vpn::tfType *tfunc );

    int compute_box_vertices();

    void compute_jitter_texture();

    size_t get_max_dimension() const;

    void initialize_camera_position();

    void calculate_camera_matrix();

    const sjDS::Grid* m_grid = nullptr;

    /// Texture handles.
    TextureHandle m_volume, m_volume_slices, m_seg_slices, m_seg, m_jitter, m_optical;

    int m_window_width, m_window_height;
    bool m_frame_buffs_initialized;
    bool m_repainting_flag;

    QPoint      m_mouse_prevpos;    /**< record previous mouse position for mouse move events. */
    Camera      m_camera;



    // shaders and program
    GLuint      m_program_cube,     /**< cube program handle */
                m_program_ray;      /**< raycasting program handle */
    QString m_path_cube_program[2] = {"../../Bhattacharyya/Graphics/cube.vert.glsl",
                                      "../../Bhattacharyya/Graphics/cube.frag.glsl"};
    QString m_path_ray_program[2]  = {"../../Bhattacharyya/Graphics/volume_rendering.vert.glsl",
                                      "../../Bhattacharyya/Graphics/volume_rendering.frag.glsl"};

    // attributes
    GLint       m_attrib_ray_vertexPosition; /**< raycasting vertices */
    // camera matrix
    GLint       m_uniform_camera_matrix; /**< camera matrix handle */
    GLint       m_uniform_ray_camera_matrix, /**< camera matrix handle */
                m_uniform_ray_model_matrix,
                m_uniform_ray_view_matrix;
    GLfloat     m_camera_matrix[16];    /**< camera transformation matrix */
    // Frame buffer related
    GLuint      m_framebuffer,      /**< draw hidden in this buffer */
                m_renderbuffer,     /**< draw final view in this buffer */
                m_backface_buffer,  /**< back face texture */
                m_final_image,      /**< final volume image texture */
                m_vao_cube,         /**< Vertex array object for cube */
                m_vao_raycasting;   /**< Vertex array object for volume */
    // BackBuffer, Segmentation, and other uniforms.
    GLint       m_uniform_ray_back_tex,
                m_uniform_ray_seg_tf_tex,
                m_uniform_ray_stepsize,
                m_uniform_ray_dim_scale,
                m_uniform_ray_maxvalue,
                m_uniform_visTexBounds,
                m_uniform_renderModes,
                m_uniform_segmentRanges,
                m_uniform_visFlags,
                m_uniform_ray_volDim;

    float       m_ray_stepsize;     /**< stepping size for raycasting ray */
    int         m_voltex_scaleoffset[2];  /**< max intensity value in volume tex */
    //int         m_pancreasOnly;     /**< Pancreas only view mode */

    size_t      m_display_box[6]; /**< display dimensions*/
    unsigned int  m_volDim[3];      /**< volume data dimensions*/
    float       m_dim_scale[3];     /**< scaled dimensions with max = 1.0 */
    double      m_voxel_spacing[3];
    bool        m_modelCentroidCalc;
    float       m_modelCentroid[3];
    float       m_visTexBounds[6];
    bool        m_grabFrame;
    std::vector<GLuint> m_SegmentRenderMode;
    std::vector<GLuint> m_SegmentRanges;
    std::vector<GLuint> m_VisFlags;

    VBOHandle<GLfloat>     m_vbo_box;            /**< colors vbo handle for the box */
    VBOHandle<GLfloat>     m_vbo_colors;         /**< colors vbo handle for the box */
    AttribHandle  m_attrib_backverts;   /**< vertex coordinates handle */
    AttribHandle  m_attrib_backcolors;  /**< vertex coordinates handle */
    AttribHandle  m_attrib_frontverts;  /**< vertex coordinates handle */
    AttribHandle  m_attrib_frontcolors; /**< vertex coordinates handle */

};
#endif // VPVOLUMERENDERER_H
