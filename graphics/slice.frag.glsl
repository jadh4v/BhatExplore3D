#version 400 core

#define MASK_PANCREAS 1
#define MASK_PANCREAS_OUTLINE 2
#define MASK_CYST 3
#define MASK_DUCT 4

in vec3 f_color;
layout(location = 0) out vec4 finalPixelColor;

uniform int segOutlineEnabled;
uniform int orient;
uniform float distance;
uniform float window[2];
uniform float plane_normal[3];
uniform float panBox[6];

uniform int pick;
uniform int sliceNumber;
uniform int maxValue;

uniform sampler2DArray volumeTexture;
uniform usampler3D selectionTexture;
uniform usampler2DArray segmentTextureArray;
uniform sampler1DArray opticalTexture;

//#################################################################################################
int labelChecki(const uint back, const uint front)
{
    if( back == front )
        return 0;
    else //if((back > 0.5f && front < 0.5f)||(front > 0.5f && back < 0.5f))
        return 1;
}

//#################################################################################################
int labelCheckf(const float back, const float front)
{
    if( back == front )
        return 0;
    else //if((back > 0.5f && front < 0.5f)||(front > 0.5f && back < 0.5f))
        return 1;
}

//#################################################################################################
int Xcheck( const float delta, const vec3 samplePos, usampler2DArray tex )
{
    uint labxb = texture( tex, vec3(samplePos.x-delta, samplePos.y, samplePos.z) ).r;
    uint labxf = texture( tex, vec3(samplePos.x+delta, samplePos.y, samplePos.z) ).r;
    return labelChecki( labxb, labxf );
}

//#################################################################################################
int Ycheck( const float delta, const vec3 samplePos, usampler2DArray tex )
{
    uint labyb = texture( tex, vec3(samplePos.x, samplePos.y-delta, samplePos.z) ).r;
    uint labyf = texture( tex, vec3(samplePos.x, samplePos.y+delta, samplePos.z) ).r;
    return labelChecki( labyb, labyf );
}


//#################################################################################################
int Zcheck( const float delta, const vec3 samplePos, sampler3D tex )
{
    float labzb = texture( tex, vec3(samplePos.x, samplePos.y, samplePos.z-delta) ).r;
    float labzf = texture( tex, vec3(samplePos.x, samplePos.y, samplePos.z+delta) ).r;
    return labelCheckf( labzb, labzf );
}

//#################################################################################################
int Ncheck( const vec3 delta, const vec3 samplePos, sampler3D tex )
{
    float labxb = texture( tex, vec3(samplePos-delta) ).r;
    float labxf = texture( tex, vec3(samplePos+delta) ).r;
    return labelCheckf( labxb, labxf );
}

/*
int labelCheck(const uint back, const uint front)
{
    if( back == front )
        return 0;
    else if((back == MASK_PANCREAS && front == 0)||(front == MASK_PANCREAS && back == 0))
        return 1;
    else if((back == MASK_CYST && front != MASK_CYST)||(back != MASK_CYST && front == MASK_CYST))
        return 2;
    else if((back == MASK_DUCT && front != MASK_DUCT)||(back != MASK_DUCT && front == MASK_DUCT))
        return 3;
}
int Xcheck( const float delta, const vec3 samplePos )
{
    uint labxb = texture( segTexture, vec3(samplePos.x-delta, samplePos.y, samplePos.z) ).r;
    uint labxf = texture( segTexture, vec3(samplePos.x+delta, samplePos.y, samplePos.z) ).r;
    return labelCheck( labxb, labxf );
}

int Ycheck( const float delta, const vec3 samplePos )
{
    uint labyb = texture( segTexture, vec3(samplePos.x, samplePos.y-delta, samplePos.z) ).r;
    uint labyf = texture( segTexture, vec3(samplePos.x, samplePos.y+delta, samplePos.z) ).r;
    return labelCheck( labyb, labyf );
}

int Zcheck( const float delta, const vec3 samplePos )
{
    uint labzb = texture( segTexture, vec3(samplePos.x, samplePos.y, samplePos.z-delta) ).r;
    uint labzf = texture( segTexture, vec3(samplePos.x, samplePos.y, samplePos.z+delta) ).r;
    return labelCheck( labzb, labzf );
}

int Ncheck( const vec3 delta, const vec3 samplePos )
{
    uint labxb = texture( segTexture, vec3(samplePos-delta) ).r;
    uint labxf = texture( segTexture, vec3(samplePos+delta) ).r;
    return labelCheck( labxb, labxf );
}
*/

//#################################################################################################
int computeLabelBorder(float delta, const vec3 pos, sampler3D tex)
{
    vec3 tmp = normalize(vec3(0.3f,0.2,0.4f));
    vec3 normal = normalize(vec3(plane_normal[0], plane_normal[1], plane_normal[2]));

    float d = dot(tmp, normal);
    vec3 d_vec = d * normal;
    tmp = tmp - d_vec;
    tmp = normalize(tmp);
    vec3 n1 = delta * tmp;
    vec3 n2 = delta * cross(tmp, normal);

    int bord1 = Ncheck( n1, pos, tex);
    int bord2 = Ncheck( n2, pos, tex);

    if(bord1 != 0)
        return bord1;
    else if(bord2 != 0)
        return bord2;
    else
        return 0;
}

//#################################################################################################
int OutlineTest( const float delta, const vec3 pos, usampler2DArray tex )
{
    int xval = Xcheck( delta, pos, tex);
    int yval = Ycheck( delta, pos, tex);

    int border = 0;
    border = xval==0? yval : xval;
    return border;
}

//#################################################################################################
bool InsidePanBox(const vec3 pos)
{
    if(    (pos.x > panBox[0] && pos.x < panBox[3])
        && (pos.y > panBox[1] && pos.y < panBox[4])
        && (pos.z > panBox[2] && pos.z < panBox[5]) )
        return true;
    else
        return false;
}

//#################################################################################################
vec3 WorldToBoxCoord(const vec3 worldPos)
{
    vec3 boxPos = worldPos - vec3(panBox[0],panBox[1],panBox[2]);
    boxPos.x = boxPos.x / (panBox[3]-panBox[0]);
    boxPos.y = boxPos.y / (panBox[4]-panBox[1]);
    boxPos.z = boxPos.z / (panBox[5]-panBox[2]);

    return boxPos;
}


//#################################################################################################
uint ComputeSegmentLabel( in usampler2DArray seg_array, in vec3 pos )
{
    // convert position from normalized values to volume dimensions.
    ivec3 texSize = textureSize(seg_array, 0);
    vec3 dim = vec3( texSize - ivec3(1,1,1) );
    vec3 p = pos * dim;
    //p = clamp( p, vec3(0,0,0), dim-2 );
    p = clamp( p, vec3(0,0,0), dim );

    vec3 floor_p = floor(p);
    vec3 frac_p = fract(p);

    /*
    for(int i=0; i < 3; ++i)
    {
        if( frac_p[i] > 0.5 )
            frac_p[i] = 1.0;
        else
            frac_p[i] = 0.0;
    }*/

    if( frac_p.x > 0.5 ) frac_p.x = 1.0;
    else                 frac_p.x = 0.0;

    if( frac_p.y > 0.5 ) frac_p.y = 1.0;
    else                 frac_p.y = 0.0;

    if( frac_p.z > 0.5 ) frac_p.z = 1.0;
    else                 frac_p.z = 0.0;

    p = floor_p + frac_p;

    uint label = 0;
    label = texelFetch( seg_array, ivec3(p), 0 ).r;

    return label;
}

//#################################################################################################
float PWC(sampler2DArray volume_texture, vec3 pos)
{
    // convert position from normalized values to volume dimensions.
    ivec3 tmp = textureSize(volume_texture, 0);
    vec3 dim = vec3( tmp - ivec3(1,1,1) );
    ivec3 p = ivec3( round(pos.x * dim.x), round(pos.y * dim.y), pos.z );
    return texelFetch( volume_texture, p, 0 ).r;
}


//#################################################################################################
const vec3 colors[7] = vec3[7]( vec3( 0.5, 0.5, 0.5 ),
                                vec3( 1.0, 0.5, 0.5 ),
                                vec3( 0.5, 1.0, 0.5 ),
                                vec3( 0.5, 0.5, 1.0 ),
                                vec3( 0.5, 1.0, 1.0 ),
                                vec3( 1.0, 0.5, 1.0 ),
                                vec3( 1.0, 1.0, 0.5 ) );

//#################################################################################################
void main(void)
{

    //orient = 2;
    if(pick == 1)
    {
        finalPixelColor = vec4( f_color, 1.0f );
        return;
    }

    //vec3 samplePos = vec3( f_color.x, f_color.y, f_color.z );
    vec3 samplePos = vec3( f_color.x, f_color.y, sliceNumber );
    float data_value = texture(volumeTexture, samplePos).r;
    float ct_intensity = data_value * float(maxValue);
    //float ct_intensity = PWC( volumeTexture, samplePos ) * float(maxValue);

    // Calculate position in selection mask texture to sample.
    // Maximum value of voxel coordinates:
    ivec3 volTexSize = textureSize(volumeTexture,0) - 1;
    // Multiply with normalized coordinates to get selected voxel coordinates in integer format.
    ivec3 selVolPos = ivec3( round(samplePos.x * volTexSize.x), round(samplePos.y * volTexSize.y), samplePos.z );

    // Calculate the sampling position to use in the selection mask texture.
    ivec3 selTexPos = ivec3( selVolPos.x / 8, selVolPos.y, selVolPos.z );
    uint selChar = texelFetch( selectionTexture, selTexPos, 0 ).r;
    bool selState = (selChar & ( 1 << (selVolPos.x % 8) )) > 0 ;

    float wcenter = window[0]; // color window center
    float wwidth  = window[1]; // color window width
    float wwidth_2 = wwidth/2.0f;

    // clamp CT intensity value between the min and max window values
    float clamped_ct = clamp( ct_intensity, wcenter - wwidth_2, wcenter + wwidth_2 );

    // shift the origin to zero and normalize
    float normalized_clamped_ct = ( clamped_ct - wcenter + wwidth_2 ) / wwidth;
    //float delta = 0.006f*distance;
    float delta = 0.001f;
    vec4 pixel = vec4( vec3(normalized_clamped_ct), 1.0f );

    // Draw segmentation outlines:
    if( segOutlineEnabled == 1 )
    {
        //uint label = 0;
        //label = ComputeSegmentLabel( segmentTexture, samplePos );
//        label = texture( segmentTexture, samplePos ).r;
//        pixel = colors[ label % 7 ];

        uint label = texture( segmentTextureArray, samplePos ).r;
        //pixel = vec4( colors[label%7], 1.0f );

        vec4 optical_value = vec4(0,0,0,0);
        //ivec2 opt_size = textureSize(opticalTexture, 0);
        if( label > 0 )//&& label < (opt_size.x * opt_size.y) )
        {
            //pixel = mix( pixel, vec4(1,0,0,1), 0.5 ); //testing
            //ivec2 labelPos = ivec2( label % opt_size.x , label / opt_size.x );
            //optical_value = texelFetch( opticalTexture, labelPos, 0 );
            //optical_value = texture( opticalTexture, vec2( normalized_clamped_ct, label) );
            //optical_value = texture( opticalTexture, vec2( 0.5, label) );
            optical_value = texture( opticalTexture, vec2( data_value, label) );

            if( optical_value.a > 0 )
            {
                //vec4 seg_color = vec4( float(optical_value.r)/255.0f, float(optical_value.g)/255.0f,
                                  //float(optical_value.b)/255.0f, float(optical_value.a)/255.0f );

                vec4 seg_color = clamp( optical_value, 0.0, 1.0 );
                //pixel = seg_color;
                //pixel = vec4(1.0, 0.0, 0.0, 1.0);
                pixel = mix( pixel, seg_color, 0.5 );
                //pixel = mix( pixel, seg_color, seg_color.a );
            }
        }
    }

    // Mix selection color with CT color to highlighted selected voxels:
    vec4 selColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    if( selState == true )
        pixel = mix( pixel, selColor, 0.5 );

    finalPixelColor = pixel;
}
