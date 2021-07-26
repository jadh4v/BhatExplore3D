#version 400 core

#define MAXTEXDIM 8192
#define RENMODE_FLAT 0
#define RENMODE_SURFACE 1
#define RENMODE_1DTF 2
#define RGB_TEXTURE 0

in vec4 f_pos;
in vec4 f_color;
in vec3 light_dir;
in vec3 eye_dir;

layout(location = 0) out vec4 finalPixelColor;

uniform usampler3D segmentTexture;
uniform sampler3D volumeTexture;
uniform usampler2DArray segmentTextureArray;
uniform sampler1DArray opticalTexture;
uniform sampler2D back_texture;
uniform sampler2D jitter_texture;

uniform int maxValue;
uniform float stepSize;
uniform float visTexBounds[6];
uniform mat4 view_transform;
uniform mat4 model_transform;
uniform uint renderModes[256];
uniform uint segmentRanges[512];
uniform uint visFlags[32];

//#################################################################################################
void Initialize( const float step_size, out vec3 pos, out vec3 delta_dir, out int totalSteps )
{
    // the start position of the ray
    vec4 start = f_color;

    // find the right place to lookup in the backside buffer
    vec2 texc = ((f_pos.xy / (f_pos.w+0.000001)) + 1) / 2;
    vec4 end  = texture( back_texture, texc );

    // Calculate ray direction
    vec3 rayDirection = end.xyz - start.xyz;
    delta_dir = normalize(rayDirection) * step_size;
    totalSteps = int( length(rayDirection.xyz) / (length(delta_dir)+0.0001));
    totalSteps = int(clamp(totalSteps, 1, 1024));
    //totalSteps = length(rayDirection) < 0.00001f ? 0 : totalSteps;

    // Calculate Jitter value and update start position.
    float jitter_value  = texture(jitter_texture, texc).r;
//    float jitter_value  = 0.0f;
//    pos = vec3(start.xyz + 4.0f * delta_dir * jitter_value);
    pos = vec3(start.xyz + 1.0f * delta_dir * jitter_value);

}

//#################################################################################################
vec3 ComputeGradient( const float grad_step, const vec3 scaled_pos, const float ct_val,
                      const sampler3D myTex )
{
    float ct_fx, ct_bx, ct_fy, ct_by, ct_fz, ct_bz;
    ct_fx = texture( myTex, scaled_pos + vec3(grad_step,0,0) ).r;
    ct_fy = texture( myTex, scaled_pos + vec3(0,grad_step,0) ).r;
    ct_fz = texture( myTex, scaled_pos + vec3(0,0,grad_step) ).r;

    ct_bx = texture( myTex, scaled_pos + vec3(-grad_step,0,0) ).r;
    ct_by = texture( myTex, scaled_pos + vec3(0,-grad_step,0) ).r;
    ct_bz = texture( myTex, scaled_pos + vec3(0,0,-grad_step) ).r;

    vec3 gradient = vec3(ct_fx - ct_bx, ct_fy - ct_by, ct_fz - ct_bz) / (2.0f*grad_step);
    //vec3 gradient = vec3(ct_fx - ct_val, ct_fy - ct_val, ct_fz - ct_val) / (2.0f*grad_step);
    gradient = normalize(gradient);
    return gradient;
}

//#################################################################################################
vec4 ComputeLighting( const vec4 ct_color, const vec3 currLightDir, const vec3 currEyeDir,
                      vec3 gradient, const vec4 ambientScaling, const float diffusePower,
                      const float specularPower, float lightGrad  )
{
    // Light computations
    vec3 L = -1.0f * normalize(currLightDir);
    vec3 E = -1.0f * normalize(currEyeDir);

    // Flip the gradient if necessary to maintain consistent shading (maximize reflection).
    float proj = dot(L,gradient);
    if( proj < 0 )
        gradient *= -1.0f;

    //vec3 R = reflect(-L, gradient);
    vec3 R = reflect(L, gradient);
    float cosAlpha = clamp( dot(E,R), 0.0f, 1.0f );
    float cosTheta = clamp( dot( gradient, L), 0.0f, 1.0f );

    //float cosTheta = clamp( lightGrad, 0.0f, 1.0f );


    //float light_dist = length(currLightDir);
    //float light_sqr_dist = light_dist * light_dist;

    vec4 pixel_color = ct_color * ambientScaling
            + diffusePower * ct_color * cosTheta / 10.0f //(2.5f + light_sqr_dist)
            + specularPower * ct_color * pow(cosAlpha,3) / 10.0f; // (2.0f + light_sqr_dist);

    // reset the opacity from material color
    pixel_color.a = ct_color.a;
    return pixel_color;
}

//#################################################################################################
uint ComputeSegmentLabel( in usampler2DArray seg_array, in vec3 pos )
{
    // convert position from normalized values to volume dimensions.
    //vec3 p = pos * dim - 0.5 + 1.0/512.0;
    ivec3 tmp = textureSize(seg_array, 0);
    vec3 dim = vec3( tmp - ivec3(1,1,1) );
    vec3 p = pos * dim;
    //p = clamp( p, vec3(0,0,0), dim-2 );
    p = clamp( p, vec3(0,0,0), dim );

    ivec3 p_cl = ivec3(ceil(p));
    p_cl = clamp( p_cl, ivec3(0,0,0), ivec3(dim) );
    ivec3 p_fl = p_cl - ivec3(1,1,1);

    // use floor value to get lower z-slice
    //vec3 P = vec3( pos.x, pos.y, p_fl.z );
    // collect the labels of the 4 neighboring pixels
    //uvec4 lf = textureGather( seg_array, P, 0 );

    uint labels[8];
    ivec3 idx = p_fl;
    labels[0] = texelFetch( seg_array, ivec3(idx.x,idx.y+1,idx.z), 0 ).r;
    labels[1] = texelFetch( seg_array, ivec3(idx.x+1,idx.y+1,idx.z), 0 ).r;
    labels[2] = texelFetch( seg_array, ivec3(idx.x+1,idx.y,idx.z), 0 ).r;
    labels[3] = texelFetch( seg_array, idx, 0 ).r;

    //idx = ivec3(p_cl.x-1, p_cl.y-1, p_cl.z);
    idx.z = idx.z + 1;
    labels[4] = texelFetch( seg_array, ivec3(idx.x,idx.y+1,idx.z), 0 ).r;
    labels[5] = texelFetch( seg_array, ivec3(idx.x+1,idx.y+1,idx.z), 0 ).r;
    labels[6] = texelFetch( seg_array, ivec3(idx.x+1,idx.y,idx.z), 0 ).r;
    labels[7] = texelFetch( seg_array, idx, 0 ).r;

    // use ceiling value to get higher z-slice
//    P.z = p_cl.z;
    // collect the labels of the 4 neighboring pixels
//    uvec4 lc  = textureGather( seg_array, P, 0 );

    // create a single array for all 8 labels.
//    labels[0] = lf.x; labels[1] = lf.y; labels[2] = lf.z; labels[3] = lf.w;
//    labels[4] = lc.x; labels[5] = lc.y; labels[6] = lc.z; labels[7] = lc.w;

    //labels[0] = 1; labels[1] = 1; labels[2] = 0; labels[3] = 0;
    //labels[4] = 1; labels[5] = 0; labels[6] = 0; labels[7] = 0;

    //for each label, use linear interpolation with a cube to test for
    // label membership for current position 'pos'.
    for( int i=0; i < 8; ++i )
    {
        // record current label to consider.
        uint curr_label = labels[i];

        // construct a cube with 8 vertices, such that we assign a +1 value to
        // a vertex if it has 'curr_label' else we assign -1.
        // Membership of 'pos' to 'curr_label' is true if the interpolation
        // returns a positive value.
        float cube[8];
        for(int k=0; k < 8; ++k)
        {
            //cube[k] = labels[k] == curr_label? 1.0f : -1.0f;
            if( labels[k] == curr_label )
                cube[k] = 1.0f;
            else
                cube[k] = 0.0f;
        }

        // Interpolate within the cube, to check membership of curr_position,
        // w.r.t 'curr_label'.
        vec3 cube_p = fract(p + 0.0f);
        float m_f1 = mix( cube[0], cube[1], cube_p.x );
        float m_f2 = mix( cube[3], cube[2], cube_p.x );
        float m_floor = mix( m_f2, m_f1, cube_p.y );

        float m_c1 = mix( cube[4], cube[5], cube_p.x );
        float m_c2 = mix( cube[7], cube[6], cube_p.x );
        float m_ceil = mix( m_c2, m_c1, cube_p.y );

        float m = mix( m_floor, m_ceil, cube_p.z );

        // Membership of 'pos' to 'curr_label' is true if the interpolation
        // returns a positive value.
        if( m >= 0.5f )
            return curr_label;
    }

    // if test fails, return a default label (this should never happen):
//    return texture( segmentTexture, pos ).r;
    return labels[0];
//    return 1;
}

//#################################################################################################
vec3 CubeToGlobalTexture(const vec3 cubePos)
{
    vec3 texPos = cubePos;
    texPos.x = texPos.x * (visTexBounds[3]-visTexBounds[0]);
    texPos.y = texPos.y * (visTexBounds[4]-visTexBounds[1]);
    texPos.z = texPos.z * (visTexBounds[5]-visTexBounds[2]);
    texPos += vec3(visTexBounds[0],visTexBounds[1],visTexBounds[2]);

    return texPos;
}

//#################################################################################################
void main(void)
{
    //finalPixelColor = f_color;
    //return;
    /*
    finalPixelColor =  vec4(1,0,0,1);
    finalPixelColor = f_color;
    if(f_color.r > 0.999 && f_color.g > 0.999 && f_color.b > 0.999)
        return;
        */

    vec3 pos, delta_dir;
    int totalSteps = 1;

    Initialize(stepSize, pos, delta_dir, totalSteps);

    // Accumulating Variables
    vec4 col_acc = vec4(0,0,0,0);
    float alpha_acc = 0;

    // Incrementing Variables
    vec3 currLightDir = light_dir;
    vec3 currEyeDir   = eye_dir;

    // Constants
    float grad_step = stepSize*0.5f;
    const float diffusePower = 10.0f, specularPower = 4.0f;
    const vec4  ambientScaling = vec4( 0.25f, 0.25f, 0.25f, 1.0f );
    uint prev_label = 0, prev_prev_label = 0;
    ivec2 opt_size = textureSize(opticalTexture, 0);

    //totalSteps = 1024;
    for(int i = 0; i < totalSteps; i++)
    {
        // Get CT-data pixel and associated color from the transfer function texture.
        vec3 scaled_pos = CubeToGlobalTexture(pos);
        float segScale      = 1.0f;
#if RGB_TEXTURE
        vec4 intensity = vec4(0.0f);
#else
        float intensity  = 0.0f;
#endif
        float lightGradient = 0.0f;
        vec3 tmp_pos  = vec3(0.0f,0.0f,0.0f);
        vec3 gradient = vec3(0.0f,0.0f,0.0f);

#if RGB_TEXTURE
         intensity = texture(volumeTexture, scaled_pos);
#else
        // Read the intensity value from the scalar data / volume:
        if( visFlags[0] == 0 )
        {
            intensity = texture(volumeTexture, scaled_pos).r;
            clamp( intensity, 0.0f, maxValue ); // for safe indexing
        }
        else
            intensity = 0.5f;
#endif

        uint label = 0;
        //label = texture( segmentTexture, scaled_pos ).r;
        label = ComputeSegmentLabel( segmentTextureArray, scaled_pos );
        clamp(label, 0U , 255U); // for safe indexing
        if (label != 0)
        {
            uvec4 optical_value = uvec4(0, 0, 0, 0);
            vec4 ct_color = vec4(0.0f, 0.0f, 0.0f, 0.0f);

#if RGB_TEXTURE
            float scalar = max(max(intensity.r, intensity.g), intensity.b);
            ct_color = texture(opticalTexture, vec2(scalar, label));
            ct_color.r = intensity.r;
            ct_color.g = intensity.g;
            ct_color.b = intensity.b;
#else
            uint range[2];
            range[0] = segmentRanges[label];
            range[1] = segmentRanges[256 + label];
            float scalar = clamp(intensity, float(range[0]), float(range[1]));
            if(scalar > 2)
                scalar = (scalar - float(range[0])) / float(1.0f + range[1] - range[0]);
            //scalar = 0.5f;
            ct_color = texture(opticalTexture, vec2(scalar, label));
#endif
            ct_color = clamp(ct_color, 0.0, 1.0);

            // Compute gradient based shading and lighting only if the current
            // fragment is sufficiently opaque:
            if (ct_color.a > 0.001f)
            {
                vec4 pixel_color;
#if RGB_TEXTURE
                gradient = ComputeGradient(grad_step, scaled_pos, scalar, volumeTexture);
#else
                gradient = ComputeGradient(grad_step, scaled_pos, intensity, volumeTexture);
#endif

                vec4 tmp_grad = vec4(gradient, 1.0f);
                vec4 tmp_orig = vec4(0, 0, 0, 1);
                tmp_grad = (view_transform * model_transform * tmp_grad);
                tmp_orig = (view_transform * model_transform * tmp_orig);
                gradient = tmp_grad.xyz - tmp_orig.xyz;

                vec3 ngradient = 1.0f * normalize(gradient);
                pixel_color = ComputeLighting(ct_color, currLightDir, currEyeDir, ngradient,
                    ambientScaling, diffusePower, specularPower,
                    lightGradient);
                //pixel_color = ct_color;

                col_acc += ((1.0f - alpha_acc) * pixel_color.a) * pixel_color;
                alpha_acc += (1.0f - alpha_acc) * pixel_color.a;
            }
        }

        pos += delta_dir;
        currLightDir += delta_dir;
        currEyeDir += delta_dir;
        prev_prev_label = prev_label;
        prev_label = label;

        // terminate if opacity > 1 or the ray is outside the volume
        if(i >= totalSteps || alpha_acc > 0.99f)
            break;
    }

    float backgroundAlpha = 1.0 - alpha_acc;
    col_acc += ((1.0f - alpha_acc) * 1.0f) * vec4(1,1,1,1);

    finalPixelColor =  col_acc;
}

