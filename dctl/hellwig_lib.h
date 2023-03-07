// Matrices calculated from {4200, -1050} white point used in Blink v30
// __CONSTANT__ float3x3 MATRIX_16 = {
//     {-0.32119474f, -0.23319618f, -0.01719972f},
//     {-0.0910343f ,  0.44249129f,  0.06447764f},
//     { 0.02945856f, -0.10641155f,  0.40821152f}
// };

// __CONSTANT__ float3x3 MATRIX_INVERSE_16 = {
//     {-2.70657868f, -1.40060996f,  0.10718864f},
//     {-0.56387056f,  1.88543648f, -0.32156592f},
//     { 0.04833176f,  0.59256575f,  2.35815011f}
// };

// Matrices calculated from Equal Energy white
__CONSTANT__ float3x3 MATRIX_16 = {
    { 0.56193142f,  0.40797761f,  0.03009097f},
    {-0.21886684f,  1.06384814f,  0.15501869f},
    { 0.08892922f, -0.32123412f,  1.2323049f }
};

__CONSTANT__ float3x3 MATRIX_INVERSE_16 = {
    { 1.54705503f, -0.58256219f,  0.03550715f},
    { 0.32230313f,  0.78421833f, -0.10652146f},
    {-0.02762598f,  0.24646862f,  0.78115737f}
};

// Input matrix
__CONSTANT__ float3x3 AP0_ACES_to_XYZ_matrix = {
    { 0.9525523959f,  0.0000000000f,  0.0000936786f},
    { 0.3439664498f,  0.7281660966f, -0.0721325464f},
    { 0.0000000000f,  0.0000000000f,  1.0088251844f}
};

__CONSTANT__ float3x3  XYZ_to_AP0_ACES_matrix = {
    { 1.0498110175f,  0.0000000000f, -0.0000974845f},
    {-0.4959030231f,  1.3733130458f,  0.0982400361f},
    { 0.0000000000f,  0.0000000000f,  0.9912520182f}
};

// Matrix for Hellwig inverse
__CONSTANT__ float3x3 panlrcm = {
    { 460.0f,  451.0f,  288.0f},
    { 460.0f, -891.0f, -261.0f},
    { 460.0f, -220.0f, -6300.0f},
};

__CONSTANT__ float float_epsilon = 0.0000000596046448f;
__CONSTANT__ float HALF_MAXIMUM = 65504.0f;

__CONSTANT__ float PI = 3.141592653589793f;

__CONSTANT__ float L_A = 100.0f;

__CONSTANT__ float Y_b = 20.0f;

__CONSTANT__ float referenceLuminance = 100.0f;

__CONSTANT__ float3 surround = {0.9f, 0.59f, 0.9f};

__CONSTANT__ float3 d65White = {95.0455927052f, 100.0f, 108.9057750760f};

// Chroma compress parameters
__CONSTANT__ float hoff = 0.835f;
__CONSTANT__ float hmul = 14.0f;
__CONSTANT__ float2 a = {-0.18f, -0.42f};
__CONSTANT__ float2 b = {0.135f, 0.13f};
__CONSTANT__ float2 c = {-0.08f, 0.0f};
__CONSTANT__ float chromaCompress = 1.0f;
__CONSTANT__ float2 chromaCompressParams = {2.1f, 0.85f};

// __CONSTANT__ float gamut_gamma = 1.137f; // surround.y * (1.48 + sqrt(Y_b / Y_w)))
__CONSTANT__ float gamut_gamma = 0.879464f; // reciprocal of above

// Gamut Compression parameters
__CONSTANT__ float cuspMidBlend = 0.8f;
__CONSTANT__ float smoothCusps = 0.0f;
__CONSTANT__ float midJ = 34.08f; // ~10 nits in Hellwig J
__CONSTANT__ float focusDistance = 2.0f;
__CONSTANT__ float4 compressionFuncParams = {0.75f, 1.2f, 1.45f, 1.0f};

// DanieleEvoCurve (ACES2 candidate) parameters
__CONSTANT__ float mmScaleFactor = 100.0f; 
__CONSTANT__ float daniele_n_r = 100.0f;    // Normalized white in nits (what 1.0 should be)
__CONSTANT__ float daniele_g = 1.15f;      // surround / contrast
__CONSTANT__ float daniele_c = 0.18f;      // scene-referred grey
__CONSTANT__ float daniele_c_d = 10.013f;    // display-referred grey (in nits)
__CONSTANT__ float daniele_w_g = 0.14f;    // grey change between different peak luminance
__CONSTANT__ float daniele_t_1 = 0.04f;     // shadow toe, flare/glare compensation - how ever you want to call it
__CONSTANT__ float daniele_r_hit_min = 128.0f;  // Scene-referred value "hitting the roof" at 100 nits
__CONSTANT__ float daniele_r_hit_max = 896.0f;  // Scene-referred value "hitting the roof" at 10,000 nits

// ST2084 constants
__CONSTANT__ float st2084_m_1=2610.0f / 4096.0f * (1.0f / 4.0f);
__CONSTANT__ float st2084_m_2=2523.0f / 4096.0f * 128.0f;
__CONSTANT__ float st2084_c_1=3424.0f / 4096.0f;
__CONSTANT__ float st2084_c_2=2413.0f / 4096.0f * 32.0f;
__CONSTANT__ float st2084_c_3=2392.0f / 4096.0f * 32.0f;
__CONSTANT__ float st2084_m_1_d = 8192.0f / 1305.0f; // 1.0f / st2084_m_1;
__CONSTANT__ float st2084_m_2_d = 32.0f / 2523.0f; // 1.0f / st2084_m_2;
__CONSTANT__ float st2084_L_p = 10000.0f;

// multiplies a 3D vector with a 3x3 matrix
__DEVICE__ inline float3 vector_dot( float3x3 m, float3 v)
{
    float3 r;

    r.x = m.x.x * v.x + m.x.y * v.y + m.x.z * v.z;
    r.y = m.y.x * v.x + m.y.y * v.y + m.y.z * v.z;
    r.z = m.z.x * v.x + m.z.y * v.y + m.z.z * v.z;
    
    return r;
}

// "safe" power function to avoid NANs or INFs when taking a fractional power of a negative base
// this one initially returned -pow(abs(b), e) for negative b
// but this ended up producing undesirable results in some cases
// so now it just returns 0.0 instead
__DEVICE__ inline float spow( float base, float exponent )
{
    if(base < 0.0f && exponent != _floorf(exponent) )
    {
         return 0.0f;
    }
    else
    {
        return _powf(base, exponent); 
    }
}

__DEVICE__ inline float3 float3spow( float3 base, float exponent )
{
    return make_float3(spow(base.x, exponent), spow(base.y, exponent), spow(base.z, exponent));
}

__DEVICE__ inline float3 float3sign( float3 v )
{
    return make_float3(_copysignf(1.0f, v.x), _copysignf(1.0f, v.y), _copysignf(1.0f, v.z));
}

__DEVICE__ inline float3 float3abs( float3 a )
{
    return make_float3(_fabs(a.x), _fabs(a.y), _fabs(a.z));
}

// "safe" div
__DEVICE__ inline float sdiv( float a, float b )
{
    if(b == 0.0f)
    {
        return 0.0f;
    }
    else
    {
        return a / b;
    }
}

// linear interpolation between two values a & b with the bias t
__DEVICE__ inline float lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

// convert radians to degrees
__DEVICE__ inline float radians_to_degrees( float radians )
{
    return radians * 180.0f / PI;
}


// convert degrees to radians
__DEVICE__ inline float degrees_to_radians( float degrees )
{
    return degrees / 180.0f * PI;
}

__DEVICE__ inline float mod(float a, float N)
{
    return a - N * _floorf(a / N);
}

__DEVICE__ inline float3 compress(float3 xyz)
{
    float x = xyz.x;
    float y = xyz.y;
    float z = xyz.z;
   
    float C = (x + y + z) / 3.0f;
    if (C < 0.000001f)
        return xyz;

    float R = _sqrtf((x-C)*(x-C) + (y-C)*(y-C) + (z-C)*(z-C));
    R = R * 0.816496580927726f; // np.sqrt(2/3)
    
//     if (R > 0.0001f)
    if (R > float_epsilon)
    {
      x = (x - C) / R;
      y = (y - C) / R;
      z = (z - C) / R;
    }
    else
    {
      return xyz;
    }
      
    float r = R / C;
    float s = -_fminf(x, _fminf(y, z));
    
    float t = 0.0f;
    if (r > 0.000001f)
    {
      t = (0.5f + _sqrtf(((s - 0.5f)*(s - 0.5f) + _powf((_sqrtf(4.0f / (r*r) + 1.0f) - 1.0f), 2.0f) / 4.0f)));
      if (t < 0.000001f)
        return xyz;
      t = 1.0f / t;
    }

    x = C * x * t + C;
    y = C * y * t + C;
    z = C * z * t + C;

    return make_float3(x, y, z);
}

__DEVICE__ inline float3 uncompress(float3 xyz)
{
    float x = xyz.x;
    float y = xyz.y;
    float z = xyz.z;

    float C = (x+y+z)*(1.0f / 3.0f) ;
    if (C < 0.000001f)
         return xyz;

    float R = _sqrtf(_powf(_fabs(x-C), 2.0f) + _powf(_fabs(y-C), 2.0f) + _powf(_fabs(z-C), 2.0f));
    R = R * 0.816496580927726; // np.sqrt(2/3)

//     if (R > 0.0001f)
    if (R > float_epsilon)
    {
        x = (x - C) / R;
        y = (y - C) / R;
        z = (z - C) / R;
    }
    else
    {
      return xyz;
    }

    float t = R / C;
    float s = -_fminf(x, _fminf(y, z));
    
    float r = 0.0f;
    if (t  > 0.000001f)
    {
         r = _sqrtf(_powf((2.0f * _sqrtf(_powf((1.0f / t - 0.5f),2.0f) - _powf((s - 0.5f), 2.0f)) + 1.0f), 2.0f) - 1.0f);
         if (r < 0.000001f)
            return xyz;
         r = 2.0f / r;
    }

    x = C * x * r + C;
    y = C * y * r + C;
    z = C * z * r + C;
    
    return make_float3(x, y, z);
}

// "PowerP" compression function (also used in the ACES Reference Gamut Compression)
// values of v above  'threshold' are compressed by a 'power' function
// so that an input value of 'limit' results in an output of 1.0
__DEVICE__ inline float compressPowerP( float v, float threshold, float limit, float power, int inverse )
{
    float s = (limit-threshold)/_powf(_powf((1.0f-threshold)/(limit-threshold),-power)-1.0f,1.0f/power);

    float vCompressed;

    if( inverse )
    {
        vCompressed = (v<threshold||limit<1.0001f||v>threshold+s)?v:threshold+s*_powf(-(_powf((v-threshold)/s,power)/(_powf((v-threshold)/s,power)-1.0f)),1.0f/power);
    }
    else
    {
        vCompressed = (v<threshold||limit<1.0001f)?v:threshold+s*((v-threshold)/s)/(_powf(1.0f+_powf((v-threshold)/s,power),1.0f/power));
    }

    return vCompressed;
}

__DEVICE__ inline float3 post_adaptation_non_linear_response_compression_forward(float3 RGB, float F_L)
{
    float3 F_L_RGB = float3spow(F_L * float3abs(RGB) / 100.0f, 0.42f);
    float3 RGB_c;
    RGB_c.x = (400.0f * _copysignf(1.0f, RGB.x) * F_L_RGB.x) / (27.13f + F_L_RGB.x) + 0.1f;
    RGB_c.y = (400.0f * _copysignf(1.0f, RGB.y) * F_L_RGB.y) / (27.13f + F_L_RGB.y) + 0.1f;
    RGB_c.z = (400.0f * _copysignf(1.0f, RGB.z) * F_L_RGB.z) / (27.13f + F_L_RGB.z) + 0.1f;

    return RGB_c;
}

__DEVICE__ inline float3 post_adaptation_non_linear_response_compression_inverse(float3 RGB,float F_L)
{
    float3 RGB_p =  (float3sign(RGB - 0.1f) * 100.0f / F_L * float3spow((27.13f * float3abs(RGB - 0.1f)) / (400.0f - float3abs(RGB - 0.1f)), 1.0f / 0.42f) );

    return RGB_p;
}

__DEVICE__ inline float3 XYZ_to_Hellwig2022_JMh( float3 XYZ, float3 XYZ_w)
{
    float Y_w = XYZ_w.y ;

    // # Step 0
    // # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
    float3 RGB_w = vector_dot(MATRIX_16, XYZ_w);

    // Ignore degree of adaptation.
    // If we always do this, some of the subsequent code can be simplified
    float D = 1.0f;

    // # Viewing conditions dependent parameters
    float k = 1.0f / (5.0f * L_A + 1.0f);
    float k4 = _powf(k,4);
    float F_L = 0.2f * k4 * (5.0f * L_A) + 0.1f * _powf((1.0f - k4), 2.0f) * spow(5.0f * L_A, 1.0f / 3.0f) ;
    float n = sdiv(Y_b, Y_w);
    float z = 1.48f + _sqrtf(n);

    float3 D_RGB = D * Y_w / RGB_w + 1.0f - D;
    float3 RGB_wc = D_RGB * RGB_w;

    // # Applying forward post-adaptation non-linear response compression.
    float3 F_L_RGB = float3spow(F_L * float3abs(RGB_wc) / 100.0f, 0.42f);

    // # Computing achromatic responses for the whitepoint.
    float3 RGB_aw = (400.0f * float3sign(RGB_wc) * F_L_RGB) / (27.13f + F_L_RGB) + 0.1f;

    // # Computing achromatic responses for the whitepoint.
    float R_aw = RGB_aw.x ;
    float G_aw = RGB_aw.y ;
    float B_aw = RGB_aw.z ;
    float A_w = 2.0f * R_aw + G_aw + 0.05f * B_aw - 0.305f;

    // # Step 1
    // # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.

    float3 RGB = vector_dot(MATRIX_16, XYZ);

    // # Step 2
    float3 RGB_c = D_RGB * RGB;

    // # Step 3
    // Always compressMode
    RGB_c = compress(RGB_c);

    float3 RGB_a = post_adaptation_non_linear_response_compression_forward(RGB_c, F_L);

    RGB_a = uncompress(RGB_a);

    // # Step 4
    // # Converting to preliminary cartesian coordinates.
    float R_a = RGB_a.x ;
    float G_a = RGB_a.y ;
    float B_a = RGB_a.z ;
    float a = R_a - 12.0f * G_a / 11.0f + B_a / 11.0f;
    float b = (R_a + G_a - 2.0f * B_a) / 9.0f;

    // # Computing the *hue* angle :math:`h`.
    // Unclear why this isnt matching the python version.
    float h = mod(radians_to_degrees(_atan2f(b, a)), 360.0f);

    // # Step 6
    // # Computing achromatic responses for the stimulus.
    float R_a2 = RGB_a.x ;
    float G_a2 = RGB_a.y ;
    float B_a2 = RGB_a.z ;
    // A = 2 * R_a + G_a + 0.05 * B_a - 0.305
    float A = 2.0f * R_a2 + G_a2 + 0.05f * B_a2 - 0.305f;

    // # Step 7
    // # Computing the correlate of *Lightness* :math:`J`.
    // with sdiv_mode():
    float J = 100.0f * spow(sdiv(A, A_w), surround.y * z);

    // # Step 9
    // # Computing the correlate of *colourfulness* :math:`M`.
    float M = 43.0f * surround.z * _sqrtf(a * a + b * b);

    // Np *Helmholtz–Kohlrausch* Effect Extension.

    if (J == 0.0f)
    {
        M = 0.0f;
    }
      return make_float3(J, M, h);
}

__DEVICE__ inline float3 Hellwig2022_JMh_to_XYZ( float3 JMh, float3 XYZ_w)
{
    float J = JMh.x;
    float M = JMh.y;
    float h = JMh.z;

    float Y_w = XYZ_w.y;

    // # Step 0
    // # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
    float3 RGB_w = vector_dot(MATRIX_16, XYZ_w);

    // Ignore degree of adaptation.
    // If we always do this, some of the subsequent code can be simplified
    float D = 1.0f;

    // # Viewing conditions dependent parameters
    float k = 1.0f / (5.0f * L_A + 1.0f);
    float k4 = _powf(k, 4.0f);
    float F_L = 0.2f * k4 * (5.0f * L_A) + 0.1f * _powf((1.0f - k4), 2.0f) * spow(5.0f * L_A, 1.0f / 3.0f) ;
    float n = sdiv(Y_b, Y_w);
    float z = 1.48f + _sqrtf(n);

    float3 D_RGB = D * Y_w / RGB_w + 1.0f - D;
    float3 RGB_wc = D_RGB * RGB_w;

    // # Applying forward post-adaptation non-linear response compression.
    float3 F_L_RGB = float3spow(F_L * float3abs(RGB_wc) / 100.0f, 0.42f);

    // # Computing achromatic responses for the whitepoint.
    float3 RGB_aw = (400.0f * float3sign(RGB_wc) * F_L_RGB) / (27.13f + F_L_RGB) + 0.1f;

    // # Computing achromatic responses for the whitepoint.
    float R_aw = RGB_aw.x ;
    float G_aw = RGB_aw.y ;
    float B_aw = RGB_aw.z ;
    float A_w = 2.0f * R_aw + G_aw + 0.05f * B_aw - 0.305f;

    float hr = degrees_to_radians(h);

    // No *Helmholtz–Kohlrausch* Effect.

    // # Computing achromatic response :math:`A` for the stimulus.
    float A = A_w * spow(J / 100.0f, 1.0f / (surround.y * z));

    // # Computing *P_p_1* to *P_p_2*.
    float P_p_1 = 43.0f * surround.z;
    float P_p_2 = A;


    // # Step 3
    // # Computing opponent colour dimensions :math:`a` and :math:`b`.
    // with sdiv_mode():
    float gamma = M / P_p_1;

    float a = gamma * _cosf(hr);
    float b = gamma * _sinf(hr);


    // # Step 4
    // # Applying post-adaptation non-linear response compression matrix.

    float3 RGB_a = vector_dot(panlrcm, make_float3(P_p_2, a, b)) / 1403.0f;

    // # Step 5
    // # Applying inverse post-adaptation non-linear response compression.

    // Always compressMode
    RGB_a = compress(RGB_a);

    float3 RGB_c = post_adaptation_non_linear_response_compression_inverse(RGB_a + 0.1f, F_L);

    RGB_c = uncompress(RGB_c);

    // # Step 6
    float3 RGB = RGB_c / D_RGB;
    

    // # Step 7
    float3 XYZ = vector_dot(MATRIX_INVERSE_16, RGB);

    return XYZ;
}

// convert JMh correlates to  RGB values in the output colorspace
// __DEVICE__ inline float3 JMh_to_luminance_RGB(float3 JMh)
// {
//     float3 luminanceXYZ = Hellwig2022_JMh_to_XYZ( JMh, d65White);
//     float3 luminanceRGB = vector_dot(XYZ_to_RGB_output, luminanceXYZ);
// 
//     return luminanceRGB;
// }

// convert RGB values in the output colorspace to the Hellwig J (lightness), M (colorfulness) and h (hue) correlates
// __DEVICE__ inline float3 luminance_RGB_to_JMh(float3 luminanceRGB)
// {
//     float3 XYZ = vector_dot(RGB_to_XYZ_output, luminanceRGB);
//     float3 JMh = XYZ_to_Hellwig2022_JMh(XYZ, d65White);
//     return JMh;
// }

__DEVICE__ inline float daniele_evo_fwd(float Y)
{
    const float daniele_r_hit = daniele_r_hit_min + (daniele_r_hit_max - daniele_r_hit_min) * (_logf(daniele_n / daniele_n_r) / _logf(10000.0f / 100.0f));
    const float daniele_m_0 = daniele_n / daniele_n_r;
    const float daniele_m_1 = 0.5f * (daniele_m_0 + _sqrtf(daniele_m_0 * (daniele_m_0 + 4.0f * daniele_t_1)));
    const float daniele_u = _powf((daniele_r_hit / daniele_m_1) / ((daniele_r_hit / daniele_m_1) + 1.0f), daniele_g);
    const float daniele_m = daniele_m_1 / daniele_u;
    const float daniele_w_i = _logf(daniele_n / 100.0f) / _logf(2.0f);
    const float daniele_c_t = daniele_c_d * (1.0f + daniele_w_i * daniele_w_g) / daniele_n_r;
    const float daniele_g_ip = 0.5f * (daniele_c_t + _sqrtf(daniele_c_t * (daniele_c_t + 4.0f * daniele_t_1)));
    const float daniele_g_ipp2 = -daniele_m_1 * _powf(daniele_g_ip / daniele_m, 1.0f / daniele_g) / (_powf(daniele_g_ip / daniele_m, 1.0f / daniele_g) - 1.0f);
    const float daniele_w_2 = daniele_c / daniele_g_ipp2;
    const float daniele_s_2 = daniele_w_2 * daniele_m_1;
    const float daniele_u_2 = _powf((daniele_r_hit / daniele_m_1) / ((daniele_r_hit / daniele_m_1) + daniele_w_2), daniele_g);
    const float daniele_m_2 = daniele_m_1 / daniele_u_2;

    float f = daniele_m_2 * _powf(_fmaxf(0.0f, Y) / (Y + daniele_s_2), daniele_g);
    float h = _fmaxf(0.0f, f * f / (f + daniele_t_1));

    return h;
}

__DEVICE__ inline float daniele_evo_rev(float Y)
{
    const float daniele_r_hit = daniele_r_hit_min + (daniele_r_hit_max - daniele_r_hit_min) * (_logf(daniele_n / daniele_n_r) / _logf(10000.0f / 100.0f));
    const float daniele_m_0 = daniele_n / daniele_n_r;
    const float daniele_m_1 = 0.5f * (daniele_m_0 + _sqrtf(daniele_m_0 * (daniele_m_0 + 4.0f * daniele_t_1)));
    const float daniele_u = _powf((daniele_r_hit / daniele_m_1) / ((daniele_r_hit / daniele_m_1) + 1.0f), daniele_g);
    const float daniele_m = daniele_m_1 / daniele_u;
    const float daniele_w_i = _logf(daniele_n / 100.0f) / _logf(2.0f);
    const float daniele_c_t = daniele_c_d * (1.0f + daniele_w_i * daniele_w_g) / daniele_n_r;
    const float daniele_g_ip = 0.5f * (daniele_c_t + _sqrtf(daniele_c_t * (daniele_c_t + 4.0f * daniele_t_1)));
    const float daniele_g_ipp2 = -daniele_m_1 * _powf(daniele_g_ip / daniele_m, 1.0f / daniele_g) / (_powf(daniele_g_ip / daniele_m, 1.0f / daniele_g) - 1.0f);
    const float daniele_w_2 = daniele_c / daniele_g_ipp2;
    const float daniele_s_2 = daniele_w_2 * daniele_m_1;
    const float daniele_u_2 = _powf((daniele_r_hit / daniele_m_1) / ((daniele_r_hit / daniele_m_1) + daniele_w_2), daniele_g);
    const float daniele_m_2 = daniele_m_1 / daniele_u_2;

    Y = max(0.0f, _fminf(daniele_n / (daniele_u_2 * daniele_n_r), Y));
    float h = (Y + _sqrtf(Y * (4.0f * daniele_t_1 + Y))) / 2.0f;
    float f = daniele_s_2 / (_powf((daniele_m_2 / h), (1.0f / daniele_g)) - 1.0f);

    return f;
}

__DEVICE__ inline float ptanh(float x, float p, float t, float pt)
{
    return x <= 10.0f ? _powf(_tanhf(_powf(x, p) / t), 1.0f / pt) : 1.0f;
}

// convert linear RGB values with the limiting primaries to Hellwig J (lightness), M (colorfulness) and h (hue) correlates
// __DEVICE__ inline float3 limit_RGB_to_JMh(float3 RGB)
// {
//     float3 luminanceRGB = RGB * boundaryRGB * referenceLuminance;
//     float3 XYZ = vector_dot(RGB_to_XYZ_limit, luminanceRGB);
//     float3 JMh = XYZ_to_Hellwig2022_JMh(XYZ, d65White);
//     return JMh;
// }

// Scaled power(p)
__DEVICE__ inline float spowerp(float x, float l, float p)
{
    x = x / l;
    x = x != 0.0f ? x / _powf(1.0f + spow(x, p), 1.0f / p) : 0.0f;
    return x * l;
}

__DEVICE__ inline float desat_curve(float x)
  {
    float m = daniele_n / daniele_n_r;
    float w = 1.18f * m;
    return (_fmaxf(0.0f, x) / (x + w)) * m;
  }

  // Hue-dependent curve used in chroma compression
  // https://www.desmos.com/calculator/lmbbu8so4c
__DEVICE__ inline float compr_hue_depend(float h)
{
    float hr = degrees_to_radians(h);
    float hr2 = hr * 2.0f;
    float hr3 = hr * 3.0f;

    return (a.x * _cosf(hr) +
            b.x * _cosf(hr2) +
            c.x * _cosf(hr3) +
            a.y * _sinf(hr) +
            b.y * _sinf(hr2) +
            c.y * _sinf(hr3) +
            hoff) * hmul;
}

  // Chroma compression
  //
  // - Compresses the scene colorfulness with desat_curve() and spowerp() for
  //   path-to-white and path-to-black.
  // - Scales the colorfulness with a cubic curve to affect the rate of change of
  //   desaturation as lightness is increased.  This is hue dependent and affects
  //   a range of colorfulness (distance from the achromatic).
  //
__DEVICE__ inline float chromaCompression(float3 JMh, float luminance, int invert)
{
    float M = JMh.y;

    // Model specific factors to avoid having to change parameters manually
    int camMode = 1;
    float model_desat_factor = camMode == 1 ? chromaCompress * 1.22f : chromaCompress * 0.8f;
    float model_factor = camMode == 1 ? 5.0f : 1.0f;

    // Path-to-white
    //
    // Compression curve based on the difference of the scene luminance and desat_curve().
    // This scales automatically, compressing less with higher peak luminance.  Higher peak
    // luminance has a slower rate of change for colorfulness so it needs less compression.
    // The end variable can be used to affect how close to white point the curve ends, and
    // prevents the curve ever going negative.
    // https://www.desmos.com/calculator/ovy5wzr7lm
    //
    float end = 0.12f;
//     float x = _log10f(luminance) - _log10f(desat_curve(luminance));
    float x = _log10f(_fmaxf(float_epsilon, luminance)) - _log10f(_fmaxf(float_epsilon, desat_curve(luminance)));
    model_desat_factor += _logf(daniele_n / daniele_n_r) * 0.08f;
    float desatcurve = spowerp(x * model_desat_factor, chromaCompressParams.x, chromaCompressParams.y);
    desatcurve = desatcurve < (1.0f - end) ? desatcurve : (1.0f - end) + end * _tanhf((desatcurve - (1.0f - end)) / end);
//     if (isnan(desatcurve))
//     {
//         desatcurve = 0.0f;
//     }
//     if (luminance > 100.0f)
//     {
//         desatcurve = 1.0f;
//     }

    // Path-to-black
    //
    // Shadow compression to reduce clipping and colorfulness of noise.
    // https://www.desmos.com/calculator/ovy5wzr7lm
    //
    float shadowcurve = ptanh(luminance, shadowCompressParams.x, shadowCompressParams.y, shadowCompressParams.z);
//     if (isnan(shadowcurve) || luminance > 1.0f) // hack to catch tanh overflow
//     {
//         shadowcurve = 1.0f;
//     }

    // In-gamut compression
    //
    // Hue-dependent compression of M with R (J) from achromatic outward.  The purpose is to make sure
    // the interior of the gamut is smooth and even.  Larger values of R will compress larger range of
    // colorfulness.  The c variable controls compression with R (1.0 no compression, 0.0001 full
    // compression).  The driver is the tonescaled lightness in 0-1 range.  The shadow_boost affects
    // saturation mainly at and under normal exposure.
    // https://www.desmos.com/calculator/nygtri388c
    //
    float R = (JMh.x + 0.01f) * model_factor * compr_hue_depend(JMh.z);
    float c = _fmaxf(1.0f - (JMh.x / limitJmax), 0.0001f) * shadow_boost;

    desatcurve = (1.0f - desatcurve) * shadowcurve;

    if (!invert)
    {
      M *= desatcurve;
      if (M != 0.0f && R != 0.0f)
      {
        M *= ((M * M + R * c) / (M * M + R));
      }
      M *= sat;
    }
    else
    {
      M /= sat;
      if (M != 0.0f && R != 0.0f)
      {
        float t0 = 3.0f * R * c;
        float p0 = M * M - t0;
        float p1 = 2.0f * M * M + 27.0f * R - 3.0f * t0;
        float p2 = spow((_sqrtf(M * M * p1 * p1 - 4.0f * p0 * p0 * p0) / 2.0f) + M * p1 / 2.0f, 1.0f / 3.0f);
        M = (p0 / (3.0f * p2) + (p2 / 3.0f) + (M / 3.0f));
      }
      M /= desatcurve;
    }

    return M;
}

__DEVICE__ inline float3 forwardTonescale( float3 inputJMh, int compressChroma)
{
    float3 outputJMh;
//     float3 monoJMh = make_float3(_fminf(inputJMh.x, daniele_r_hit_max), 0.0f, 0.0f);
    float3 monoJMh = make_float3(inputJMh.x, 0.0f, 0.0f);
//     float3 linearJMh = JMh_to_luminance_RGB(monoJMh);
    float3 luminanceXYZ = Hellwig2022_JMh_to_XYZ( monoJMh, d65White);
//     float linear = linearJMh.x / referenceLuminance;
    float linear = luminanceXYZ.y / referenceLuminance;

    // only Daniele Evo tone scale
//     float luminanceTS = daniele_evo_fwd(linear) * mmScaleFactor;
    float luminanceTS = daniele_evo_fwd(linear);

//     float3 tonemappedmonoJMh = luminance_RGB_to_JMh(make_float3(luminanceTS,luminanceTS,luminanceTS));
    float3 tonemappedmonoJMh = XYZ_to_Hellwig2022_JMh(d65White * luminanceTS, d65White);
    float3 tonemappedJMh = make_float3(tonemappedmonoJMh.x, inputJMh.y, inputJMh.z);

    outputJMh = tonemappedJMh;

    // Chroma Compression)
    if (compressChroma)
    {
        outputJMh.y = chromaCompression(outputJMh, linear, 0);
    }

    return outputJMh;
}

__DEVICE__ inline float3 inverseTonescale( float3 JMh, int compressChroma)
  {
    float3 tonemappedJMh = JMh;

    float3 untonemappedColourJMh = tonemappedJMh;
    
    float3 monoTonemappedJMh = make_float3(tonemappedJMh.x, 0.0f, 0.0f);
//     float3 monoTonemappedRGB = JMh_to_luminance_RGB(monoTonemappedJMh);
//     float3 newMonoTonemappedJMh = luminance_RGB_to_JMh(monoTonemappedRGB);
//     float luminance = monoTonemappedRGB.x;
    float3 luminanceXYZ = Hellwig2022_JMh_to_XYZ( monoTonemappedJMh, d65White);
    float luminance = luminanceXYZ.y;

    float linear = daniele_evo_rev(luminance / mmScaleFactor);

//     linear = linear * referenceLuminance;
  
//     float3 untonemappedMonoJMh = luminance_RGB_to_JMh(make_float3(linear,linear,linear));
    float3 untonemappedMonoJMh = XYZ_to_Hellwig2022_JMh(d65White * linear, d65White);
    untonemappedColourJMh = make_float3(untonemappedMonoJMh.x,tonemappedJMh.y,tonemappedJMh.z); 

    if (compressChroma)
    {
//       untonemappedColourJMh.y = chromaCompression(tonemappedJMh, linear/referenceLuminance, 1);
      untonemappedColourJMh.y = chromaCompression(tonemappedJMh, linear, 1);
    }

    return  untonemappedColourJMh;
  }

__DEVICE__ inline float2 cuspFromTable(float h)
{
    int lo = (int)_floorf(mod(h, 360.0f));
    int hi = (int)_ceilf(mod(h, 360.0f));
    if (hi == 360)
    {
        hi = 0;
    }
    float t = _fmod(h, 1.0f);
    float2 out;
    out.x = lerp(gamutCuspTable[lo].x, gamutCuspTable[hi].x, t);
    out.y = lerp(gamutCuspTable[lo].y, gamutCuspTable[hi].y, t);

    return out;
}

// Smooth minimum of a and b
__DEVICE__ inline float smin(float a, float b, float s)
{
    float h = _fmaxf(s - _fabs(a - b), 0.0f) / s;
    return _fminf(a, b) - h * h * h * s * (1.0f / 6.0f);
}

// Approximation of the gamut intersection to a curved and smoothened triangle
// along the projection line 'from -> to'. 
__DEVICE__ inline float2 find_gamut_intersection(float2 cusp, float2 from, float2 to, float smoothing)
{
    float t0, t1;

    // Scale the cusp outward when smoothing to avoid reducing the gamut.  Reduce
    // smoothing for high cusps because smin() will bias it too much for the longer line.
    float s = _fmaxf(lerp(smoothing, smoothing * 0.01f, cusp.x / limitJmax), 0.0001f);
    cusp.y += 15.0f * s;
    cusp.x += 5.0f * s;

    // Line below the cusp is curved with gamut_gamma
    float toJ_gamma = cusp.x * spow(to.x / cusp.x, gamut_gamma);
    float fromJ_gamma = cusp.x * spow(from.x / cusp.x, gamut_gamma);
    t0 = cusp.y * toJ_gamma / (from.y * cusp.x + cusp.y * (toJ_gamma - fromJ_gamma));

    // Line above the cusp
    t1 = cusp.y * (to.x - limitJmax) / (from.y * (cusp.x - limitJmax) + cusp.y * (to.x - from.x));

    // Smooth minimum to smooth the cusp
    t1 = smin(_fabs(t0), _fabs(t1), s);

    return make_float2(to.x * (1.0f - t1) + t1 * from.x, t1 * from.y);
}

__DEVICE__ inline float3 compressGamut(float3 JMh, int invert)
{
    float2 project_from = make_float2(JMh.x, JMh.y);
    float2 JMcusp = cuspFromTable(JMh.z);

    if (project_from.y == 0.0f)
      return JMh;

    // Calculate where the out of gamut color is projected to
    float focusJ = lerp(JMcusp.x, midJ, cuspMidBlend);

    // https://www.desmos.com/calculator/9u0wiiz9ys
    float Mratio = project_from.y / (focusDistance * JMcusp.y);
    float a = _fmaxf(0.001f, Mratio / focusJ);
    float b0 = 1.0f - Mratio;
    float b1 = -(1.0f + Mratio + (a * limitJmax));
    float b = project_from.x < focusJ ? b0 : b1;
    float c0 = -project_from.x;
    float c1 = project_from.x + limitJmax * Mratio;
    float c = project_from.x < focusJ ? c0 : c1;

    float J0 = _sqrtf(b * b - 4.0f * a * c);
    float J1 = (-b - J0) / (2.0f * a);
          J0 = (-b + J0) / (2.0f * a);
    float projectJ = project_from.x < focusJ ? J0 : J1;

    // Find gamut intersection
    float2 project_to = make_float2(projectJ, 0.0f);
    float2 JMboundary = find_gamut_intersection(JMcusp, project_from, project_to, smoothCusps);

    // Compress the out of gamut color along the projection line
    float v = project_from.y / JMboundary.y;
    v = compressPowerP(v, compressionFuncParams.x, lerp(compressionFuncParams.z, compressionFuncParams.y, projectJ / limitJmax), compressionFuncParams.w, invert);
    float2 JMcompressed = project_to + v * (JMboundary - project_to);

    return make_float3(JMcompressed.x, JMcompressed.y, JMh.z);
}

  // encode linear values as ST2084 PQ
__DEVICE__ inline float linear_to_ST2084( float v )
{
    float Y_p = spow(_fmaxf(0.0f, v) / st2084_L_p, st2084_m_1);

    return spow((st2084_c_1 + st2084_c_2 * Y_p) / (st2084_c_3 * Y_p + 1.0f), st2084_m_2);
}