// Matrices calculated from Equal Energy white
__CONSTANT__ float3x3 MATRIX_16 = 
{
    { 0.3640744836f,  0.5947008157f,  0.0411012735f },
    {-0.2222450987f,  1.0738554823f,  0.1479453361f },
    {-0.0020676190f,  0.0488260454f,  0.9503875570f }
};

__CONSTANT__ float3x3 MATRIX_INVERSE_16 = 
{
    { 2.0512756811f, -1.1400313440f,  0.0887556628f },
    { 0.4269389763f,  0.7005835278f, -0.1275225041f },
    {-0.0174712780f, -0.0384725929f,  1.0589468739f }
};

// Input matrix
__CONSTANT__ float3x3 AP0_ACES_to_XYZ_matrix =
{
    { 0.9525523959f,  0.0000000000f,  0.0000936786f},
    { 0.3439664498f,  0.7281660966f, -0.0721325464f},
    { 0.0000000000f,  0.0000000000f,  1.0088251844f}
};

__CONSTANT__ float3x3  XYZ_to_AP0_ACES_matrix =
{
    { 1.0498110175f,  0.0000000000f, -0.0000974845f},
    {-0.4959030231f,  1.3733130458f,  0.0982400361f},
    { 0.0000000000f,  0.0000000000f,  0.9912520182f}
};

__CONSTANT__ float3x3 XYZ_to_AP1_ACES_matrix = 
{
    { 1.6410233797f, -0.3248032942f, -0.2364246952f },
    {-0.6636628587f,  1.6153315917f,  0.0167563477f },
    { 0.0117218943f, -0.0082844420f,  0.9883948585f }
};

__CONSTANT__ float3x3 AP1_ACES_to_XYZ_matrix = 
{
    { 0.6624541811f,  0.1340042065f,  0.1561876870f },
    { 0.2722287168f,  0.6740817658f,  0.0536895174f },
    {-0.0055746495f,  0.0040607335f,  1.0103391003f }
};

// Matrix for Hellwig inverse
__CONSTANT__ float3x3 panlrcm =
{
    { 460.0f,  451.0f,  288.0f},
    { 460.0f, -891.0f, -261.0f},
    { 460.0f, -220.0f, -6300.0f},
};

__CONSTANT__ float PI = 3.141592653589793f;

__CONSTANT__ float L_A = 100.0f;

__CONSTANT__ float Y_b = 20.0f;

__CONSTANT__ float ra = 2.0f;

__CONSTANT__ float ba = 0.05f;

__CONSTANT__ float referenceLuminance = 100.0f;

__CONSTANT__ float3 surround = {0.9f, 0.59f, 0.9f};

__CONSTANT__ float3 d65White = { 95.0455927052f,  100.0f,  108.9057750760f };

__CONSTANT__ float3 inWhite = { 95.2646074570f,  100.0f,  100.8825184352f };

// __CONSTANT__ float gamut_gamma = 1.137f; // surround.y * (1.48 + sqrt(Y_b / Y_w)))
__CONSTANT__ float model_gamma = 0.8794641436f; // reciprocal of above
__CONSTANT__ float lowerHullGamma = 1.14f;

// Gamut Compression parameters
__CONSTANT__ float cuspMidBlend = 1.3f;
__CONSTANT__ float smoothCusps = 0.12f;
__CONSTANT__ float smoothJ = 0.0f;
__CONSTANT__ float smoothM = 0.27f;
__CONSTANT__ float focusAdjustGain = 0.55f;
__CONSTANT__ float focusGainBlend = 0.3f;
__CONSTANT__ float4 compressionFuncParams = {0.75f, 1.1f, 1.3f, 1.0f};

// DanieleEvoCurve (ACES2 candidate) parameters
__CONSTANT__ float daniele_n_r = 100.0f;    // Normalized white in nits (what 1.0 should be)
__CONSTANT__ float daniele_g = 1.15f;      // surround / contrast
__CONSTANT__ float daniele_t_1 = 0.04f;     // shadow toe, flare/glare compensation - how ever you want to call it

// Soft clip parameters
__CONSTANT__ float clamp_thr = 0.99f;
__CONSTANT__ float clamp_dist = 1.1f;

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

  // clamp the components of a 3D vector between a min & max value
__DEVICE__ inline float3 clamp3(float3 v, float min, float max)
  {
    v.x = _clampf(v.x, min, max);
    v.y = _clampf(v.y, min, max);
    v.z = _clampf(v.z, min, max);
    return v;
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

  // basic 3D hypotenuse function, does not deal with under/overflow
__DEVICE__ inline float hypot_float3(float3 xyz)
  {
    return _sqrtf(xyz.x * xyz.x + xyz.y * xyz.y + xyz.z * xyz.z);
  }

__DEVICE__ inline float3 compress_bjorn(float3 xyz)
  {
    float C = (xyz.x + xyz.y + xyz.z) / 3.0f;

    float3 xyz_temp = xyz - C;
    float R = hypot_float3(xyz_temp);

    if (R == 0.0f || C == 0.0f)
      return xyz;

    R = R * 0.816496580927726f;     // sqrt(2/3)

    xyz_temp = xyz_temp / R;

    float r = R / C ;
    r = r * r ;
    r = _sqrtf(4.0f / r + 1.0f) - 1.0f;

    float s = -_fminf(xyz_temp.x, _fminf(xyz_temp.y, xyz_temp.z));
    s = s - 0.5f;

    float t = 0.5f + _sqrtf(s*s + r*r / 4.0f);
    t = C / t;                                 // t is always >= 0.5f

    xyz_temp = xyz_temp * t + C;

    return xyz_temp;
  }

__DEVICE__ inline float3 uncompress_bjorn(float3 xyz)
{
    float C = (xyz.x + xyz.y + xyz.z) / 3.0f;

    float3 xyz_temp = xyz - C;
    float R = hypot_float3(xyz_temp);

    if (R == 0.0f || C == 0.0f)
      return xyz;

    R = R * 0.816496580927726f;     // sqrt(2/3)

    xyz_temp = xyz_temp / R;

    float t = C / R;
    t = t - 0.5f;

    float s = -_fminf(xyz_temp.x, _fminf(xyz_temp.y, xyz_temp.z));
    s = s -0.5f;

    float r = 2 * _sqrtf(_fabs(t*t - s*s)) + 1.0f;
    r = _sqrtf(_fabs(r*r - 1.0f));
    if (r == 0.0f)
      return xyz;
    r = C * 2.0f / r;

    xyz_temp = xyz_temp * r + C;

    return xyz_temp;
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
        vCompressed = (v<threshold||limit<=1.0001f||v>threshold+s)?v:threshold+s*_powf(-(_powf((v-threshold)/s,power)/(_powf((v-threshold)/s,power)-1.0f)),1.0f/power);
    }
    else
    {
        vCompressed = (v<threshold||limit<=1.0001f)?v:threshold+s*((v-threshold)/s)/(_powf(1.0f+_powf((v-threshold)/s,power),1.0f/power));
    }

    return vCompressed;
}

__DEVICE__ inline float3 compress_aces(float3 rgb, float3 c, float3 m, float3 y, int invert)
  {
    float ach = _fmaxf(rgb.x, _fmaxf(rgb.y, rgb.z));
    float3 d = make_float3(0.0f, 0.0f, 0.0f);

    if (ach)
    {
      d.x = (ach - rgb.x) / _fabs(ach);
      d.y = (ach - rgb.y) / _fabs(ach);
      d.z = (ach - rgb.z) / _fabs(ach);
    }

    rgb.x = compressPowerP(d.x, c.x, c.y, c.z, invert);
    rgb.y = compressPowerP(d.y, m.x, m.y, m.z, invert);
    rgb.z = compressPowerP(d.z, y.x, y.y, y.z, invert);

    rgb = ach - rgb * _fabs(ach);

    return rgb;
  }

__DEVICE__ inline float3 post_adaptation_non_linear_response_compression_forward(float3 RGB, float F_L)
{
    float3 F_L_RGB = float3spow(F_L * float3abs(RGB) / 100.0f, 0.42f);
    float3 RGB_c;
    RGB_c.x = (400.0f * _copysignf(1.0f, RGB.x) * F_L_RGB.x) / (27.13f + F_L_RGB.x);
    RGB_c.y = (400.0f * _copysignf(1.0f, RGB.y) * F_L_RGB.y) / (27.13f + F_L_RGB.y);
    RGB_c.z = (400.0f * _copysignf(1.0f, RGB.z) * F_L_RGB.z) / (27.13f + F_L_RGB.z);

    return RGB_c;
}

__DEVICE__ inline float3 post_adaptation_non_linear_response_compression_inverse(float3 RGB,float F_L)
{
    float3 RGB_p =  (float3sign(RGB) * 100.0f / F_L * float3spow((27.13f * float3abs(RGB)) / (400.0f - float3abs(RGB)), 1.0f / 0.42f) );

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
    float3 RGB_aw = (400.0f * float3sign(RGB_wc) * F_L_RGB) / (27.13f + F_L_RGB);

    // # Computing achromatic responses for the whitepoint.
    float R_aw = RGB_aw.x ;
    float G_aw = RGB_aw.y ;
    float B_aw = RGB_aw.z ;
    float A_w = A_w = ra * R_aw + G_aw + ba * B_aw;

    // # Step 1
    // # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.

    float3 RGB = vector_dot(MATRIX_16, XYZ);

    // # Step 2
    float3 RGB_c = D_RGB * RGB;

    // # Step 3
    // No compressMode
//     RGB_c = compress_bjorn(RGB_c);

    float3 RGB_a = post_adaptation_non_linear_response_compression_forward(RGB_c, F_L);

//     RGB_a = uncompress_bjorn(RGB_a);

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
    float A = 2.0f * R_a2 + G_a2 + 0.05f * B_a2;

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
    float3 RGB_aw = (400.0f * float3sign(RGB_wc) * F_L_RGB) / (27.13f + F_L_RGB);

    // # Computing achromatic responses for the whitepoint.
    float R_aw = RGB_aw.x ;
    float G_aw = RGB_aw.y ;
    float B_aw = RGB_aw.z ;
    float A_w = A_w = ra * R_aw + G_aw + ba * B_aw;

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

    // No compressMode
//     RGB_a = compress_bjorn(RGB_a);

    float3 RGB_c = post_adaptation_non_linear_response_compression_inverse(RGB_a, F_L);

//     RGB_c = uncompress_bjorn(RGB_c);

    // # Step 6
    float3 RGB = RGB_c / D_RGB;
    

    // # Step 7
    float3 XYZ = vector_dot(MATRIX_INVERSE_16, RGB);

    return XYZ;
}

__DEVICE__ inline float Y_to_J( float Y )
  {
      // # Viewing conditions dependent parameters (could be pre-calculated)
      float k = 1.0f / (5.0f * L_A + 1.0f);
      float k4 = k*k*k*k;
      float F_L = 0.2f * k4 * (5.0f * L_A) + 0.1f * _powf((1.0f - k4), 2.0f) * spow(5.0f * L_A, 1.0f / 3.0f) ;
      float n = Y_b / 100.0f;
      float z = 1.48f + _sqrtf(n);
      float F_L_W = _powf(F_L, 0.42f);
      float A_w = (400.0f * F_L_W) / (27.13f + F_L_W);

      float F_L_Y = _powf(F_L * _fabs(Y) / 100.0f, 0.42f);

      return _copysignf(100.0f * _powf(((400.0f * F_L_Y) / (27.13f + F_L_Y)) / A_w, surround.y * z), Y);
  }

__DEVICE__ inline float J_to_Y( float J )
  {
      // # Viewing conditions dependent parameters (could be pre-calculated)
      float k = 1.0f / (5.0f * L_A + 1.0f);
      float k4 = k*k*k*k;
      float F_L = 0.2f * k4 * (5.0f * L_A) + 0.1f * _powf((1.0f - k4), 2.0f) * spow(5.0f * L_A, 1.0f / 3.0f) ;
      float n = Y_b / 100.0f;
      float z = 1.48f + _sqrtf(n);
      float F_L_W = _powf(F_L, 0.42f);
      float A_w = (400.0f * F_L_W) / (27.13f + F_L_W);

      float A = _copysignf(A_w * _powf(_fabs(J) / 100.0f, 1.0f / (surround.y * z)), J);

      return _copysignf(100.0f / F_L * _powf((27.13f * _fabs(A)) / (400.0f - _fabs(A)), 1.0f / 0.42f), A);
  }

__DEVICE__ inline float daniele_evo_fwd(float Y)
{
    float f = daniele_m_2 * _powf(_fmaxf(0.0f, Y) / (Y + daniele_s_2), daniele_g);
    float h = _fmaxf(0.0f, f * f / (f + daniele_t_1));

    return h;
}

__DEVICE__ inline float daniele_evo_rev(float Y)
{
    Y = _fmaxf(0.0f, _fminf(daniele_n / (daniele_u_2 * daniele_n_r), Y));
    float h = (Y + _sqrtf(Y * (4.0f * daniele_t_1 + Y))) / 2.0f;
    float f = daniele_s_2 / (_powf((daniele_m_2 / h), (1.0f / daniele_g)) - 1.0f);

    return f;
}

__DEVICE__ inline int midpoint(int low, int high)
  {
    int mid = (high + low) / 2; // Integer division
    return mid;
  }

__DEVICE__ inline float wrap_to_360(float hue)
  {
    float y = _fmod(hue, 360.0f);
    if (y < 0.0f)
    {
      y = y + 360.0f;
    }
    return y;
  }

__DEVICE__ inline int hue_position_in_uniform_table(float hue, int table_size)
  {
    const float wrapped_hue = wrap_to_360(hue);
    int hue_pos = wrapped_hue / 360.0f * table_size;
    return hue_pos;
  }

__DEVICE__ inline float2 cuspFromTable(float h)
{
    float3 lo;
    float3 hi;

    if( h <= gamutCuspTable[0].z )
    {
      lo = gamutCuspTable[359];
      lo.z = lo.z-360.0f;
      hi = gamutCuspTable[0];
    }
    else if( h >= gamutCuspTable[359].z )
    {
      lo = gamutCuspTable[359];
      hi = gamutCuspTable[0];
      hi.z = hi.z+360.0f;
    }
    else
    {
      int low_i  = 0;
      int high_i = 359;
      int i      = hue_position_in_uniform_table(h, 360);

      while (low_i + 1 < high_i)
      {
        if (h > gamutCuspTable[i].z)
        {
          low_i = i;
        }
        else
        {
          high_i = i;
        }
        i = midpoint(low_i, high_i);
      }
      lo = gamutCuspTable[high_i - 1];
      hi = gamutCuspTable[high_i];
    }

    float t = (h - lo.z) / (hi.z - lo.z);

    float cuspJ = lerp(lo.x, hi.x, t);
    float cuspM = lerp(lo.y, hi.y, t);

    return make_float2(cuspJ,cuspM);
}

__DEVICE__ inline float2 cCuspFromTable(float h)
{
    float3 lo;
    float3 hi;

    if( h <= cGamutCuspTable[0].z )
    {
      lo = cGamutCuspTable[359];
      lo.z = lo.z-360.0f;
      hi = cGamutCuspTable[0];
    }
    else if( h >= cGamutCuspTable[359].z )
    {
      lo = cGamutCuspTable[359];
      hi = cGamutCuspTable[0];
      hi.z = hi.z+360.0f;
    }
    else
    {
      int low_i  = 0;
      int high_i = 359;
      int i      = hue_position_in_uniform_table(h, 360);

      while (low_i + 1 < high_i)
      {
        if (h > cGamutCuspTable[i].z)
        {
          low_i = i;
        }
        else
        {
          high_i = i;
        }
        i = midpoint(low_i, high_i);
      }
      lo = cGamutCuspTable[high_i - 1];
      hi = cGamutCuspTable[high_i];
    }

    float t = (h - lo.z) / (hi.z - lo.z);

    float cuspJ = lerp(lo.x, hi.x, t);
    float cuspM = lerp(lo.y, hi.y, t);

    return make_float2(cuspJ,cuspM);
}

__DEVICE__ inline float reachFromTable(float h)
{
    int lo = (int)_floorf(mod(h, 360.0f));
    int hi = (int)_ceilf(mod(h, 360.0f));
    if (hi == 360)
    {
        hi = 0;
    }
    float t = _fmod(h, 1.0f);

    return lerp(gamutCuspTableReach[lo], gamutCuspTableReach[hi], t);
}

  // A "toe" function that remaps the given value x between 0 and limit.
  // The k1 and k2 parameters change the size and shape of the toe.
  // https://www.desmos.com/calculator/6vplvw14ti
__DEVICE__ inline float toe(float x, float limit, float k1, float k2, int inverse)
  {
    if (x > limit)
      return x;

    k2 = max(k2, 0.001f);
    k1 = _sqrtf(k1 * k1 + k2 * k2);
    float k3 = (limit + k1) / (limit + k2);

    if (!inverse)
      return 0.5f * (k3 * x - k1 + _sqrtf((k3 * x - k1) * (k3 * x - k1) + 4.0f * k2 * k3 * x));
    else
      return (x * x + k1 * x) / (k3 * (x + k2));
  }

  // Chroma compression
  //
  // Compresses colors inside the gamut with the aim for colorfulness to have an
  // appropriate rate of change from display black to display white, and from
  // achromatic outward to purer colors.
  //
__DEVICE__ inline float chromaCompression(float3 JMh, float origJ, float linear, int invert)
  {
    float M = JMh.y;
    if (M == 0.0f)
      return M;

    float nJ = JMh.x / limitJmax;
    float snJ = max(0.0f, 1.0f - nJ);
    float Mnorm = cCuspFromTable(JMh.z).y;
    float limit = _powf(nJ, model_gamma) * reachFromTable(JMh.z) / Mnorm;

    if (!invert)
    {
      //
      // Rescaling of M with the tonescaled J to get the M to the same range as
      // J after the tonescale.  The rescaling uses the Hellwig2022 model gamma to
      // keep the M/J ratio correct (keeping the chromaticities constant).
      //
      M *= _powf(JMh.x / origJ, model_gamma);

      // Normalize M with the rendering space cusp M
      M /= Mnorm;

      //
      // Expand the colorfulness by running the toe function in reverse.  The goal is to
      // expand less saturated colors less and more saturated colors more.  The expansion
      // increases saturation in the shadows and mid-tones but not in the highlights.
      // The 0.001 offset starts the expansions slightly above zero.  The sat_thr makes
      // the toe less aggressive near black to reduce the expansion of noise.
      //
      M = limit - toe(limit - M, limit - 0.001f, snJ * sat, _sqrtf(nJ * nJ + sat_thr), 0);

      //
      // Compress the colorfulness.  The goal is to compress less saturated colors more and
      // more saturated colors less, especially in the highlights.  This step creates the
      // saturation roll-off in the highlights, but attemps to preserve pure colors.  This
      // mostly affects highlights and mid-tones, and does not compress shadows.
      //
      M = toe(M, limit, nJ * compr, snJ, 0);

      // Denormalize
      M *= Mnorm;
    }
    else
    {
      M /= Mnorm;
      M = toe(M, limit, nJ * compr, snJ, 1);
      M = limit - toe(limit - M, limit - 0.001f, snJ * sat, _sqrtf(nJ * nJ + sat_thr), 1);
      M *= Mnorm;

      M *= _powf(JMh.x / origJ, -model_gamma);
    }

    return M;
  }

__DEVICE__ inline float3 forwardTonescale( float3 inputJMh, int compressChroma)
{
    float3 outputJMh;
    float linear;
    linear = J_to_Y(inputJMh.x) / referenceLuminance;

    // only Daniele Evo tone scale
    float luminanceTS = daniele_evo_fwd(linear);

    float tonemappedJ = Y_to_J(luminanceTS * referenceLuminance);

    outputJMh = make_float3(tonemappedJ, inputJMh.y, inputJMh.z);


    // Chroma Compression)
    if (compressChroma)
    {
        outputJMh.y = chromaCompression(outputJMh, inputJMh.x, linear, 0);
    }

    return outputJMh;
}

__DEVICE__ inline float3 inverseTonescale( float3 JMh, int compressChroma)
  {
    float luminance;
    float3 untonemappedColourJMh = JMh;
    float3 tonemappedJMh = JMh;
    
    luminance = J_to_Y(JMh.x);

    float linear = daniele_evo_rev(luminance / referenceLuminance);

    float untonemappedJ = Y_to_J(linear * referenceLuminance);
    untonemappedColourJMh = make_float3(untonemappedJ,tonemappedJMh.y,tonemappedJMh.z);

    if (compressChroma)
    {
      untonemappedColourJMh.y = chromaCompression(tonemappedJMh, untonemappedColourJMh.x, linear, 1);
    }

    return  untonemappedColourJMh;
  }

// Smooth minimum of a and b
__DEVICE__ inline float smin(float a, float b, float s)
{
    float h = _fmaxf(s - _fabs(a - b), 0.0f) / s;
    return _fminf(a, b) - h * h * h * s * (1.0f / 6.0f);
}

__DEVICE__ inline float hueDependantUpperHullGamma(float h)
  {
    int index = (int)_floorf(h);
    float t = (h - index);
    float gamma = 1.0f;
    if (index < 359)
    {
        gamma = lerp(gamutTopGamma[index], gamutTopGamma[index+1], t);
    }
    else
    {
        gamma = lerp(gamutTopGamma[359], gamutTopGamma[0], t);
    }

    return gamma;
  }


// reimplemented from https://github.com/nick-shaw/aces-ot-vwg-experiments/blob/master/python/intersection_approx.py
__DEVICE__ inline float solve_J_intersect(float2 JM, float focusJ, float maxJ, float slope_gain)
  {
    float a = JM.y / (focusJ * slope_gain);
    float b = 0.0f;
    float c = 0.0f;
    float intersectJ = 0.0f;
    
    if (JM.x < focusJ)
    {
        b = 1.0f - JM.y / slope_gain;
    } 
    else
    {
        b= -(1.0f + JM.y / slope_gain + maxJ * JM.y / (focusJ * slope_gain));
    } 

    if (JM.x < focusJ)
    {
        c = -JM.x;
    } 
    else
    {
        c = maxJ * JM.y / slope_gain + JM.x;
    }

    float root = _sqrtf(b*b - 4.0f * a * c);

    if (JM.x < focusJ)
    {
        intersectJ = 2.0f * c / (-b - root);
    } 
    else
    {
        intersectJ = 2.0f * c / (-b + root);
    } 

    return intersectJ;
}

// reimplemented from https://github.com/nick-shaw/aces-ot-vwg-experiments/blob/master/python/intersection_approx.py
__DEVICE__ inline float3 findGamutBoundaryIntersection(float3 JMh_s, float2 JM_cusp, float J_focus, float J_max, float slope_gain, float smoothness)
  {
    float2 JM_source = make_float2(JMh_s.x, JMh_s.y);
    float gamma_top = hueDependantUpperHullGamma(JMh_s.z);
    float gamma_bottom = lowerHullGamma + 0.07 * _log10f(daniele_n / daniele_n_r);

    float slope = 0.0f;

    float s = _fmaxf(0.000001f, smoothness);
    JM_cusp.x *= 1.0f + smoothJ * s;   // J
    JM_cusp.y *= 1.0f + smoothM * s;   // M

    float J_intersect_source = solve_J_intersect(JM_source, J_focus, J_max, slope_gain);
    float J_intersect_cusp = solve_J_intersect(JM_cusp, J_focus, J_max, slope_gain);

    if (J_intersect_source < J_focus)
    {
        slope = J_intersect_source * (J_intersect_source - J_focus) / (J_focus * slope_gain);
    }
    else
    {
        slope = (J_max - J_intersect_source) * (J_intersect_source - J_focus) / (J_focus * slope_gain);

    } 

    float M_boundary_lower = J_intersect_cusp * _powf(J_intersect_source / J_intersect_cusp, 1 / gamma_bottom) / (JM_cusp.x / JM_cusp.y - slope);

    float M_boundary_upper = JM_cusp.y * (J_max - J_intersect_cusp) * _powf((J_max - J_intersect_source) / (J_max - J_intersect_cusp), 1.0f / gamma_top) / (slope * JM_cusp.y + J_max - JM_cusp.x);

    float M_boundary = JM_cusp.y * smin(M_boundary_lower / JM_cusp.y, M_boundary_upper / JM_cusp.y, s);

    float J_boundary = J_intersect_source + slope * M_boundary;

    return make_float3(J_boundary, M_boundary,J_intersect_source);  
}

  // https://www.desmos.com/calculator/oe2fscya80
__DEVICE__ inline float getFocusGain(float J, float cuspJ)
  {
    float thr = lerp(cuspJ, limitJmax, focusGainBlend);
    if (J > thr)
    {
      // Approximate inverse required above threshold
      float gain = (limitJmax - thr) / _fmaxf(0.0001f, (limitJmax - _fminf(limitJmax, J)));
      return _powf(_log10f(gain), 1.0f / focusAdjustGain) + 1.0f;
    }
    else
    {
      // Analytic inverse possible below cusp
      return 1.0f;
    }
  }

__DEVICE__ inline float3 compressGamut2(float3 JMh, int invert, float Jx)
  {
    float2 project_from = make_float2(JMh.x, JMh.y);
    float2 JMcusp = cuspFromTable(JMh.z);

    if (JMh.y < 0.0001f || JMh.x > limitJmax)
    {
      return make_float3(JMh.x, 0.0f, JMh.z);
    }

    // Calculate where the out of gamut color is projected to
    float focusJ = lerp(JMcusp.x, midJ, _fminf(1.0f, cuspMidBlend - (JMcusp.x / limitJmax)));

    float slope_gain = limitJmax * focusDist * getFocusGain(Jx, JMcusp.x);

    // Find gamut intersection
    float3 nickBoundryReturn = findGamutBoundaryIntersection(JMh, JMcusp, focusJ, limitJmax, slope_gain, smoothCusps);
    float2 JMboundary        = make_float2(nickBoundryReturn.x, nickBoundryReturn.y);
    float2 project_to        = make_float2(nickBoundryReturn.z, 0.0f);
    float projectJ           = nickBoundryReturn.z;

    // Calculate AP1 Reach boundary
    float reachMaxM = reachFromTable(JMh.z);

    // slope_gain changed to use boundary J instead of source J (unsure if this is needed)
    slope_gain = limitJmax * focusDist * getFocusGain(JMboundary.x, JMcusp.x);
    
    // J intersection recalculated with new slope_gain (unsure if this is needed)
    projectJ = solve_J_intersect(JMboundary, focusJ, limitJmax, slope_gain);

    // slope is recalculated here because it was a local variable in findGamutBoundaryIntersection
    // but also because it will have changed slightly due to the changed slope_gain
    float slope;

    if (projectJ < focusJ)
    {
        slope = projectJ * (projectJ - focusJ) / (focusJ * slope_gain);
    }
    else
    {
        slope = (limitJmax - projectJ) * (projectJ - focusJ) / (focusJ * slope_gain);
    }
    
    float boundaryNick = limitJmax * _powf(projectJ/limitJmax, model_gamma) * reachMaxM / (limitJmax - slope * reachMaxM);
    
    float difference = _fmaxf(1.0001f, boundaryNick / JMboundary.y);
    float threshold = _fmaxf(compressionFuncParams.x, 1.0f / difference);

    // Compress the out of gamut color along the projection line
    float v = project_from.y / JMboundary.y;
    v = compressPowerP(v, threshold, difference, compressionFuncParams.w, invert);
    float2 JMcompressed = project_to + v * (JMboundary - project_to);

    return make_float3(JMcompressed.x, JMcompressed.y, JMh.z);
  }

__DEVICE__ inline float3 compressGamut(float3 JMh, int invert)
  {
    if (!invert)
    {
      return compressGamut2(JMh, 0, JMh.x);
    }
    else
    {
      float2 JMcusp = cuspFromTable(JMh.z);
      float  Jx     = JMh.x;

      // Analytic inverse below threshold
      if (Jx <= lerp(JMcusp.x, limitJmax, focusGainBlend))
        return compressGamut2(JMh, 1, Jx);

      // Approximation above
      Jx = compressGamut2(JMh, 1, Jx).x;
      return compressGamut2(JMh, 1, Jx);
    }
  }

  // encode linear values as ST2084 PQ
__DEVICE__ inline float linear_to_ST2084( float v )
{
    float Y_p = spow(_fmaxf(0.0f, v) / st2084_L_p, st2084_m_1);

    return spow((st2084_c_1 + st2084_c_2 * Y_p) / (st2084_c_3 * Y_p + 1.0f), st2084_m_2);
}

  // convert ST2084 PQ encoded values to linear
__DEVICE__ inline float ST2084_to_linear(float v)
  {
    float V_p = spow(v, st2084_m_2_d);

    return spow((max(0.0f, V_p - st2084_c_1) / (st2084_c_2 - st2084_c_3 * V_p)), st2084_m_1_d) * st2084_L_p;
  }
