// Matrices calculated from Equal Energy white
__CONSTANT__ float3x3 MATRIX_16 = {
//     { 0.56193142f,  0.40797761f,  0.03009097f},
//     {-0.21886684f,  1.06384814f,  0.15501869f},
//     { 0.08892922f, -0.32123412f,  1.2323049f }
    { 0.5951576789f,  0.4394092886f, -0.0344634736f},
    {-0.2333583733f,  1.0893484122f,  0.1435787936f},
    { 0.0572735340f, -0.3038780496f,  1.2428721668f}
};

__CONSTANT__ float3x3 MATRIX_INVERSE_16 = {
//     { 1.54705503f, -0.58256219f,  0.03550715f},
//     { 0.32230313f,  0.78421833f, -0.10652146f},
//     {-0.02762598f,  0.24646862f,  0.78115737f}
    { 1.4519606301f, -0.5565110365f,  0.1045504064f},
    { 0.3098696467f,  0.7705537428f, -0.0804233895f},
    { 0.0088534185f,  0.2140427063f,  0.7801068782f}
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

__CONSTANT__ float3 cGamutCuspTable[360] = {
{66.2342f, 138.09f, 0.0874088f},
{66.0272f, 138.684f, 0.645807f},
{65.8179f, 139.244f, 1.22086f},
{65.6058f, 139.76f, 1.81442f},
{65.3902f, 140.219f, 2.42868f},
{65.1703f, 140.606f, 3.06605f},
{64.9451f, 140.905f, 3.7293f},
{64.7132f, 141.099f, 4.42165f},
{64.4732f, 141.166f, 5.14662f},
{64.2233f, 141.087f, 5.9082f},
{63.9616f, 140.838f, 6.71076f},
{63.6858f, 140.404f, 7.55892f},
{63.3938f, 139.774f, 8.45781f},
{63.0835f, 138.941f, 9.41272f},
{62.7525f, 137.899f, 10.4293f},
{62.3988f, 136.647f, 11.5131f},
{62.0203f, 135.187f, 12.6695f},
{61.6155f, 133.53f, 13.9034f},
{61.1834f, 131.693f, 15.2181f},
{60.7238f, 129.7f, 16.6159f},
{60.2374f, 127.586f, 18.0964f},
{59.7257f, 125.392f, 19.6567f},
{60.415f, 121.297f, 20.8148f},
{61.0871f, 117.437f, 21.9958f},
{61.7436f, 113.796f, 23.2003f},
{62.3857f, 110.361f, 24.4291f},
{63.0146f, 107.119f, 25.6826f},
{63.6314f, 104.059f, 26.9611f},
{64.237f, 101.171f, 28.2649f},
{64.832f, 98.4447f, 29.594f},
{65.4174f, 95.8732f, 30.9485f},
{65.9935f, 93.4484f, 32.3278f},
{66.5611f, 91.1636f, 33.7317f},
{67.1206f, 89.0124f, 35.1596f},
{67.6725f, 86.9889f, 36.6107f},
{68.2171f, 85.088f, 38.0841f},
{68.7549f, 83.3047f, 39.5787f},
{69.2861f, 81.6344f, 41.0931f},
{69.811f, 80.0728f, 42.626f},
{70.3301f, 78.6161f, 44.1757f},
{70.8434f, 77.2602f, 45.7405f},
{71.3512f, 76.0017f, 47.3183f},
{71.8538f, 74.8371f, 48.9072f},
{72.3513f, 73.7629f, 50.5052f},
{72.8439f, 72.7763f, 52.1097f},
{73.3318f, 71.8738f, 53.7187f},
{73.8152f, 71.0527f, 55.3296f},
{74.2942f, 70.3098f, 56.9401f},
{74.769f, 69.6423f, 58.5478f},
{75.2396f, 69.0474f, 60.1502f},
{75.7062f, 68.5222f, 61.7451f},
{76.1689f, 68.0639f, 63.3301f},
{76.6279f, 67.67f, 64.9028f},
{77.0832f, 67.3376f, 66.4614f},
{77.5349f, 67.0641f, 68.0037f},
{77.9831f, 66.8469f, 69.5279f},
{78.4279f, 66.6835f, 71.0322f},
{78.8695f, 66.5713f, 72.5151f},
{79.3078f, 66.508f, 73.9751f},
{79.743f, 66.491f, 75.4109f},
{80.175f, 66.5181f, 76.8215f},
{80.6041f, 66.5869f, 78.2057f},
{81.0303f, 66.6953f, 79.563f},
{81.4535f, 66.8411f, 80.8924f},
{81.874f, 67.0223f, 82.1936f},
{82.2916f, 67.2367f, 83.4663f},
{82.7066f, 67.4826f, 84.7099f},
{83.1189f, 67.758f, 85.9246f},
{83.5287f, 68.0612f, 87.1103f},
{83.9359f, 68.3905f, 88.267f},
{84.3406f, 68.7442f, 89.3948f},
{84.7429f, 69.1208f, 90.4941f},
{85.1427f, 69.5189f, 91.5651f},
{85.5402f, 69.937f, 92.6082f},
{85.9353f, 70.3737f, 93.6237f},
{86.3282f, 70.8279f, 94.6124f},
{86.7189f, 71.2983f, 95.5745f},
{87.1074f, 71.7837f, 96.5105f},
{87.4937f, 72.2832f, 97.4213f},
{87.8778f, 72.7956f, 98.3072f},
{88.2599f, 73.3202f, 99.1689f},
{88.6399f, 73.8557f, 100.007f},
{88.3336f, 73.9088f, 100.834f},
{88.026f, 73.9784f, 101.666f},
{87.717f, 74.0646f, 102.501f},
{87.4065f, 74.1679f, 103.341f},
{87.0947f, 74.2884f, 104.184f},
{86.7814f, 74.4266f, 105.03f},
{86.4666f, 74.5827f, 105.879f},
{86.1504f, 74.757f, 106.731f},
{85.8326f, 74.9498f, 107.585f},
{85.5133f, 75.1613f, 108.44f},
{85.1925f, 75.3919f, 109.297f},
{84.8701f, 75.6417f, 110.155f},
{84.5462f, 75.911f, 111.013f},
{84.2206f, 76.2003f, 111.872f},
{83.8934f, 76.5095f, 112.731f},
{83.5646f, 76.8391f, 113.589f},
{83.234f, 77.1891f, 114.446f},
{82.9018f, 77.56f, 115.303f},
{82.5678f, 77.9518f, 116.157f},
{82.2321f, 78.3649f, 117.01f},
{81.8946f, 78.7995f, 117.86f},
{81.5553f, 79.2556f, 118.708f},
{81.2141f, 79.7336f, 119.553f},
{80.8711f, 80.2337f, 120.394f},
{80.5262f, 80.7561f, 121.232f},
{80.1794f, 81.301f, 122.066f},
{79.8306f, 81.8685f, 122.895f},
{79.4799f, 82.4589f, 123.72f},
{79.1271f, 83.0722f, 124.54f},
{78.7722f, 83.7089f, 125.355f},
{78.4153f, 84.369f, 126.165f},
{78.0563f, 85.0526f, 126.969f},
{77.695f, 85.76f, 127.767f},
{77.3316f, 86.4914f, 128.559f},
{76.966f, 87.2469f, 129.344f},
{76.5981f, 88.0268f, 130.123f},
{76.2279f, 88.8312f, 130.896f},
{75.8552f, 89.6603f, 131.662f},
{75.4802f, 90.5144f, 132.42f},
{75.1028f, 91.3934f, 133.172f},
{74.7228f, 92.2978f, 133.916f},
{74.3403f, 93.2278f, 134.652f},
{73.9553f, 94.1834f, 135.382f},
{73.5675f, 95.165f, 136.103f},
{73.1771f, 96.1729f, 136.817f},
{72.7839f, 97.2071f, 137.524f},
{72.3879f, 98.268f, 138.222f},
{71.989f, 99.3558f, 138.913f},
{71.5872f, 100.471f, 139.595f},
{71.1824f, 101.613f, 140.27f},
{70.7745f, 102.784f, 140.937f},
{70.3635f, 103.982f, 141.596f},
{69.9493f, 105.209f, 142.247f},
{69.5318f, 106.464f, 142.89f},
{69.111f, 107.749f, 143.525f},
{68.6867f, 109.063f, 144.152f},
{68.2589f, 110.406f, 144.772f},
{67.8276f, 111.78f, 145.384f},
{67.3925f, 113.185f, 145.988f},
{66.9536f, 114.621f, 146.584f},
{67.3403f, 114.122f, 147.465f},
{67.7194f, 113.643f, 148.361f},
{68.0903f, 113.182f, 149.271f},
{68.4522f, 112.739f, 150.193f},
{68.8043f, 112.311f, 151.125f},
{69.1461f, 111.899f, 152.064f},
{69.4771f, 111.499f, 153.007f},
{69.7968f, 111.111f, 153.954f},
{70.1049f, 110.734f, 154.901f},
{70.4013f, 110.366f, 155.847f},
{70.6858f, 110.005f, 156.789f},
{70.9586f, 109.652f, 157.726f},
{71.2198f, 109.303f, 158.656f},
{71.4698f, 108.96f, 159.578f},
{71.7089f, 108.62f, 160.49f},
{71.9376f, 108.284f, 161.392f},
{72.1564f, 107.95f, 162.281f},
{72.3659f, 107.617f, 163.159f},
{72.5667f, 107.286f, 164.023f},
{72.7594f, 106.956f, 164.874f},
{72.9447f, 106.627f, 165.712f},
{73.1232f, 106.298f, 166.535f},
{73.2956f, 105.969f, 167.344f},
{73.4624f, 105.639f, 168.139f},
{73.6242f, 105.308f, 168.921f},
{73.7817f, 104.976f, 169.688f},
{73.9354f, 104.643f, 170.441f},
{74.0858f, 104.308f, 171.181f},
{74.2334f, 103.97f, 171.908f},
{74.3787f, 103.629f, 172.621f},
{74.5222f, 103.285f, 173.322f},
{74.6643f, 102.936f, 174.01f},
{74.8054f, 102.581f, 174.686f},
{74.9458f, 102.223f, 175.35f},
{75.0859f, 101.864f, 176.003f},
{75.2258f, 101.506f, 176.644f},
{75.3657f, 101.151f, 177.275f},
{75.5057f, 100.801f, 177.895f},
{75.6459f, 100.457f, 178.505f},
{75.7863f, 100.121f, 179.105f},
{75.927f, 99.7949f, 179.695f},
{76.068f, 99.4786f, 180.277f},
{76.2092f, 99.1733f, 180.849f},
{76.3507f, 98.8798f, 181.413f},
{76.4924f, 98.5984f, 181.969f},
{76.6343f, 98.3293f, 182.516f},
{76.7764f, 98.0728f, 183.056f},
{76.9187f, 97.8287f, 183.587f},
{77.0611f, 97.5969f, 184.112f},
{77.2036f, 97.3772f, 184.629f},
{77.3462f, 97.1693f, 185.139f},
{77.4888f, 96.9728f, 185.642f},
{77.6316f, 96.7874f, 186.138f},
{77.7744f, 96.6126f, 186.628f},
{77.9173f, 96.4481f, 187.111f},
{78.0602f, 96.2932f, 187.588f},
{78.2032f, 96.1478f, 188.059f},
{78.3463f, 96.0113f, 188.523f},
{78.4894f, 95.8833f, 188.982f},
{78.6326f, 95.7635f, 189.435f},
{78.1651f, 95.0134f, 189.89f},
{77.6947f, 94.2623f, 190.355f},
{77.2214f, 93.5099f, 190.829f},
{76.7451f, 92.7568f, 191.314f},
{76.2658f, 92.0026f, 191.809f},
{75.7835f, 91.2472f, 192.316f},
{75.2981f, 90.4909f, 192.834f},
{74.8097f, 89.7337f, 193.364f},
{74.3181f, 88.9754f, 193.906f},
{73.8234f, 88.2162f, 194.461f},
{73.3256f, 87.4562f, 195.028f},
{72.8245f, 86.6953f, 195.61f},
{72.3203f, 85.9336f, 196.205f},
{71.8128f, 85.1712f, 196.815f},
{71.3021f, 84.4082f, 197.441f},
{70.7881f, 83.6447f, 198.082f},
{70.2708f, 82.8809f, 198.739f},
{69.7502f, 82.1168f, 199.413f},
{69.2263f, 81.3527f, 200.105f},
{68.6991f, 80.5887f, 200.816f},
{68.1685f, 79.825f, 201.545f},
{67.6345f, 79.0619f, 202.294f},
{67.0972f, 78.2996f, 203.063f},
{66.5565f, 77.5384f, 203.854f},
{66.0124f, 76.7788f, 204.668f},
{65.4649f, 76.021f, 205.504f},
{64.914f, 75.2653f, 206.364f},
{64.3598f, 74.5125f, 207.25f},
{63.8021f, 73.7628f, 208.162f},
{63.2411f, 73.017f, 209.1f},
{62.6766f, 72.2755f, 210.067f},
{62.1088f, 71.5389f, 211.064f},
{61.5377f, 70.8083f, 212.091f},
{60.9632f, 70.0842f, 213.15f},
{60.3854f, 69.3674f, 214.242f},
{59.8042f, 68.6592f, 215.368f},
{59.2198f, 67.9605f, 216.531f},
{58.6322f, 67.2723f, 217.73f},
{58.0413f, 66.596f, 218.968f},
{57.4473f, 65.933f, 220.246f},
{56.8501f, 65.2846f, 221.565f},
{56.2498f, 64.6525f, 222.926f},
{55.6464f, 64.0385f, 224.332f},
{55.0401f, 63.4442f, 225.783f},
{54.4308f, 62.8719f, 227.281f},
{53.8186f, 62.3236f, 228.827f},
{53.2037f, 61.8018f, 230.422f},
{52.586f, 61.3088f, 232.066f},
{51.9656f, 60.8473f, 233.762f},
{51.3427f, 60.4202f, 235.508f},
{50.7173f, 60.0304f, 237.307f},
{50.0895f, 59.6811f, 239.157f},
{49.4594f, 59.3758f, 241.059f},
{48.8271f, 59.1178f, 243.012f},
{48.1927f, 58.9109f, 245.015f},
{47.5565f, 58.7588f, 247.067f},
{46.9184f, 58.6657f, 249.166f},
{46.2786f, 58.6353f, 251.31f},
{45.6373f, 58.672f, 253.496f},
{44.9946f, 58.78f, 255.72f},
{45.5579f, 57.6759f, 258.735f},
{46.1187f, 56.7423f, 261.813f},
{46.6769f, 55.9802f, 264.938f},
{47.2324f, 55.3889f, 268.091f},
{47.7853f, 54.9664f, 271.255f},
{48.3353f, 54.7091f, 274.411f},
{48.8826f, 54.6119f, 277.541f},
{49.427f, 54.6685f, 280.626f},
{49.9685f, 54.8715f, 283.65f},
{50.5071f, 55.2125f, 286.6f},
{51.0428f, 55.6824f, 289.465f},
{51.5755f, 56.2716f, 292.233f},
{52.1053f, 56.9705f, 294.899f},
{52.632f, 57.7694f, 297.458f},
{53.1557f, 58.6588f, 299.906f},
{53.6763f, 59.6295f, 302.244f},
{54.1938f, 60.6726f, 304.471f},
{54.7081f, 61.78f, 306.59f},
{55.2194f, 62.9439f, 308.603f},
{55.7274f, 64.1571f, 310.514f},
{56.2323f, 65.4132f, 312.327f},
{56.7341f, 66.706f, 314.047f},
{57.2327f, 68.03f, 315.678f},
{57.7281f, 69.3803f, 317.224f},
{58.2204f, 70.7526f, 318.691f},
{58.7095f, 72.1427f, 320.083f},
{59.1955f, 73.5471f, 321.405f},
{59.6784f, 74.9625f, 322.66f},
{60.1581f, 76.386f, 323.853f},
{60.6348f, 77.8153f, 324.987f},
{61.1083f, 79.2479f, 326.067f},
{61.5788f, 80.6818f, 327.096f},
{62.0463f, 82.1154f, 328.076f},
{62.5107f, 83.5468f, 329.012f},
{62.9721f, 84.975f, 329.905f},
{63.4305f, 86.3985f, 330.759f},
{63.886f, 87.8164f, 331.575f},
{64.3385f, 89.2277f, 332.356f},
{64.7881f, 90.6318f, 333.105f},
{65.2349f, 92.0277f, 333.822f},
{65.6787f, 93.4148f, 334.511f},
{66.1197f, 94.793f, 335.172f},
{66.558f, 96.1613f, 335.808f},
{66.9934f, 97.5197f, 336.419f},
{67.426f, 98.8679f, 337.007f},
{67.8559f, 100.205f, 337.574f},
{68.2832f, 101.532f, 338.12f},
{68.7077f, 102.848f, 338.647f},
{69.1296f, 104.153f, 339.156f},
{69.5489f, 105.447f, 339.647f},
{69.9655f, 106.73f, 340.122f},
{70.3796f, 108.001f, 340.581f},
{70.7912f, 109.261f, 341.026f},
{71.2002f, 110.511f, 341.456f},
{71.6068f, 111.749f, 341.873f},
{72.0109f, 112.976f, 342.277f},
{72.4125f, 114.192f, 342.669f},
{72.8118f, 115.397f, 343.05f},
{73.2087f, 116.592f, 343.42f},
{73.6032f, 117.775f, 343.779f},
{73.4252f, 118.173f, 344.133f},
{73.2469f, 118.576f, 344.489f},
{73.0681f, 118.983f, 344.847f},
{72.889f, 119.396f, 345.207f},
{72.7095f, 119.815f, 345.569f},
{72.5296f, 120.238f, 345.933f},
{72.3492f, 120.668f, 346.299f},
{72.1684f, 121.102f, 346.668f},
{71.9872f, 121.542f, 347.039f},
{71.8054f, 121.987f, 347.412f},
{71.6232f, 122.439f, 347.788f},
{71.4405f, 122.896f, 348.166f},
{71.2572f, 123.359f, 348.547f},
{71.0734f, 123.827f, 348.931f},
{70.889f, 124.303f, 349.318f},
{70.704f, 124.784f, 349.709f},
{70.5184f, 125.273f, 350.102f},
{70.3321f, 125.768f, 350.499f},
{70.1452f, 126.271f, 350.899f},
{69.9575f, 126.781f, 351.303f},
{69.7692f, 127.299f, 351.711f},
{69.5801f, 127.825f, 352.123f},
{69.3902f, 128.361f, 352.54f},
{69.1995f, 128.905f, 352.961f},
{69.0081f, 129.46f, 353.387f},
{68.8158f, 130.024f, 353.817f},
{68.6227f, 130.599f, 354.254f},
{68.4287f, 131.184f, 354.696f},
{68.2339f, 131.78f, 355.145f},
{68.0382f, 132.387f, 355.599f},
{67.8417f, 133.004f, 356.061f},
{67.6443f, 133.631f, 356.531f},
{67.446f, 134.265f, 357.008f},
{67.2468f, 134.907f, 357.495f},
{67.0466f, 135.552f, 357.991f},
{66.8454f, 136.198f, 358.497f},
{66.643f, 136.84f, 359.014f},
{66.4394f, 137.473f, 359.544f}
};

__CONSTANT__ float float_epsilon = 0.0000000596046448f;
__CONSTANT__ float HALF_MAXIMUM = 65504.0f;

__CONSTANT__ float PI = 3.141592653589793f;

__CONSTANT__ float L_A = 100.0f;

__CONSTANT__ float Y_b = 20.0f;

__CONSTANT__ float referenceLuminance = 100.0f;

__CONSTANT__ float3 surround = {0.9f, 0.59f, 0.9f};

__CONSTANT__ float3 d65White = {95.0455927052f, 100.0f, 108.9057750760f};

// __CONSTANT__ float gamut_gamma = 1.137f; // surround.y * (1.48 + sqrt(Y_b / Y_w)))
__CONSTANT__ float model_gamma = 0.879464f; // reciprocal of above
__CONSTANT__ float lowerHullGamma = 1.18;

// Gamut Compression parameters
__CONSTANT__ float cuspMidBlend = 0.5f;
__CONSTANT__ float smoothCusps = 0.0f;
__CONSTANT__ float midJ = 34.0965f; // ~10 nits in Hellwig J
__CONSTANT__ float focusDistance = 3.5f;
__CONSTANT__ float4 compressionFuncParams = {0.75f, 1.1f, 1.05f, 1.2f};

// DanieleEvoCurve (ACES2 candidate) parameters
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

__DEVICE__ inline float3 compress_aces(float3 rgb, float3 c, float3 m, float3 y, int invert)
  {
    float ach = max(rgb.x, max(rgb.y, rgb.z));
    float3 d = 0.0f;

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
    float A_w = 2.0f * R_aw + G_aw + 0.05f * B_aw;

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
    float A_w = 2.0f * R_aw + G_aw + 0.05f * B_aw;

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

    float3 RGB_c = post_adaptation_non_linear_response_compression_inverse(RGB_a, F_L);

    RGB_c = uncompress(RGB_c);

    // # Step 6
    float3 RGB = RGB_c / D_RGB;
    

    // # Step 7
    float3 XYZ = vector_dot(MATRIX_INVERSE_16, RGB);

    return XYZ;
}

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
      for(int i = 1; i < 360; ++i)
      {
        if( h <= gamutCuspTable[i].z && h > gamutCuspTable[i-1].z )
        {
          lo = gamutCuspTable[i-1];
          hi = gamutCuspTable[i];
//           break;
        }
      }
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
      for(int i = 1; i < 360; ++i)
      {
        if( h <= cGamutCuspTable[i].z && h > cGamutCuspTable[i-1].z )
        {
          lo = cGamutCuspTable[i-1];
          hi = cGamutCuspTable[i];
//           break;
        }
      }
    }

    float t = (h - lo.z) / (hi.z - lo.z);

    float cuspJ = lerp(lo.x, hi.x, t);
    float cuspM = lerp(lo.y, hi.y, t);

    return make_float2(cuspJ,cuspM);
}

__DEVICE__ inline float reachFromTableAP1(float h)
{
    int lo = (int)_floorf(mod(h, 360.0f));
    int hi = (int)_ceilf(mod(h, 360.0f));
    if (hi == 360)
    {
        hi = 0;
    }
    float t = _fmod(h, 1.0f);

    return lerp(gamutCuspTableAP1[lo], gamutCuspTableAP1[hi], t);
}

  // Compress/expand a range of values from 0 to limit (0 being the achromatic).  Doesn't
  // affect anything beyond the limit.  The k1 parameter affects the strength of the curve,
  // the k2 parameter affects the expansion rate of the curve.
  // https://www.desmos.com/calculator/vqxgfzzyvx
__DEVICE__ inline float chroma_range(float x, float limit, float k1, float k2, int inverse)
  {
    if (x > limit)
      return x;

    k2 = max(k2, 0.002f);
    k1 = _sqrtf(k1 * k1 + k2 * k2);
    float k3 = (limit + k1) / (limit + k2);

    if (!inverse)
      return 0.5f * (k3 * x - k1 + _sqrtf((k3 * x - k1) * (k3 * x - k1) + 4 * k2 * k3 * x));
    else
      return (x * x + k1 * x) / (k3 * (x + k2));
  }

  // In-gamut chroma compression
  //
  // Compresses colors inside the gamut with the aim for colorfulness to have an
  // appropriate rate of change from display black to display white, and from
  // achromatic outward to purer colors.
  //
  // Steps:
  //  - Scale down M by tonescaledJ / origJ
  //  - Normalize M to compression gamut boundary (becomes hue-dependent)
  //  - Expand and compress M with chroma_range().  Compression is increased as tonescaledJ
  //    increases to create the path-to-white.
  //  - Denormalize M with the gamut cusp
  //
__DEVICE__ inline float chromaCompression(float3 JMh, float origJ, float linear, int invert)
  {
    float M = JMh.y;
    if (M == 0.0f)
      return M;

    // Enforce sane input
    M = min(2500.0f, M);

    float nJ = JMh.x / limitJmax;
    float snJ = _powf(max(0.0f, 1.0f - nJ), ccParams.z);
    float scaling = _powf(JMh.x / origJ, model_gamma);
    float Mcusp = cCuspFromTable(JMh.z).y;
    float limit = _powf(nJ, model_gamma) * reachFromTableAP1(JMh.z) / Mcusp;

    if (!invert)
    {
        M *= scaling;
        M /= Mcusp;
        M = chroma_range(M, limit, snJ * sat, _sqrtf(nJ * nJ + sat_thr), 1);
        M = chroma_range(M, limit, nJ * ccParams.y, snJ, 0);
        M *= Mcusp;
    }
    else
    {
        M /= Mcusp;
        M = chroma_range(M, limit, nJ * ccParams.y, snJ, 1);
        M = chroma_range(M, limit, snJ * sat, _sqrtf(nJ * nJ + sat_thr), 0);
        M *= Mcusp;
        M /= scaling;
    }

    return M;
  }

__DEVICE__ inline float3 forwardTonescale( float3 inputJMh, int compressChroma)
{
    float3 outputJMh;
    float3 monoJMh = make_float3(inputJMh.x, 0.0f, 0.0f);
    float3 luminanceXYZ = Hellwig2022_JMh_to_XYZ( monoJMh, d65White);
    float linear = luminanceXYZ.y / referenceLuminance;

    // only Daniele Evo tone scale
    float luminanceTS = daniele_evo_fwd(linear);

    float3 tonemappedmonoJMh = XYZ_to_Hellwig2022_JMh(d65White * luminanceTS, d65White);
    float3 tonemappedJMh = make_float3(tonemappedmonoJMh.x, inputJMh.y, inputJMh.z);

    outputJMh = tonemappedJMh;

    // Chroma Compression)
    if (compressChroma)
    {
        outputJMh.y = chromaCompression(outputJMh, inputJMh.x, linear, 0);
    }

    return outputJMh;
}

__DEVICE__ inline float3 inverseTonescale( float3 JMh, int compressChroma)
  {
    float3 tonemappedJMh = JMh;

    float3 untonemappedColourJMh = tonemappedJMh;
    
    float3 monoTonemappedJMh = make_float3(tonemappedJMh.x, 0.0f, 0.0f);

    float3 luminanceXYZ = Hellwig2022_JMh_to_XYZ( monoTonemappedJMh, d65White);
    float luminance = luminanceXYZ.y;

    float linear = daniele_evo_rev(luminance / referenceLuminance);

    float3 untonemappedMonoJMh = XYZ_to_Hellwig2022_JMh(d65White * linear, d65White);
    untonemappedColourJMh = make_float3(untonemappedMonoJMh.x,tonemappedJMh.y,tonemappedJMh.z); 

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
    // take float h, divide by 10, and lerp between index values from upperHullGamma_0
    int index = int(h/10.0f);
    float t = (h - index*10.0f) / 10.0f;
    float gamma = 1.0f;
    if (index < 35)
    {
        gamma = lerp(upperHullGamma[index], upperHullGamma[index+1], t);
    }
    else
    {
        gamma = lerp(upperHullGamma[35], upperHullGamma[0], t);
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
    float gamma_bottom = lowerHullGamma;

    float slope = 0.0f;

    float s = max(0.000001f, smoothness);
    JM_cusp.x *= 1.0f + 0.06f * s;   // J
    JM_cusp.y *= 1.0f + 0.18f * s;   // M

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

__DEVICE__ inline float3 compressGamut(float3 JMh, int invert)
{
    float2 project_from = make_float2(JMh.x, JMh.y);
    float2 JMcusp = cuspFromTable(JMh.z);

    if (project_from.y == 0.0f)
      return JMh;

    // Calculate where the out of gamut color is projected to
    float focusJ = lerp(JMcusp.x, midJ, cuspMidBlend);

    float slope_gain = limitJmax * focusDistance;

    // Find gamut intersection
    float3 nickBoundryReturn =  findGamutBoundaryIntersection(JMh, JMcusp, focusJ, limitJmax, slope_gain, smoothCusps);
    float2 JMboundary = make_float2(nickBoundryReturn.x,nickBoundryReturn.y);
    float2 project_to = make_float2(nickBoundryReturn.z,0.0f);
    float projectJ = nickBoundryReturn.z;

    // Calculate AP1 Reach boundary
    float reachMaxM = reachFromTableAP1(JMh.z);

    // slope is recalculated here because it was a local variable in findGamutBoundaryIntersection
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
    
    float difference = max(1.0001f, boundaryNick / JMboundary.y);
    float threshold = max(compressionFuncParams.x, 1.0f / difference);

    // Compress the out of gamut color along the projection line
    float v = project_from.y / JMboundary.y;
    v = compressPowerP(v, threshold, difference, compressionFuncParams.w, invert);
    float2 JMcompressed = project_to + v * (JMboundary - project_to);

    return make_float3(JMcompressed.x, JMcompressed.y, JMh.z);
}

  // encode linear values as ST2084 PQ
__DEVICE__ inline float linear_to_ST2084( float v )
{
    float Y_p = spow(_fmaxf(0.0f, v) / st2084_L_p, st2084_m_1);

    return spow((st2084_c_1 + st2084_c_2 * Y_p) / (st2084_c_3 * Y_p + 1.0f), st2084_m_2);
}