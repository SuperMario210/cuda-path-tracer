#ifndef CUDA_BASE_LINEAR_MATH_CUH


#include <math.h>
#include <cuda_runtime.h> // for __host__  __device__

#define FW_ASSERT(X) ((void)0)
// FW_ASSERT(X) ((X) ? ((void)0) : FW::fail("Assertion failed!\n%s:%d\n%s", __FILE__, __LINE__, #X)) in DEBUG


//----------------------------------
// Matrix algebra
//----------------------------------

class Mat4f
{
public:

    inline    const float*          getPtr(void) const            { return &m00; }
    inline    float*                getPtr(void)                  { return &m00; }
    inline    const float4&        col(int c) const				{ FW_ASSERT(c >= 0 && c < 4); return *(const float4*)(getPtr() + c * 4); }
    inline    float4&	          col(int c)					{ FW_ASSERT(c >= 0 && c < 4); return *(float4*)(getPtr() + c * 4); }
    inline    const float4&        getCol(int c) const		    { return col(c); }
    inline    float4			      getCol0() const		    { float4 col; col.x = m00; col.y = m01; col.z = m02; col.w = m03; return col; }
    inline    float4			      getCol1() const		    { float4 col; col.x = m10; col.y = m11; col.z = m12; col.w = m13; return col; }
    inline    float4			      getCol2() const		    { float4 col; col.x = m20; col.y = m21; col.z = m22; col.w = m23; return col; }
    inline    float4			      getCol3() const		    { float4 col; col.x = m30; col.y = m31; col.z = m32; col.w = m33; return col; }
    inline    float4               getRow(int r) const;
    inline    Mat4f               inverted4x4(void);
    inline    void                invert(void)                  { set(inverted4x4()); }
    inline    const float&        get(int idx) const             { FW_ASSERT(idx >= 0 && idx < 4 * 4); return getPtr()[idx]; }
    inline    float&              get(int idx)                   { FW_ASSERT(idx >= 0 && idx < 4 * 4); return getPtr()[idx]; }
    inline    const float&        get(int r, int c) const        { FW_ASSERT(r >= 0 && r < 4 && c >= 0 && c < 4); return getPtr()[r + c * 4]; }
    inline    float&              get(int r, int c)              { FW_ASSERT(r >= 0 && r < 4 && c >= 0 && c < 4); return getPtr()[r + c * 4]; }
    inline    void                set(const float& a)            { for (int i = 0; i < 4 * 4; i++) get(i) = a; }
    inline    void                set(const float* ptr)          { FW_ASSERT(ptr); for (int i = 0; i < 4 * 4; i++) get(i) = ptr[i]; }
    inline    void                setZero(void)                  { set((float)0); }
    inline    void                setIdentity(void)              { setZero(); for (int i = 0; i < 4; i++) get(i, i) = (float)1; }
    inline    void				  setCol(int c, const float4& v)   { col(c) = v; }
    inline    void				  setCol0(const float4& v)   { m00 = v.x; m01 = v.y; m02 = v.z; m03 = v.w; }
    inline    void				  setCol1(const float4& v)   { m10 = v.x; m11 = v.y; m12 = v.z; m13 = v.w; }
    inline    void				  setCol2(const float4& v)   { m20 = v.x; m21 = v.y; m22 = v.z; m23 = v.w; }
    inline    void				  setCol3(const float4& v)   { m30 = v.x; m31 = v.y; m32 = v.z; m33 = v.w; }
    inline    void                setRow(int r, const float4& v);
    inline    void                set(const Mat4f& v) { set(v.getPtr()); }
    inline    Mat4f&              operator=   (const float& a)                { set(a); return *(Mat4f*)this; }
    inline    Mat4f               operator*   (const float& a) const          { Mat4f r; for (int i = 0; i < 4 * 4; i++) r.get(i) = get(i) * a; return r; }
    inline    const float&        operator()  (int r, int c) const        { return get(r, c); }
    inline    float&              operator()  (int r, int c)              { return get(r, c); }

    inline                    Mat4f(void)                      { setIdentity(); }
    inline    explicit        Mat4f(float a)                     { set(a); }
    static inline Mat4f       fromPtr(const float* ptr)            { Mat4f v; v.set(ptr); return v; }

    inline Mat4f(const Mat4f& v) { set(v); }
    inline Mat4f& operator=(const Mat4f& v) { set(v); return *this; }

public:
    float             m00, m10, m20, m30;
    float             m01, m11, m21, m31;
    float             m02, m12, m22, m32;
    float             m03, m13, m23, m33;
};

inline Mat4f invert(Mat4f& v)  { return v.inverted4x4(); }

//------------------------------------------------------------------------

float4 Mat4f::getRow(int idx) const
{
    float4 r;
    for (int i = 0; i < 4; i++)
        (&r.x)[i] = get(idx, i);
    return r;
}

void Mat4f::setRow(int idx, const float4& v)
{
    for (int i = 0; i < 4; i++)
        get(idx, i) = (&v.x)[i];
}

// efficient column major matrix inversion function from http://rodolphe-vaillant.fr/?e=7

Mat4f Mat4f::inverted4x4(void)
{
    float inv[16];
    float m[16] = {	m00, m10, m20, m30,
                       m01, m11, m21, m31,
                       m02, m12, m22, m32,
                       m03, m13, m23, m33 };

    inv[0]  =  m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4]  = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8]  =  m[4] * m[9] *  m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] *  m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1]  = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5]  =  m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9]  = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] =  m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2]  =  m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6]  = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] =  m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3]  = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7]  =  m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] =  m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

    float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return Mat4f();

    det = 1.f / det;
    Mat4f inverse;
    for (int i = 0; i < 16; i++)
        inverse.get(i) = inv[i] * det;

    return inverse;
}

#define CUDA_BASE_LINEAR_MATH_CUH

#endif //CUDA_BASE_LINEAR_MATH_CUH
