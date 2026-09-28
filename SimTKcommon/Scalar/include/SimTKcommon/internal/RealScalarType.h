#ifndef SimTK_SIMTKCOMMON_REAL_SCALAR_TYPE_H_
#define SimTK_SIMTKCOMMON_REAL_SCALAR_TYPE_H_

/* -------------------------------------------------------------------------- *
 *                       Simbody(tm): SimTKcommon                             *
 * -------------------------------------------------------------------------- *
 * This is part of the SimTK biosimulation toolkit originating from           *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org/home/simbody.  *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 * Authors: Nick Bianco                                                       *
 * Contributors:                                                              *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may    *
 * not use this file except in compliance with the License. You may obtain a  *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.         *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 * -------------------------------------------------------------------------- */

/** @file
Support for using a user-defined, class-type real scalar T (for
example an automatic differentiation or symbolic type such as casadi::SX) as
the element type of the SimTK small matrix and mechanics classes (Vec, Mat,
SymMat, Row, Rotation_, Transform_, SpatialVec_, MassProperties_, etc.), and
with Simbody's scalar-templatized multibody tree computations.

The scalar type T must:
  - be default constructible, copyable, and implicitly constructible from
    double (and hence from int);
  - support +, -, *, / with itself and with double, unary -, and +=, -=, *=,
    /=; the results must be (convertible to) T;
  - provide sin, cos, tan, asin, acos, atan, atan2, sqrt, exp, log, pow, and
    abs, findable by argument-dependent lookup (i.e., in T's namespace).

It need \e not be comparable or convertible to bool. The small matrix and
multibody computations that must work with T do not branch on values.

To define the traits for T, write a header that, in this order:
  -# includes "SimTKcommon/Scalar.h";
  -# defines, in namespace SimTK, the value queries
     @code
        bool isNaN(const T&);
        bool isFinite(const T&);
        bool isInf(const T&);
        bool isNumericallyEqual(const T& a, const T& b, double tol);
     @endcode
  -# invokes SimTK_DEFINE_REAL_SCALAR_TRAITS(T) at global scope;
  -# includes "SimTKcommon/SmallMatrix.h";
  -# invokes SimTK_DEFINE_REAL_SCALAR_MATRIX_OPERATORS(T) at global scope.

Then include that header \e instead of (or before) any other SimTKcommon
header in translation units that use T. The traits must be visible before
any SimTK type is instantiated with T. See the Dual type in
SimTKcommon/tests/TestDualScalar.cpp in the Simbody source for an example.

T is treated as a real scalar with double precision for purposes of
tolerances, constants, and digit counts. Only header-only functionality is
available: non-inline members of the mechanics classes (e.g. most Rotation_
angle and quaternion conversions) are compiled into SimTKcommon for float and
double only. **/

#include "SimTKcommon/Scalar.h"

#include <cmath>
#include <limits>
#include <type_traits>

namespace SimTK {
// Helpers that find the math functions for a class-type scalar by
// argument-dependent lookup (these can't be called from inside NTraits<T>,
// whose static members of the same names would hide them).
namespace RealScalarDetail {
template <class T> inline T sqrtOf(const T& t) {using std::sqrt; return sqrt(t);}
template <class T> inline T absOf (const T& t) {using std::abs;  return abs(t);}
} // namespace RealScalarDetail
} // namespace SimTK

/** Define SimTK's scalar traits (NTraits, CNT, Widest, etc.) and scalar
functions (square(), cube(), and value queries for negator<T>) for a
class-type real scalar T. See RealScalarType.h for the requirements. **/
#define SimTK_DEFINE_REAL_SCALAR_TRAITS(T_)                                    \
namespace SimTK {                                                              \
inline T_ square(const T_& x) {return x*x;}                                    \
inline T_ cube(const T_& x)   {return x*x*x;}                                  \
                                                                               \
inline bool isNumericallyEqual(const T_& a, const T_& b)                       \
{   return isNumericallyEqual(a, b, RTraits<double>::getDefaultTolerance()); } \
inline bool isNumericallyEqual(const T_& a, double b,                          \
        double tol = RTraits<double>::getDefaultTolerance())                   \
{   return isNumericallyEqual(a, T_(b), tol); }                                \
inline bool isNumericallyEqual(double a, const T_& b,                          \
        double tol = RTraits<double>::getDefaultTolerance())                   \
{   return isNumericallyEqual(T_(a), b, tol); }                                \
inline bool isNumericallyEqual(const T_& a, float b,                           \
        double tol = RTraits<float>::getDefaultTolerance())                    \
{   return isNumericallyEqual(a, T_(b), tol); }                                \
inline bool isNumericallyEqual(float a, const T_& b,                           \
        double tol = RTraits<float>::getDefaultTolerance())                    \
{   return isNumericallyEqual(T_(a), b, tol); }                                \
inline bool isNumericallyEqual(const T_& a, int b,                             \
        double tol = RTraits<double>::getDefaultTolerance())                   \
{   return isNumericallyEqual(a, T_(b), tol); }                                \
inline bool isNumericallyEqual(int a, const T_& b,                             \
        double tol = RTraits<double>::getDefaultTolerance())                   \
{   return isNumericallyEqual(T_(a), b, tol); }                                \
                                                                               \
template <> class NTraits<T_> {                                                \
public:                                                                        \
    typedef T_               T;                                                \
    typedef negator<T>       TNeg;                                             \
    typedef T                TWithoutNegator;                                  \
    typedef T                TReal;                                            \
    typedef T                TImag;                                            \
    typedef complex<T>       TComplex;                                         \
    typedef T                THerm;                                            \
    typedef T                TPosTrans;                                        \
    typedef T                TSqHermT;                                         \
    typedef T                TSqTHerm;                                         \
    typedef T                TElement;                                         \
    typedef T                TRow;                                             \
    typedef T                TCol;                                             \
    typedef T                TSqrt;                                            \
    typedef T                TAbs;                                             \
    typedef T                TStandard;                                        \
    typedef T                TInvert;                                          \
    typedef T                TNormalize;                                       \
    typedef T                Scalar;                                           \
    typedef T                ULessScalar;                                      \
    typedef T                Number;                                           \
    typedef T                StdNumber;                                        \
    typedef T                Precision;                                        \
    typedef T                ScalarNormSq;                                     \
    /* Any operation between a T and another numerical type P produces      */ \
    /* whatever P produces when combined with a double, but with T elements.*/ \
    template <class P> struct Result {                                         \
        typedef typename CNT<P>::template Result<T>::Mul Mul;                  \
        typedef typename CNT< typename CNT<P>::THerm >::template               \
                                                    Result<T>::Mul Dvd;        \
        typedef typename CNT<P>::template Result<T>::Add Add;                  \
        typedef typename CNT< typename CNT<P>::TNeg >::template                \
                                                    Result<T>::Add Sub;        \
    };                                                                         \
    template <class P> struct Substitute {typedef P Type;};                    \
    enum {                                                                     \
        NRows               = 1,                                               \
        NCols               = 1,                                               \
        RowSpacing          = 1,                                               \
        ColSpacing          = 1,                                               \
        NPackedElements     = 1,                                               \
        NActualElements     = 1,                                               \
        NActualScalars      = 1,                                               \
        ImagOffset          = 0,                                               \
        RealStrideFactor    = 1,                                               \
        ArgDepth            = SCALAR_DEPTH,                                    \
        IsScalar            = 1,                                               \
        IsULessScalar       = 1,                                               \
        IsNumber            = 1,                                               \
        IsStdNumber         = 1,                                               \
        IsPrecision         = 1,                                               \
        SignInterpretation  = 1                                                \
    };                                                                         \
    static const T* getData(const T& t) {return &t;}                           \
    static T*       updData(T& t)       {return &t;}                           \
    static const T& real(const T& t) {return t;}                               \
    static T&       real(T& t)       {return t;}                               \
    static const T& imag(const T&)   {static const T v(0); return v;}          \
    static T&       imag(T&)         {static T v(0); assert(false); return v;} \
    static const TNeg& negate(const T& t)                                      \
    {   return reinterpret_cast<const TNeg&>(t); }                             \
    static TNeg& negate(T& t) {return reinterpret_cast<TNeg&>(t);}             \
    static const THerm& transpose(const T& t) {return t;}                      \
    static       THerm& transpose(T& t) {return t;}                            \
    static const TPosTrans& positionalTranspose(const T& t) {return t;}        \
    static       TPosTrans& positionalTranspose(T& t) {return t;}              \
    static const TWithoutNegator& castAwayNegatorIfAny(const T& t) {return t;} \
    static       TWithoutNegator& updCastAwayNegatorIfAny(T& t) {return t;}    \
    static ScalarNormSq scalarNormSqr(const T& t) {return t*t;}                \
    static TSqrt sqrt(const T& t) {return RealScalarDetail::sqrtOf(t);}        \
    static TAbs  abs(const T& t)  {return RealScalarDetail::absOf(t);}         \
    static const TStandard& standardize(const T& t) {return t;}                \
    static TNormalize normalize(const T& t) {return t/abs(t);}                 \
    static TInvert invert(const T& t) {return T(1)/t;}                         \
    /* properties of this floating point representation */                    \
    static T getEps()           {return T(RTraits<double>::getEps());}         \
    static T getSignificant()   {return T(RTraits<double>::getSignificant());} \
    static T getNaN()   {return T(std::numeric_limits<double>::quiet_NaN());}  \
    static T getInfinity() {return T(std::numeric_limits<double>::infinity());}\
    static T getLeastPositive() {return T(std::numeric_limits<double>::min());}\
    static T getMostPositive()  {return T(std::numeric_limits<double>::max());}\
    static T getLeastNegative() {return T(-std::numeric_limits<double>::min());}\
    static T getMostNegative()  {return T(-std::numeric_limits<double>::max());}\
    static T getSqrtEps()       {return T(NTraits<double>::getSqrtEps());}     \
    static T getTiny()          {return T(NTraits<double>::getTiny());}        \
    static bool isFinite(const T& t) {return SimTK::isFinite(t);}              \
    static bool isNaN   (const T& t) {return SimTK::isNaN(t);}                 \
    static bool isInf   (const T& t) {return SimTK::isInf(t);}                 \
    static double getDefaultTolerance()                                        \
    {   return RTraits<double>::getDefaultTolerance(); }                       \
    static bool isNumericallyEqual(const T& t, const T& u)                     \
    {   return SimTK::isNumericallyEqual(t,u,getDefaultTolerance()); }         \
    static bool isNumericallyEqual(const T& t, const float& f)                 \
    {   return SimTK::isNumericallyEqual(t,f); }                               \
    static bool isNumericallyEqual(const T& t, const double& d)                \
    {   return SimTK::isNumericallyEqual(t,d); }                               \
    static bool isNumericallyEqual(const T& t, int i)                          \
    {   return SimTK::isNumericallyEqual(t,i); }                               \
    static bool isNumericallyEqual(const T& t, const T& u, double tol)         \
    {   return SimTK::isNumericallyEqual(t,u,tol); }                           \
    static bool isNumericallyEqual(const T& t, const float& f, double tol)     \
    {   return SimTK::isNumericallyEqual(t,f,tol); }                           \
    static bool isNumericallyEqual(const T& t, const double& d, double tol)    \
    {   return SimTK::isNumericallyEqual(t,d,tol); }                           \
    static bool isNumericallyEqual(const T& t, int i, double tol)              \
    {   return SimTK::isNumericallyEqual(t,i,tol); }                           \
    /* Carefully calculated constants. */                                      \
    static T getZero()         {return T(0);}                                  \
    static T getOne()          {return T(1);}                                  \
    static T getMinusOne()     {return T(-1);}                                 \
    static T getTwo()          {return T(2);}                                  \
    static T getThree()        {return T(3);}                                  \
    static T getOneHalf()      {return T(NTraits<double>::getOneHalf());}      \
    static T getOneThird()     {return T(NTraits<double>::getOneThird());}     \
    static T getOneFourth()    {return T(NTraits<double>::getOneFourth());}    \
    static T getOneFifth()     {return T(NTraits<double>::getOneFifth());}     \
    static T getOneSixth()     {return T(NTraits<double>::getOneSixth());}     \
    static T getOneSeventh()   {return T(NTraits<double>::getOneSeventh());}   \
    static T getOneEighth()    {return T(NTraits<double>::getOneEighth());}    \
    static T getOneNinth()     {return T(NTraits<double>::getOneNinth());}     \
    static T getPi()           {return T(NTraits<double>::getPi());}           \
    static T getOneOverPi()    {return T(NTraits<double>::getOneOverPi());}    \
    static T getE()            {return T(NTraits<double>::getE());}            \
    static T getLog2E()        {return T(NTraits<double>::getLog2E());}        \
    static T getLog10E()       {return T(NTraits<double>::getLog10E());}       \
    static T getSqrt2()        {return T(NTraits<double>::getSqrt2());}        \
    static T getOneOverSqrt2() {return T(NTraits<double>::getOneOverSqrt2());} \
    static T getSqrt3()        {return T(NTraits<double>::getSqrt3());}        \
    static T getOneOverSqrt3() {return T(NTraits<double>::getOneOverSqrt3());} \
    static T getCubeRoot2()    {return T(NTraits<double>::getCubeRoot2());}    \
    static T getCubeRoot3()    {return T(NTraits<double>::getCubeRoot3());}    \
    static T getLn2()          {return T(NTraits<double>::getLn2());}          \
    static T getLn10()         {return T(NTraits<double>::getLn10());}         \
    /* integer digit counts useful for formatted input and output */           \
    static constexpr int getNumDigits()                                        \
    {   return NTraits<double>::getNumDigits(); }                              \
    static constexpr int getLosslessNumDigits()                                \
    {   return NTraits<double>::getLosslessNumDigits(); }                      \
};                                                                             \
                                                                               \
/* Any combination of T with a built-in real type or another T is a T. */     \
template<> struct NTraits<T_>::Result<float>                                   \
  {typedef T_ Mul; typedef Mul Dvd; typedef Mul Add; typedef Mul Sub;};        \
template<> struct NTraits<T_>::Result<double>                                  \
  {typedef T_ Mul; typedef Mul Dvd; typedef Mul Add; typedef Mul Sub;};        \
template<> struct NTraits<T_>::Result<T_>                                      \
  {typedef T_ Mul; typedef Mul Dvd; typedef Mul Add; typedef Mul Sub;};        \
template<> template<> struct NTraits<float>::Result<T_>                        \
  {typedef T_ Mul; typedef Mul Dvd; typedef Mul Add; typedef Mul Sub;};        \
template<> template<> struct NTraits<double>::Result<T_>                       \
  {typedef T_ Mul; typedef Mul Dvd; typedef Mul Add; typedef Mul Sub;};        \
                                                                               \
template <> class CNT<T_> : public NTraits<T_> { };                            \
                                                                               \
inline T_ square(const negator<T_>& x) {return square(-x);}                    \
inline T_ cube(const negator<T_>& x)   {return -cube(-x);}                     \
inline bool isNaN(const negator<T_>& x)    {return isNaN(-x);}                 \
inline bool isFinite(const negator<T_>& x) {return isFinite(-x);}              \
inline bool isInf(const negator<T_>& x)    {return isInf(-x);}                 \
                                                                               \
template <> struct Widest<T_,T_>                                               \
  {typedef T_ Type; typedef T_ Precision;};                                    \
template <> struct Widest<T_,double>                                           \
  {typedef T_ Type; typedef T_ Precision;};                                    \
template <> struct Widest<double,T_>                                           \
  {typedef T_ Type; typedef T_ Precision;};                                    \
template <> struct Widest<T_,float>                                            \
  {typedef T_ Type; typedef T_ Precision;};                                    \
template <> struct Widest<float,T_>                                            \
  {typedef T_ Type; typedef T_ Precision;};                                    \
                                                                               \
SimTK_SPECIALIZE_FLOATING_TYPE(T_);                                            \
} /* namespace SimTK */                                                        \
static_assert(true, "") // require a semicolon

// Helper for the operators below: one set of the global operators between a
// scalar T and a small matrix class CLS with template parameters TPARAMS.
// The parameter lists are parenthesized so that they may contain commas.
#define SimTK_REAL_SCALAR_UNPAREN_(...) __VA_ARGS__
#define SimTK_REAL_SCALAR_OPS_(T_, TPARAMS, CLS)                               \
template <SimTK_REAL_SCALAR_UNPAREN_ TPARAMS> inline                           \
typename SimTK_REAL_SCALAR_UNPAREN_ CLS::template Result<T_>::Mul              \
operator*(const SimTK_REAL_SCALAR_UNPAREN_ CLS& l, const T_& r)                \
{   return SimTK_REAL_SCALAR_UNPAREN_ CLS::template Result<T_>::MulOp          \
                                                        ::perform(l,r); }      \
template <SimTK_REAL_SCALAR_UNPAREN_ TPARAMS> inline                           \
typename SimTK_REAL_SCALAR_UNPAREN_ CLS::template Result<T_>::Mul              \
operator*(const T_& l, const SimTK_REAL_SCALAR_UNPAREN_ CLS& r) {return r*l;}  \
template <SimTK_REAL_SCALAR_UNPAREN_ TPARAMS> inline                           \
typename SimTK_REAL_SCALAR_UNPAREN_ CLS::template Result<T_>::Dvd              \
operator/(const SimTK_REAL_SCALAR_UNPAREN_ CLS& l, const T_& r)                \
{   return SimTK_REAL_SCALAR_UNPAREN_ CLS::template Result<T_>::DvdOp          \
                                                        ::perform(l,r); }      \
template <SimTK_REAL_SCALAR_UNPAREN_ TPARAMS> inline                           \
typename SimTK_REAL_SCALAR_UNPAREN_ CLS::template Result<T_>::Add              \
operator+(const SimTK_REAL_SCALAR_UNPAREN_ CLS& l, const T_& r)                \
{   return SimTK_REAL_SCALAR_UNPAREN_ CLS::template Result<T_>::AddOp          \
                                                        ::perform(l,r); }      \
template <SimTK_REAL_SCALAR_UNPAREN_ TPARAMS> inline                           \
typename SimTK_REAL_SCALAR_UNPAREN_ CLS::template Result<T_>::Add              \
operator+(const T_& l, const SimTK_REAL_SCALAR_UNPAREN_ CLS& r) {return r+l;}  \
template <SimTK_REAL_SCALAR_UNPAREN_ TPARAMS> inline                           \
typename SimTK_REAL_SCALAR_UNPAREN_ CLS::template Result<T_>::Sub              \
operator-(const SimTK_REAL_SCALAR_UNPAREN_ CLS& l, const T_& r)                \
{   return SimTK_REAL_SCALAR_UNPAREN_ CLS::template Result<T_>::SubOp          \
                                                        ::perform(l,r); }      \
template <SimTK_REAL_SCALAR_UNPAREN_ TPARAMS> inline                           \
typename SimTK_REAL_SCALAR_UNPAREN_ CLS::template Result<T_>::Sub              \
operator-(const T_& l, const SimTK_REAL_SCALAR_UNPAREN_ CLS& r)                \
{   return -(r-l); }

/** Define the global operators (*, /, +, -) between a class-type real scalar
T and the small matrix classes. The small matrix classes provide these for
each of the built-in scalar types explicitly (see the comments in Vec.h), so
they must be provided for T as well. These mirror the "double" operators
exactly. Invoke this after including SimTKcommon/SmallMatrix.h; see
RealScalarType.h. **/
#define SimTK_DEFINE_REAL_SCALAR_MATRIX_OPERATORS(T_)                          \
namespace SimTK {                                                              \
SimTK_REAL_SCALAR_OPS_(T_, (int M, class E, int S), (Vec<M,E,S>))              \
SimTK_REAL_SCALAR_OPS_(T_, (int N, class E, int S), (Row<N,E,S>))              \
SimTK_REAL_SCALAR_OPS_(T_, (int M, class E, int S), (SymMat<M,E,S>))           \
SimTK_REAL_SCALAR_OPS_(T_, (int M, int N, class E, int CS, int RS),            \
                           (Mat<M,N,E,CS,RS>))                                 \
} /* namespace SimTK */                                                        \
static_assert(true, "") // require a semicolon

#endif // SimTK_SIMTKCOMMON_REAL_SCALAR_TYPE_H_
