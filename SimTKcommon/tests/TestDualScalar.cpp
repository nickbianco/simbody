/* -------------------------------------------------------------------------- *
 *                               Simbody(tm)                                  *
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

// Tests that the SimTK small matrix and mechanics classes work with a
// user-defined class-type scalar (see SimTKcommon/internal/RealScalarType.h),
// using the minimal forward-mode automatic differentiation type Dual below:
// the values computed with Dual must equal those computed with double, and
// the derivatives must agree with finite differences.

// The scalar traits for Dual must be defined before any other SimTKcommon
// header is included.
#include "SimTKcommon/Scalar.h"
#include "SimTKcommon/internal/RealScalarType.h"

#include <cmath>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

//==============================================================================
//                                   DUAL
//==============================================================================
// A minimal forward-mode automatic differentiation scalar ("dual number"),
// which carries a value and the derivative of that value along a single
// direction. Deliberately, Dual provides no comparison operators and no
// conversion to bool or double, like symbolic types such as casadi::SX. So any
// code that branches on a value will not compile with Dual.
namespace SimTKDualTest {

class Dual {
public:
    Dual() = default;
    Dual(double value, double deriv = 0) : v(value), d(deriv) {}

    double value() const {return v;}
    double deriv() const {return d;}

    Dual& operator+=(const Dual& b) {v += b.v; d += b.d; return *this;}
    Dual& operator-=(const Dual& b) {v -= b.v; d -= b.d; return *this;}
    Dual& operator*=(const Dual& b) {d = d*b.v + v*b.d; v *= b.v; return *this;}
    Dual& operator/=(const Dual& b)
    {   d = (d*b.v - v*b.d)/(b.v*b.v); v /= b.v; return *this; }

private:
    double v = 0, d = 0;
};

inline Dual operator-(const Dual& a) {return Dual(-a.value(), -a.deriv());}
inline Dual operator+(const Dual& a) {return a;}

#define SimTK_TEST_DUAL_BINARY_OP(OP)                                          \
inline Dual operator OP(Dual a, const Dual& b) {return a OP##= b;}             \
inline Dual operator OP(Dual a, double b)      {return a OP##= Dual(b);}       \
inline Dual operator OP(double a, const Dual& b) {return Dual(a) OP##= b;}     \
inline Dual operator OP(Dual a, int b)         {return a OP##= Dual(b);}       \
inline Dual operator OP(int a, const Dual& b)  {return Dual(a) OP##= b;}
SimTK_TEST_DUAL_BINARY_OP(+)
SimTK_TEST_DUAL_BINARY_OP(-)
SimTK_TEST_DUAL_BINARY_OP(*)
SimTK_TEST_DUAL_BINARY_OP(/)
#undef SimTK_TEST_DUAL_BINARY_OP

// Math functions, found by argument-dependent lookup.
inline Dual sin(const Dual& a)
{   return Dual(std::sin(a.value()),  std::cos(a.value())*a.deriv()); }
inline Dual cos(const Dual& a)
{   return Dual(std::cos(a.value()), -std::sin(a.value())*a.deriv()); }
inline Dual tan(const Dual& a) {
    const double t = std::tan(a.value());
    return Dual(t, (1 + t*t)*a.deriv());
}
inline Dual asin(const Dual& a) {
    return Dual(std::asin(a.value()),
                a.deriv()/std::sqrt(1 - a.value()*a.value()));
}
inline Dual acos(const Dual& a) {
    return Dual(std::acos(a.value()),
                -a.deriv()/std::sqrt(1 - a.value()*a.value()));
}
inline Dual atan(const Dual& a) {
    return Dual(std::atan(a.value()),
                a.deriv()/(1 + a.value()*a.value()));
}
inline Dual atan2(const Dual& y, const Dual& x) {
    const double r2 = x.value()*x.value() + y.value()*y.value();
    return Dual(std::atan2(y.value(), x.value()),
                (x.value()*y.deriv() - y.value()*x.deriv())/r2);
}
inline Dual sqrt(const Dual& a) {
    const double s = std::sqrt(a.value());
    return Dual(s, a.deriv()/(2*s));
}
inline Dual exp(const Dual& a) {
    const double e = std::exp(a.value());
    return Dual(e, e*a.deriv());
}
inline Dual log(const Dual& a)
{   return Dual(std::log(a.value()), a.deriv()/a.value()); }
inline Dual pow(const Dual& a, const Dual& b)
{   return exp(b*log(a)); }
inline Dual pow(const Dual& a, double b) {
    return Dual(std::pow(a.value(), b),
                b*std::pow(a.value(), b-1)*a.deriv());
}
inline Dual abs(const Dual& a)
{   return a.value() < 0 ? -a : a; }
inline Dual fabs(const Dual& a) {return abs(a);}

inline std::ostream& operator<<(std::ostream& o, const Dual& a)
{   return o << a.value() << "+" << a.deriv() << "e"; }

} // namespace SimTKDualTest

namespace SimTK {
inline bool isNaN(const SimTKDualTest::Dual& x)
{   return std::isnan(x.value()) || std::isnan(x.deriv()); }
inline bool isFinite(const SimTKDualTest::Dual& x)
{   return std::isfinite(x.value()) && std::isfinite(x.deriv()); }
inline bool isInf(const SimTKDualTest::Dual& x)
{   return std::isinf(x.value()) || std::isinf(x.deriv()); }
inline bool isNumericallyEqual(const SimTKDualTest::Dual& a,
                               const SimTKDualTest::Dual& b, double tol) {
    return isNumericallyEqual(a.value(), b.value(), tol)
        && isNumericallyEqual(a.deriv(), b.deriv(), tol);
}
} // namespace SimTK

SimTK_DEFINE_REAL_SCALAR_TRAITS(SimTKDualTest::Dual);

#include "SimTKcommon/SmallMatrix.h"

SimTK_DEFINE_REAL_SCALAR_MATRIX_OPERATORS(SimTKDualTest::Dual);

#include "SimTKcommon.h"
#include "SimTKcommon/Testing.h"

using namespace SimTK;
using SimTKDualTest::Dual;

//==============================================================================
//                               COMPUTATIONS
//==============================================================================
// Computations with the SimTK small matrix and mechanics classes, written as
// function templates of the scalar type P. Each Computation<P> has a static
// compute() that maps a vector of inputs to a vector of outputs.

// Append the elements of SimTK objects to a list of outputs.
template <class P>
void append(std::vector<P>& out, const P& x) { out.push_back(x); }
template <class P, int M, int S>
void append(std::vector<P>& out, const Vec<M,P,S>& v)
{   for (int i=0; i < M; ++i) out.push_back(v[i]); }
template <class P, int M, int N, int CS, int RS>
void append(std::vector<P>& out, const Mat<M,N,P,CS,RS>& m)
{   for (int i=0; i < M; ++i) for (int j=0; j < N; ++j) out.push_back(m(i,j)); }
template <class P>
void append(std::vector<P>& out, const SpatialVec_<P>& v)
{   append(out, v[0]); append(out, v[1]); }
template <class P>
void append(std::vector<P>& out, const Rotation_<P>& R)
{   append(out, R.asMat33()); }
template <class P>
void append(std::vector<P>& out, const Transform_<P>& X)
{   append(out, X.R()); append(out, X.p()); }

template <class P>
struct SmallMatrixOps {
    static std::vector<P> compute(const std::vector<P>& x) {
        const Vec<3,P> a(x[0], x[1], x[2]), b(x[3], x[4], x[5]);
        const Mat<3,3,P> M(x[0], x[1], x[2],
                           x[3], x[4], x[5],
                           x[6], x[7], x[8]);
        const SymMat<3,P> S(x[0],
                            x[1], x[4],
                            x[2], x[5], x[8]);
        const Row<3,P> r = ~b;
        const P s = x[9];

        std::vector<P> out;
        append(out, a + b);
        append(out, a - b);
        append(out, Vec<3,P>(-a));
        append(out, a % b);
        append(out, P(~a * b));
        append(out, dot(a, b));
        append(out, a.normSqr());
        append(out, a.norm());
        append(out, s*a);
        append(out, a*s);
        append(out, a/s);
        append(out, a + s);
        append(out, a - s);
        append(out, 2*a + 0.5*b - Vec3(1,2,3));
        append(out, M*a);
        append(out, ~M*a);
        append(out, M*M);
        append(out, ~M);
        append(out, s*M - M/s);
        append(out, crossMat(a));
        append(out, crossMat(a)*b);
        append(out, Mat<3,3,P>(S)*a);
        append(out, S*a);
        append(out, Vec<3,P>(~(r*M)));
        append(out, Mat<3,3,P>(a*r));
        append(out, sin(s) + cos(s) + sqrt(s*s + 1) + atan2(x[0], x[1]));
        append(out, Mat<3,3,P>(1)*a + Mat33(2)*b);
        return out;
    }
};

template <class P>
struct RotationOps {
    static std::vector<P> compute(const std::vector<P>& x) {
        Rotation_<P> Rx, Ry, Rz;
        Rx.setRotationFromAngleAboutX(x[0]);
        Ry.setRotationFromAngleAboutY(x[1]);
        Rz.setRotationFromAngleAboutZ(x[2]);
        const Rotation_<P> R = Rx * Ry * Rz;
        const Vec<3,P> v(x[3], x[4], x[5]);

        std::vector<P> out;
        append(out, R);
        append(out, Rotation_<P>(x[1], YAxis));
        append(out, R * v);
        append(out, ~R * v);
        append(out, Rotation_<P>(~R * Ry));
        append(out, Rotation_<P>(Rx * ~Rz));
        append(out, R.x() + R.y() + R.z());
        append(out, Vec<3,P>(~R.row(1)));
        return out;
    }
};

template <class P>
struct TransformAndSpatialOps {
    static std::vector<P> compute(const std::vector<P>& x) {
        const Rotation_<P> R1(x[0], XAxis), R2(x[1], ZAxis);
        const Vec<3,P> p1(x[2], x[3], x[4]), p2(x[5], x[6], x[7]);
        const Transform_<P> X_GA(R1, p1), X_GB(R2, p2);
        const SpatialVec_<P> V_GA(Vec<3,P>(x[8], x[9], x[10]),
                                  Vec<3,P>(x[11], x[0], x[1]));
        const SpatialVec_<P> V_GB(Vec<3,P>(x[2], x[3], x[4]),
                                  Vec<3,P>(x[5], x[6], x[7]));
        const SpatialVec_<P> A_GA(Vec<3,P>(x[3], x[7], x[1]),
                                  Vec<3,P>(x[9], x[2], x[5]));
        const SpatialVec_<P> A_GB(Vec<3,P>(x[4], x[8], x[0]),
                                  Vec<3,P>(x[10], x[11], x[6]));

        std::vector<P> out;
        append(out, X_GA * X_GB);
        append(out, Transform_<P>(~X_GA));
        append(out, Transform_<P>(~X_GA * X_GB));
        append(out, X_GA * p2);
        append(out, X_GA.xformFrameVecToBase(p2));
        append(out, X_GA.shiftBaseStationToFrame(p2));
        append(out, findRelativeVelocity(X_GA, V_GA, X_GB, V_GB));
        append(out, findRelativeAcceleration(X_GA, V_GA, A_GA,
                                             X_GB, V_GB, A_GB));
        append(out, reverseRelativeVelocity(X_GA, V_GA));
        append(out, shiftVelocityBy(V_GA, p2));
        append(out, shiftVelocityFromTo(V_GA, p1, p2));
        append(out, shiftAccelerationBy(A_GA, V_GA[0], p1));
        append(out, shiftForceBy(V_GB, p1));
        append(out, shiftForceFromTo(V_GB, p1, p2));
        const SpatialMat_<P> M(crossMat(p1), Mat<3,3,P>(x[0]),
                               Mat<3,3,P>(x[1]), crossMat(p2));
        append(out, M * V_GA);
        append(out, P(~V_GA * V_GB));
        append(out, R1 * V_GA);
        return out;
    }
};

// Mass properties, articulated inertias, and the Phi shift matrix, as used by
// Simbody's multibody tree computations.
template <class P>
struct MassPropertiesOps {
    static void appendSpatialMat(std::vector<P>& out, const SpatialMat_<P>& M)
    {   for (int i=0; i < 2; ++i) for (int j=0; j < 2; ++j) append(out, M(i,j)); }

    static std::vector<P> compute(const std::vector<P>& x) {
        const Rotation_<P> R(x[0], XAxis);
        const Vec<3,P> com(x[1], x[2], x[3]), s(x[4], x[5], x[6]);
        const UnitInertia_<P> G(Vec<3,P>(2+x[7]*x[7], 2+x[8]*x[8], 3), 
                                Vec<3,P>(0.1, 0.2, 0.3)*x[9]);
        const SpatialInertia_<P> M(1 + x[8]*x[8], com, G);
        const SpatialVec_<P> V(Vec<3,P>(x[7], x[8], x[9]), s);
        const PhiMatrix_<P> phi(s);

        std::vector<P> out;
        append(out, Mat<3,3,P>(G.reexpress(R).asSymMat33()));
        append(out, Mat<3,3,P>(G.reexpress(~R).asSymMat33()));
        append(out, Mat<3,3,P>(crossMatSq(com)));
        append(out, M * V);
        append(out, M.shift(s) * V);
        SpatialInertia_<P> Msum(M); Msum += M.shift(com);
        append(out, Msum * V);
        const ArticulatedInertia_<P> A(M);
        appendSpatialMat(out, A.toSpatialMat());
        appendSpatialMat(out, A.shift(s).toSpatialMat());
        appendSpatialMat(out, (A + A.shift(com) - A).toSpatialMat());
        append(out, A * V);
        append(out, phi * V);
        append(out, ~phi * V);
        appendSpatialMat(out, phi * A.toSpatialMat() * ~phi);
        appendSpatialMat(out, phi.toSpatialMat());
        return out;
    }
};

//==============================================================================
//                                  TESTS
//==============================================================================
// Dual is deliberately not comparable, so that value-dependent branching in
// code that must work with such types is caught at compile time.
template <class T, class = void>
struct IsComparable : std::false_type {};
template <class T>
struct IsComparable<T, decltype(void(std::declval<T>() < std::declval<T>()))>
:   std::true_type {};
static_assert(!IsComparable<Dual>::value, "Dual should not be comparable.");
static_assert(!std::is_convertible<Dual, bool>::value,
              "Dual should not convert to bool.");

// Evaluate Computation::compute() with doubles and with Duals at random
// inputs x0, seeding the Dual derivatives with a random direction dx. The
// values must match, and the derivatives must match central differences
// of the double computation along dx.
template <template <class> class Computation>
void compareDoubleAndDual(int numInputs) {
    std::vector<double> x0(numInputs), dx(numInputs);
    for (auto& x : x0) x = Test::randReal();
    for (auto& x : dx) x = Test::randReal();

    std::vector<Dual> xDual(numInputs);
    for (int i=0; i < numInputs; ++i) xDual[i] = Dual(x0[i], dx[i]);
    const std::vector<Dual> yDual = Computation<Dual>::compute(xDual);

    const std::vector<double> y = Computation<double>::compute(x0);
    const double h = 1e-6;
    std::vector<double> xp(x0), xm(x0);
    for (int i=0; i < numInputs; ++i) {xp[i] += h*dx[i]; xm[i] -= h*dx[i];}
    const std::vector<double> yp = Computation<double>::compute(xp);
    const std::vector<double> ym = Computation<double>::compute(xm);

    SimTK_TEST(yDual.size() == y.size());
    for (size_t i=0; i < y.size(); ++i) {
        SimTK_TEST_EQ(yDual[i].value(), y[i]);
        SimTK_TEST_EQ_TOL(yDual[i].deriv(), (yp[i] - ym[i])/(2*h), 1e-6);
    }
}

void testSmallMatrix() {compareDoubleAndDual<SmallMatrixOps>(10);}
void testRotation()    {compareDoubleAndDual<RotationOps>(6);}
void testTransformAndSpatialAlgebra()
{   compareDoubleAndDual<TransformAndSpatialOps>(12); }
void testMassProperties() {compareDoubleAndDual<MassPropertiesOps>(10);}

void testScalarQueries() {
    const Dual x(2, 1);
    SimTK_TEST(!isNaN(x));
    SimTK_TEST(isFinite(x));
    SimTK_TEST(!isInf(x));
    SimTK_TEST(isNaN(NTraits<Dual>::getNaN()));
    SimTK_TEST(isInf(-NTraits<Dual>::getInfinity()));
    SimTK_TEST(isNaN(-negator<Dual>::recast(NTraits<Dual>::getNaN())));
    SimTK_TEST(isNumericallyEqual(Dual(1.), 1.));
    SimTK_TEST(!isNumericallyEqual(x, Dual(2, 1.1)));
    SimTK_TEST((Vec<3,Dual>(x, 1, 2).isNumericallyEqual(Vec<3,Dual>(x, 1, 2))));
    SimTK_TEST_EQ(NTraits<Dual>::getPi().value(), Pi);
    SimTK_TEST_EQ(NTraits<Dual>::sqrt(x).deriv(), 1/(2*std::sqrt(2.)));
    SimTK_TEST_EQ(square(x).deriv(), 4.);
}

// A ScalarState copies a State's time, continuous variables, and their
// layout, and tracks its stage and cache entries like a State.
void testState() {
    // A State with two subsystems, as a System would create it.
    const SubsystemIndex sub0(0), sub1(1);
    State s;
    s.setNumSubsystems(2);
    s.initializeSubsystem(sub0, "sub0", "1");
    s.initializeSubsystem(sub1, "sub1", "1");
    for (const Stage g : {Stage::Topology, Stage::Model, Stage::Instance}) {
        if (g == Stage::Model) {
            s.allocateQ(sub1, Vector(Vec2(1, 2)));
            s.allocateU(sub1, Vector(Vec2(3, 4)));
            s.allocateZ(sub0, Vector(Vec3(5, 6, 7)));
            s.allocateQ(sub0, Vector(1, 8.));
        }
        s.advanceSubsystemToStage(sub0, g);
        s.advanceSubsystemToStage(sub1, g);
        s.advanceSystemToStage(g);
    }
    s.setTime(0.5);

    ScalarState<Dual> sx(s);
    SimTK_TEST(sx.getSystemStage() == Stage::Model);
    SimTK_TEST(sx.getNumSubsystems() == 2);
    SimTK_TEST(sx.getTime().value() == 0.5);
    SimTK_TEST(sx.getNQ() == 3 && sx.getNU() == 2 && sx.getNZ() == 3);
    for (const SubsystemIndex i : {sub0, sub1}) {
        SimTK_TEST(sx.getQStart(i) == s.getQStart(i));
        SimTK_TEST(sx.getNQ(i) == s.getNQ(i));
        SimTK_TEST(sx.getUStart(i) == s.getUStart(i));
        SimTK_TEST(sx.getNU(i) == s.getNU(i));
        SimTK_TEST(sx.getZStart(i) == s.getZStart(i));
        SimTK_TEST(sx.getNZ(i) == s.getNZ(i));
    }
    for (int i=0; i < 3; ++i) {
        SimTK_TEST(sx.getQ()[i].value() == s.getQ()[i]);
        SimTK_TEST(sx.getZ()[i].value() == s.getZ()[i]);
    }
    for (int i=0; i < 2; ++i) SimTK_TEST(sx.getU()[i].value() == s.getU()[i]);
    SimTK_TEST(sx.getRealState().getNQ() == 3);

    // Stages: advancing one at a time, and invalidation by the variables.
    SimTK_TEST_MUST_THROW(sx.advanceSystemToStage(Stage::Position));
    for (const Stage g : {Stage::Instance, Stage::Time, Stage::Position,
                          Stage::Velocity, Stage::Dynamics})
        sx.advanceSystemToStage(g);
    sx.updZ();
    SimTK_TEST(sx.getSystemStage() == Stage::Velocity);
    sx.updU();
    SimTK_TEST(sx.getSystemStage() == Stage::Position);
    sx.updQ();
    SimTK_TEST(sx.getSystemStage() == Stage::Time);
    sx.setTime(Dual(1, 1));
    SimTK_TEST(sx.getSystemStage() == Stage::Instance);
    SimTK_TEST_MUST_THROW(sx.setQ(std::vector<Dual>(2)));
    sx.invalidateAll(Stage::Instance);
    SimTK_TEST(sx.getSystemStage() == Stage::Model);

    // Cache entries are valid at and above their stage, and copies are
    // independent.
    const CacheEntryIndex e = 
        sx.allocateCacheEntry(sub1, Stage::Position, new Value<Dual>(Dual(2)));
    SimTK_TEST(sx.getNumCacheEntries(sub1) == 1);
    SimTK_TEST(sx.getNumCacheEntries(sub0) == 0);
    SimTK_TEST_MUST_THROW(sx.getCacheEntry(sub1, e));
    Value<Dual>::updDowncast(sx.updCacheEntry(sub1, e)) = Dual(3, 1);
    for (const Stage g : {Stage::Instance, Stage::Time, Stage::Position})
        sx.advanceSystemToStage(g);
    SimTK_TEST(sx.getCacheEntry(sub1, e).getValue<Dual>().value() == 3);
    ScalarState<Dual> copy(sx);
    Value<Dual>::updDowncast(copy.updCacheEntry(sub1, e)) = Dual(4);
    copy.updQ()[0] = Dual(9);
    SimTK_TEST(sx.getCacheEntry(sub1, e).getValue<Dual>().value() == 3);
    SimTK_TEST(sx.getQ()[0].value() == 8);
    SimTK_TEST(sx.getSystemStage() == Stage::Position);
    SimTK_TEST(copy.getSystemStage() == Stage::Time);
    SimTK_TEST_MUST_THROW(sx.getCacheEntry(sub0, e));

    // It must be created from a State realized through Instance.
    State empty;
    SimTK_TEST_MUST_THROW(ScalarState<Dual> bad(empty));
}

int main() {
    SimTK_START_TEST("TestDualScalar");
        SimTK_SUBTEST(testSmallMatrix);
        SimTK_SUBTEST(testRotation);
        SimTK_SUBTEST(testTransformAndSpatialAlgebra);
        SimTK_SUBTEST(testMassProperties);
        SimTK_SUBTEST(testScalarQueries);
        SimTK_SUBTEST(testState);
    SimTK_END_TEST();
}
