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

// Tests that a ScalarState<T>, with which Simbody's own RigidBodyNode
// computations are evaluated with a user-defined scalar type T, works for the
// minimal forward-mode automatic differentiation type Dual below (the same as
// in SimTKcommon's TestDualScalar.cpp), without depending on any external
// library. The values must equal Simbody's, and the derivatives along a
// random direction must match central differences of Simbody's results, for
// body kinematics, forward and inverse dynamics, mass matrix products, system
// Jacobian products, and kinetic energy, and, for systems with enabled
// constraints, the constraint errors and the constraint forces for given
// multipliers.

// The scalar traits for Dual must be defined before any other SimTKcommon
// header is included.
#include "SimTKcommon/Scalar.h"
#include "SimTKcommon/internal/RealScalarType.h"

#include <cmath>
#include <iostream>
#include <string>
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

#include "Simbody.h"

// Evaluate a (Real) Function at Dual arguments with the chain rule, for
// CoordinateCoupler constraints.
namespace SimTK {
template <> struct FunctionEvaluator<SimTKDualTest::Dual> {
    using Dual = SimTKDualTest::Dual;
    static Vector values(const std::vector<Dual>& x) {
        Vector v((int)x.size());
        for (int i=0; i < (int)x.size(); ++i) v[i] = x[i].value();
        return v;
    }
    // The derivative of the given partial derivative along x's direction.
    static Real directional(const Function& f, std::vector<int> components,
                            const Vector& v, const std::vector<Dual>& x) {
        components.push_back(0);
        Real d = 0;
        for (int i=0; i < (int)x.size(); ++i) {
            components.back() = i;
            d += f.calcDerivative(components, v) * x[i].deriv();
        }
        return d;
    }
    static Dual calcValue(const Function& f, const std::vector<Dual>& x) {
        const Vector v = values(x);
        return Dual(f.calcValue(v), directional(f, {}, v, x));
    }
    static Dual calcDerivative(const Function& f,
                               const std::vector<int>& derivComponents,
                               const std::vector<Dual>& x) {
        const Vector v = values(x);
        return Dual(f.calcDerivative(derivComponents, v),
                    directional(f, derivComponents, v, x));
    }
};
} // namespace SimTK

#include "simbody/internal/tree/MultibodyScalarImpl.h"
#include "SimTKcommon/Testing.h"

using namespace SimTK;
using SimTKDualTest::Dual;

// Instantiate everything so we catch compilation problems.
SimTK_INSTANTIATE_MULTIBODY_SCALAR(Dual);

namespace {

const Vec3 Gravity(0.3, -9.8, 0.5);

MassProperties randMassProperties() {
    const Real  mass = 1 + std::abs(Test::randReal());
    const Vec3  com  = Test::randVec3();
    const Vec3  halfLengths = Vec3(1.5) + Test::randVec3();
    return MassProperties(mass, com,
        UnitInertia::brick(halfLengths).shiftFromCentroid(com));
}

// A multibody system with pin, slider, planar, and weld mobilizers (some
// reversed) connected with random inboard and outboard frames. parents[i] is
// the index of the parent of body i+1.
// Add a mobilized body with the named mobilizer type to parent.
void addMobilizedBody(const std::string& kind, MobilizedBody& parent,
                      const Transform& X_PF, const Body& body,
                      const Transform& X_BM, MobilizedBody::Direction dir) {
    if      (kind == "Pin")       MobilizedBody::Pin(parent, X_PF, body, X_BM, dir);
    else if (kind == "Slider")    MobilizedBody::Slider(parent, X_PF, body, X_BM, dir);
    else if (kind == "Planar")    MobilizedBody::Planar(parent, X_PF, body, X_BM, dir);
    else if (kind == "Weld")      MobilizedBody::Weld(parent, X_PF, body, X_BM);
    else if (kind == "Gimbal")    MobilizedBody::Gimbal(parent, X_PF, body, X_BM, dir);
    else if (kind == "Ball")      MobilizedBody::Ball(parent, X_PF, body, X_BM, dir);
    else if (kind == "Universal") MobilizedBody::Universal(parent, X_PF, body, X_BM, dir);
    else if (kind == "Cylinder")  MobilizedBody::Cylinder(parent, X_PF, body, X_BM, dir);
    else if (kind == "Screw")     MobilizedBody::Screw(parent, X_PF, body, X_BM, 0.3, dir);
    else if (kind == "Translation")
        MobilizedBody::Translation(parent, X_PF, body, X_BM, dir);
    else if (kind == "BendStretch")
        MobilizedBody::BendStretch(parent, X_PF, body, X_BM, dir);
    else if (kind == "SphericalCoords")
        MobilizedBody::SphericalCoords(parent, X_PF, body, X_BM, 0.2, true,
                                       -0.4, false, XAxis, true, dir);
    else if (kind == "CantileverFreeBeam")
        MobilizedBody::CantileverFreeBeam(parent, X_PF, body, X_BM, 1.7, dir);
    else if (kind == "Ellipsoid")
        MobilizedBody::Ellipsoid(parent, X_PF, body, X_BM,
                                 Vec3(0.6, 0.4, 0.3), dir);
    else if (kind == "EllipsoidMirrored")
        MobilizedBody::Ellipsoid(parent, X_PF, body, X_BM,
                                 Vec3(0.5, 0.7, 0.2), YAxis, dir);
    else if (kind == "LineOrientation")
        MobilizedBody::LineOrientation(parent, X_PF, body, X_BM, dir);
    else if (kind == "Free")      MobilizedBody::Free(parent, X_PF, body, X_BM, dir);
    else if (kind == "FreeLine")  MobilizedBody::FreeLine(parent, X_PF, body, X_BM, dir);
    else if (kind == "Bushing")   MobilizedBody::Bushing(parent, X_PF, body, X_BM, dir);
    else if (kind == "FunctionBased") {
        // Two mobilities, with the default axes; constant functions (of no
        // q's) are allowed.
        Vector poly(3); poly[0] = 0.4; poly[1] = -0.3; poly[2] = 0.2;
        Vector lin2(3); lin2[0] = 0.5; lin2[1] = -0.7; lin2[2] = 0.1;
        Vector lin1(2); lin1[0] = 1.2; lin1[1] = -0.05;
        const std::vector<const Function*> functions{
            new Function::Sinusoid(0.8, 1.3, 0.2), new Function::Polynomial(poly),
            new Function::Constant(0.3, 0), new Function::Linear(lin1),
            new Function::Constant(-0.2, 0), new Function::Linear(lin2)};
        const std::vector<std::vector<int>> coordIndices{
            {0}, {1}, {}, {0}, {}, {0, 1}};
        MobilizedBody::FunctionBased(parent, X_PF, body, X_BM, 2, functions,
                                     coordIndices, dir);
    } else if (kind == "FunctionBasedAxes") {
        // Three mobilities, with non-orthogonal axes; a rotation is a
        // function of two q's.
        Vector poly(3); poly[0] = -0.6; poly[1] = 0.9; poly[2] = 0.1;
        Vector lin2(3); lin2[0] = 0.6; lin2[1] = 0.3; lin2[2] = -0.1;
        Vector lin1(2); lin1[0] = 0.7; lin1[1] = 0.2;
        const std::vector<const Function*> functions{
            new Function::Linear(lin2), new Function::Polynomial(poly),
            new Function::Sinusoid(1.1, 0.7, -0.3), new Function::Linear(lin1),
            new Function::Sinusoid(0.5, 2.0, 0.4), new Function::Polynomial(poly)};
        const std::vector<std::vector<int>> coordIndices{
            {0, 2}, {1}, {2}, {1}, {0}, {2}};
        const std::vector<Vec3> axes{Vec3(1, 0.2, 0), Vec3(0.1, 1, 0.3),
            Vec3(0, -0.2, 1), Vec3(1, 0, 0.4), Vec3(0.3, 1, 0), Vec3(0, 0.1, 1)};
        MobilizedBody::FunctionBased(parent, X_PF, body, X_BM, 3, functions,
                                     coordIndices, axes, dir);
    }
    else if (kind == "FunctionBasedConstantRotation") {
        // A fixed 90 degree rotation about x, then a rotation about y by q0:
        // the second axis is z in F (the H_FM of a constant rotation).
        const std::vector<const Function*> functions{
            new Function::Constant(Pi/2, 0), new Function::Linear(Vector(Vec2(1, 0))),
            new Function::Constant(0, 0), new Function::Constant(0, 0),
            new Function::Constant(0, 0), new Function::Constant(0, 0)};
        const std::vector<std::vector<int>> coordIndices{
            {}, {0}, {}, {}, {}, {}};
        MobilizedBody::FunctionBased(parent, X_PF, body, X_BM, 1, functions,
                                     coordIndices, dir);
    } else if (kind == "FunctionBasedSwappedCoordinates") {
        // A rotation about x by q1, then about y by q0 (the HDot_FM of
        // rotations that don't depend on the q's in order).
        const std::vector<const Function*> functions{
            new Function::Linear(Vector(Vec2(1, 0))),
            new Function::Linear(Vector(Vec2(1, 0))),
            new Function::Constant(0, 0), new Function::Constant(0, 0),
            new Function::Constant(0, 0), new Function::Constant(0, 0)};
        const std::vector<std::vector<int>> coordIndices{
            {1}, {0}, {}, {}, {}, {}};
        MobilizedBody::FunctionBased(parent, X_PF, body, X_BM, 2, functions,
                                     coordIndices, dir);
    }
    else SimTK_ASSERT1_ALWAYS(false, "Unknown mobilizer type %s.", kind.c_str());
}

// The mobilizer types with scalar-type support, each of which is tested both
// forward and reversed.
const std::vector<std::string> SupportedMobilizers{
    "Pin", "Slider", "Planar", "Weld", "Gimbal", "Universal", "Cylinder",
    "Screw", "Translation", "BendStretch", "SphericalCoords",
    "CantileverFreeBeam", "Ball", "Ellipsoid", "EllipsoidMirrored",
    "LineOrientation", "Free", "FreeLine", "Bushing", "FunctionBased",
    "FunctionBasedAxes", "FunctionBasedConstantRotation",
    "FunctionBasedSwappedCoordinates"};

struct TestSystem {
    MultibodySystem           system;
    SimbodyMatterSubsystem    matter;
    GeneralForceSubsystem     forces;
    Force::Gravity            gravity;
    Force::DiscreteForces     discreteForces;
    State                     state;

    // Body i+1 has parent parents[i] and, if kinds is given, mobilizer type
    // kinds[i].
    explicit TestSystem(const std::vector<int>& parents,
                        const std::vector<std::string>& kinds = {},
                        bool useEulerAngles = false)
    :   matter(system), forces(system),
        gravity(forces, matter, Gravity),
        discreteForces(forces, matter) {
        for (int i=0; i < (int)parents.size(); ++i) {
            MobilizedBody& parent =
                matter.updMobilizedBody(MobilizedBodyIndex(parents[i]));
            const Body::Rigid body(randMassProperties());
            const Transform X_PF = Test::randTransform();
            const Transform X_BM = Test::randTransform();
            if (!kinds.empty()) {
                addMobilizedBody(kinds[i], parent, X_PF, body, X_BM,
                    (i % 2) ? MobilizedBody::Reverse : MobilizedBody::Forward);
                continue;
            }
            const MobilizedBody::Direction dir = (i % 4 == 1)
                ? MobilizedBody::Reverse : MobilizedBody::Forward;
            if      (i == 3 || i == 5)
                MobilizedBody::Planar(parent, X_PF, body, X_BM, dir);
            else if (i % 5 == 4) MobilizedBody::Weld  (parent, X_PF, body, X_BM);
            else if (i % 3 == 2) MobilizedBody::Slider(parent, X_PF, body, X_BM, dir);
            else                 MobilizedBody::Pin   (parent, X_PF, body, X_BM, dir);
        }
        state = system.realizeTopology();
        // Mobilizers that can use quaternions use Euler angles instead.
        matter.setUseEulerAngles(state, useEulerAngles);
        system.realizeModel(state);
        // Ellipsoid radii are Instance-stage state variables; change them
        // from their defaults.
        for (MobilizedBodyIndex b(1); b < matter.getNumBodies(); ++b) {
            const MobilizedBody& mobod = matter.getMobilizedBody(b);
            if (MobilizedBody::Ellipsoid::isInstanceOf(mobod)) {
                const auto& ellipsoid = MobilizedBody::Ellipsoid::downcast(mobod);
                ellipsoid.setRadii(state, 1.1*ellipsoid.getDefaultRadii());
            }
        }
        system.realize(state, Stage::Instance);
    }
};

// Gravity body forces for a ScalarState, computed as Force::Gravity does.
// Force subsystems don't support other scalar types, so applied forces are
// computed by the caller.
template <class T>
std::vector<SpatialVec_<T>> calcGravityForces(const TestSystem& sys,
                                              const ScalarState<T>& s) {
    const int nb = sys.matter.getNumBodies();
    std::vector<SpatialVec_<T>> F(nb, SpatialVec_<T>(Vec<3,T>(0), Vec<3,T>(0)));
    const Vec<3,T> g = Vec<3,T>(T(Gravity[0]), T(Gravity[1]), T(Gravity[2]));
    for (MobilizedBodyIndex b(1); b < nb; ++b) {
        const MobilizedBody& mobod = sys.matter.getMobilizedBody(b);
        const MassProperties& mp = mobod.getBodyMassProperties(sys.state);
        const Vec3& com = mp.getMassCenter();
        const Vec<3,T> p_BBc_G = mobod.getBodyTransform(s).R()
                               * Vec<3,T>(T(com[0]), T(com[1]), T(com[2]));
        const Vec<3,T> mg = T(mp.getMass()) * g;
        F[b] = SpatialVec_<T>(p_BBc_G % mg, mg);
    }
    return F;
}

Vector randVector(int n) {
    Vector v(n);
    for (int i=0; i < n; ++i) v[i] = Test::randReal();
    return v;
}
Vector_<SpatialVec> randSpatialVecs(int n) {
    Vector_<SpatialVec> v(n);
    for (int i=0; i < n; ++i) v[i] = Test::randSpatialVec();
    return v;
}

// Duals with the given values and derivatives.
std::vector<Dual> toDual(const Vector& v, const Vector& dv) {
    std::vector<Dual> d(v.size());
    for (int i=0; i < v.size(); ++i) d[i] = Dual(v[i], dv[i]);
    return d;
}
std::vector<SpatialVec_<Dual>> toDual(const Vector_<SpatialVec>& v,
                                      const Vector_<SpatialVec>& dv) {
    std::vector<SpatialVec_<Dual>> d(v.size());
    for (int i=0; i < v.size(); ++i)
        for (int k=0; k < 2; ++k) for (int j=0; j < 3; ++j)
            d[i][k][j] = Dual(v[i][k][j], dv[i][k][j]);
    return d;
}

// Flatten scalars, vectors, body poses, and spatial vectors.
template <class P>
void append(std::vector<P>& out, const P& x) {out.push_back(x);}
template <class P>
void append(std::vector<P>& out, const std::vector<P>& v)
{   out.insert(out.end(), v.begin(), v.end()); }
template <class P>
void append(std::vector<P>& out, const Transform_<P>& X) {
    for (int i=0; i < 3; ++i) for (int j=0; j < 3; ++j)
        out.push_back(X.R()[i][j]);
    for (int i=0; i < 3; ++i) out.push_back(X.p()[i]);
}
template <class P>
void append(std::vector<P>& out, const SpatialVec_<P>& V) {
    for (int k=0; k < 2; ++k) for (int i=0; i < 3; ++i)
        out.push_back(V[k][i]);
}
template <class P>
void append(std::vector<P>& out, const std::vector<SpatialVec_<P>>& V)
{   for (const auto& v : V) append(out, v); }
void append(std::vector<Real>& out, const Vector& v)
{   for (int i=0; i < v.size(); ++i) out.push_back(v[i]); }
void append(std::vector<Real>& out, const Vector_<SpatialVec>& V)
{   for (int i=0; i < V.size(); ++i) append(out, V[i]); }

void compareDualToSimbody(const std::vector<int>& parents,
                          const std::vector<std::string>& kinds = {},
                          bool useEulerAngles = false) {
    TestSystem sys(parents, kinds, useEulerAngles);
    const int nq = sys.state.getNQ(), nu = sys.state.getNU();
    const int nb = sys.matter.getNumBodies();

    // Simbody's results at the given inputs, flattened in the same order as
    // the ScalarState results below.
    auto simbody = [&](const Vector& q, const Vector& u, const Vector& f,
                       const Vector& knownUDot, const Vector& v,
                       const Vector_<SpatialVec>& F) {
        State s = sys.state;
        s.updQ() = q; s.updU() = u;
        sys.discreteForces.setAllMobilityForces(s, f);
        sys.system.realize(s, Stage::Acceleration);
        std::vector<Real> out;
        for (MobilizedBodyIndex b(0); b < nb; ++b) {
            const MobilizedBody& mobod = sys.matter.getMobilizedBody(b);
            append(out, mobod.getBodyTransform(s));
            append(out, mobod.getBodyVelocity(s));
            append(out, mobod.getBodyAcceleration(s));
        }
        append(out, s.getUDot());
        append(out, s.getQErr());
        Vector residual, Mv, MinvV, JtF;
        Vector_<SpatialVec> Jv;
        sys.matter.calcResidualForceIgnoringConstraints(s,
            sys.system.getMobilityForces(s, Stage::Dynamics),
            sys.system.getRigidBodyForces(s, Stage::Dynamics),
            knownUDot, residual);
        sys.matter.multiplyByM(s, v, Mv);
        sys.matter.multiplyByMInv(s, v, MinvV);
        sys.matter.multiplyBySystemJacobian(s, v, Jv);
        sys.matter.multiplyBySystemJacobianTranspose(s, F, JtF);
        append(out, residual); append(out, Mv); append(out, MinvV);
        append(out, Jv); append(out, JtF);
        append(out, sys.matter.calcKineticEnergy(s));
        return out;
    };

    // The same results computed with Duals.
    ScalarState<Dual> sx = sys.system.createScalarState<Dual>(sys.state);
    SimTK_TEST(sx.getNQ() == nq && sx.getNU() == nu);
    auto dual = [&](const std::vector<Dual>& q, const std::vector<Dual>& u,
                    const std::vector<Dual>& f,
                    const std::vector<Dual>& knownUDot,
                    const std::vector<Dual>& v,
                    const std::vector<SpatialVec_<Dual>>& F) {
        sx.setQ(q); sx.setU(u);
        SimTK_TEST(sx.getSystemStage() <= Stage::Time);
        sys.system.realize(sx, Stage::Velocity);
        SimTK_TEST(sx.getSystemStage() == Stage::Velocity);
        const std::vector<SpatialVec_<Dual>> gravity = calcGravityForces(sys, sx);
        std::vector<Dual> udot, residual, Mv, MinvV, JtF;
        std::vector<SpatialVec_<Dual>> A_GB, Jv;
        sys.matter.calcAccelerationIgnoringConstraints(sx, f, gravity, 
                                                       udot, A_GB);
        std::vector<Dual> out;
        for (MobilizedBodyIndex b(0); b < nb; ++b) {
            const MobilizedBody& mobod = sys.matter.getMobilizedBody(b);
            append(out, mobod.getBodyTransform(sx));
            append(out, mobod.getBodyVelocity(sx));
            append(out, A_GB[b]);
        }
        append(out, udot);
        append(out, sx.getQErr());
        sys.matter.calcResidualForceIgnoringConstraints(sx, f, gravity,
                                                        knownUDot, residual);
        sys.matter.multiplyByM(sx, v, Mv);
        sys.matter.multiplyByMInv(sx, v, MinvV);
        sys.matter.multiplyBySystemJacobian(sx, v, Jv);
        sys.matter.multiplyBySystemJacobianTranspose(sx, F, JtF);
        append(out, residual); append(out, Mv); append(out, MinvV);
        append(out, Jv); append(out, JtF);
        append(out, sys.matter.calcKineticEnergy(sx));
        return out;
    };

    for (int trial=0; trial < 5; ++trial) {
        // A random point and a random direction for all the inputs.
        const Vector q = randVector(nq), u = randVector(nu),
                     f = 10*randVector(nu), knownUDot = randVector(nu),
                     v = randVector(nu);
        const Vector_<SpatialVec> F = randSpatialVecs(nb);
        const Vector dq = randVector(nq), du = randVector(nu),
                     df = randVector(nu), dknownUDot = randVector(nu),
                     dv = randVector(nu);
        const Vector_<SpatialVec> dF = randSpatialVecs(nb);

        const std::vector<Dual> y = dual(toDual(q, dq), toDual(u, du),
            toDual(f, df), toDual(knownUDot, dknownUDot), toDual(v, dv),
            toDual(F, dF));

        const Real h = 1e-6;
        const std::vector<Real> y0 = simbody(q, u, f, knownUDot, v, F);
        const std::vector<Real> yp = simbody(q + h*dq, u + h*du, f + h*df,
            knownUDot + h*dknownUDot, v + h*dv, F + h*dF);
        const std::vector<Real> ym = simbody(q - h*dq, u - h*du, f - h*df,
            knownUDot - h*dknownUDot, v - h*dv, F - h*dF);

        // The values match to roundoff, but not exactly: for mobilizers with
        // more than 3 dofs, other scalar types invert the articulated body
        // inertia's D matrix by Cholesky factorization instead of LAPACK.
        SimTK_TEST(y.size() == y0.size());
        for (size_t i=0; i < y0.size(); ++i) {
            SimTK_TEST_EQ_TOL(y[i].value(), y0[i], 1e-12);
            SimTK_TEST_EQ_TOL(y[i].deriv(), (yp[i] - ym[i])/(2*h), 1e-6);
        }
    }
}

// Body i+1 has parent parents[i].
const std::vector<int> Chain{0, 1, 2, 3, 4, 5, 6, 7};
const std::vector<int> Tree{0, 1, 1, 0, 4, 2, 2, 5, 3};

//==============================================================================
//                                CONSTRAINTS
//==============================================================================
// A coupling function like the one OpenSim's CoordinateCouplerConstraint
// creates, f(x) = s*g(x0, x1) - x2, with derivatives of all orders:
// g = a sin(b x0) + c x0 x1 + d x1^2.
class CouplerTestFunction : public Function {
public:
    Real calcValue(const Vector& x) const override {
        return a*std::sin(b*x[0]) + c*x[0]*x[1] + d*x[1]*x[1] - x[2];
    }
    Real calcDerivative(const Array_<int>& components,
                        const Vector& x) const override {
        int n[3] = {0, 0, 0};
        for (int i : components) ++n[i];
        if (n[2]) return n[2] == 1 && n[0] + n[1] == 0 ? -1 : 0;
        Real result = 0;
        if (n[1] == 0) // a sin(b x0)
            result += a*std::pow(b, n[0])*std::sin(b*x[0] + n[0]*Pi/2);
        if (n[0] <= 1 && n[1] <= 1) // c x0 x1
            result += c*(n[0] ? 1 : x[0])*(n[1] ? 1 : x[1]);
        if (n[0] == 0 && n[1] <= 2) // d x1^2
            result += n[1] == 0 ? d*x[1]*x[1] : n[1] == 1 ? 2*d*x[1] : 2*d;
        return result;
    }
    int getArgumentSize() const override {return 3;}
    int getMaxDerivativeOrder() const override {return 10;}
private:
    Real a = 0.7, b = 1.3, c = -0.4, d = 0.25;
};

// A tree of pin, slider, and planar bodies with each of the supported
// constraint types. Their Ancestors (the frames the constraint equations are
// written in) are Ground, one of the constrained bodies, and a body that isn't
// constrained. A CoordinateCoupler constrains q's instead of bodies. A
// disabled constraint of an unsupported type is ignored.
struct ConstraintTestSystem {
    MultibodySystem           system;
    SimbodyMatterSubsystem    matter;
    State                     state;

    ConstraintTestSystem() : matter(system) {
        const auto rigid = [] {return Body::Rigid(randMassProperties());};
        const auto randT = [] {return Test::randTransform();};
        const auto randP = [] {return Test::randVec3();};
        const auto randN = [] {return UnitVec3(Test::randVec3());};
        MobilizedBody::Pin    b1(matter.Ground(), randT(), rigid(), randT());
        MobilizedBody::Slider b2(b1, randT(), rigid(), randT());
        MobilizedBody::Planar b3(b1, randT(), rigid(), randT());
        MobilizedBody::Pin    b4(b2, randT(), rigid(), randT(),
                                 MobilizedBody::Reverse);
        MobilizedBody::Pin    b5(b3, randT(), rigid(), randT());
        MobilizedBody::Planar b6(matter.Ground(), randT(), rigid(), randT());
        MobilizedBody::Slider b7(b6, randT(), rigid(), randT(),
                                 MobilizedBody::Reverse);
        MobilizedBody::Pin    b8(b4, randT(), rigid(), randT());

        Constraint::Ball(b4, randP(), b5, randP());             // Ancestor b1
        Constraint::Weld(b1, randT(), b8, randT());             // b1
        Constraint::Rod(b7, randP(), b5, randP(), 1.5);         // Ground
        Constraint::PointInPlane(b3, randN(), 0.3, b5, randP());// b3
        Constraint::PointOnLine(b6, randN(), randP(), b7, randP()); // b6
        Constraint::ConstantAngle(b2, randN(), b3, randN(), 1); // b1
        Constraint::NoSlip1D(b1, randP(), randN(), b2, b3);     // b1
        Constraint::NoSlip1D(matter.Ground(), randP(), randN(),
                             b6, b1);                           // Ground
        // Couples one of the Planar's q's and two other mobilizers' q's.
        Constraint::CoordinateCoupler(matter, new CouplerTestFunction,
            std::vector<MobilizedBodyIndex>{b3, b2, b5},
            std::vector<MobilizerQIndex>{MobilizerQIndex(2), MobilizerQIndex(0),
                                         MobilizerQIndex(0)});
        Constraint::ConstantSpeed lock(b2, 0.);
        lock.setDisabledByDefault(true);

        state = system.realizeTopology();
        system.realize(state, Stage::Instance);
    }
};

// The constraint errors at each level, and the constraint forces for
// multipliers lambda, computed with Duals, must equal Simbody's, and their
// derivatives must match central differences.
void compareConstraintsDualToSimbody() {
    ConstraintTestSystem sys;
    const int nq = sys.state.getNQ(), nu = sys.state.getNU();
    const int m = sys.state.getNUDotErr();
    SimTK_TEST(sys.state.getNQErr() == 15 && sys.state.getNUErr() == 17 
               && m == 17);

    auto simbody = [&](const Vector& q, const Vector& u, const Vector& udot,
                       const Vector& lambda) {
        State s = sys.state;
        s.updQ() = q; s.updU() = u;
        sys.system.realize(s, Stage::Velocity);
        std::vector<Real> out;
        append(out, s.getQErr());
        append(out, s.getUErr());
        Vector pvaerr, bias, mobilityForces, GtLambda;
        Vector_<SpatialVec> bodyForces;
        sys.matter.calcConstraintAccelerationErrors(s, udot, pvaerr);
        sys.matter.calcConstraintAccelerationErrors(s, Vector(), bias);
        sys.matter.calcConstraintForcesFromMultipliers(s, lambda, bodyForces,
                                                       mobilityForces);
        sys.matter.multiplyByGTranspose(s, lambda, GtLambda);
        append(out, pvaerr); append(out, bias);
        append(out, bodyForces); append(out, mobilityForces);
        append(out, GtLambda);
        return out;
    };

    ScalarState<Dual> sx = sys.system.createScalarState<Dual>(sys.state);
    SimTK_TEST(sx.getNQErr() == sys.state.getNQErr()
               && sx.getNUErr() == sys.state.getNUErr());
    auto dual = [&](const std::vector<Dual>& q, const std::vector<Dual>& u,
                    const std::vector<Dual>& udot,
                    const std::vector<Dual>& lambda) {
        sx.setQ(q); sx.setU(u);
        SimTK_TEST_MUST_THROW(sx.getQErr());
        sys.system.realize(sx, Stage::Position);
        SimTK_TEST_MUST_THROW(sx.getUErr());
        sys.system.realize(sx, Stage::Velocity);
        std::vector<Dual> out;
        append(out, sx.getQErr());
        append(out, sx.getUErr());
        std::vector<Dual> pvaerr, bias, mobilityForces, GtLambda;
        std::vector<SpatialVec_<Dual>> bodyForces;
        sys.matter.calcConstraintAccelerationErrors(sx, udot, pvaerr);
        sys.matter.calcConstraintAccelerationErrors(sx, {}, bias);
        sys.matter.calcConstraintForcesFromMultipliers(sx, lambda, bodyForces,
                                                       mobilityForces);
        sys.matter.multiplyByGTranspose(sx, lambda, GtLambda);
        append(out, pvaerr); append(out, bias);
        append(out, bodyForces); append(out, mobilityForces);
        append(out, GtLambda);
        return out;
    };

    for (int trial=0; trial < 5; ++trial) {
        const Vector q = randVector(nq), u = randVector(nu),
                     udot = randVector(nu), lambda = randVector(m);
        const Vector dq = randVector(nq), du = randVector(nu),
                     dudot = randVector(nu), dlambda = randVector(m);
        const std::vector<Dual> y = dual(toDual(q, dq), toDual(u, du),
            toDual(udot, dudot), toDual(lambda, dlambda));
        const Real h = 1e-6;
        const std::vector<Real> y0 = simbody(q, u, udot, lambda);
        const std::vector<Real> yp = simbody(q + h*dq, u + h*du,
            udot + h*dudot, lambda + h*dlambda);
        const std::vector<Real> ym = simbody(q - h*dq, u - h*du,
            udot - h*dudot, lambda - h*dlambda);
        // The values match to roundoff, but not exactly: for mobilizers with
        // more than 3 dofs, other scalar types invert the articulated body
        // inertia's D matrix by Cholesky factorization instead of LAPACK.
        SimTK_TEST(y.size() == y0.size());
        for (size_t i=0; i < y0.size(); ++i) {
            SimTK_TEST_EQ_TOL(y[i].value(), y0[i], 1e-12);
            SimTK_TEST_EQ_TOL(y[i].deriv(), (yp[i] - ym[i])/(2*h), 1e-6);
        }
    }

    // Wrong numbers of multipliers or udots are errors.
    std::vector<Dual> f;
    std::vector<SpatialVec_<Dual>> F;
    SimTK_TEST_MUST_THROW(sys.matter.calcConstraintForcesFromMultipliers(
        sx, std::vector<Dual>(m-1), F, f));
    SimTK_TEST_MUST_THROW(sys.matter.calcConstraintAccelerationErrors(
        sx, std::vector<Dual>(nu+1), f));
}

} // anonymous namespace

// Check the mobilizers' kinematics by differentiation, independently of
// Simbody's Real results for H and HDot: with q's derivative set to qdot, the
// derivatives of the bodies' poses must be their velocities, and with u's
// derivative also set to udot, the derivatives of their velocities must be
// their accelerations.
void checkKinematicsByDifferentiation(const std::vector<int>& parents,
                                      const std::vector<std::string>& kinds,
                                      bool useEulerAngles) {
    TestSystem sys(parents, kinds, useEulerAngles);
    const int nq = sys.state.getNQ(), nu = sys.state.getNU();
    const int nb = sys.matter.getNumBodies();
    ScalarState<Dual> sx = sys.system.createScalarState<Dual>(sys.state);
    for (int trial=0; trial < 3; ++trial) {
        State s = sys.state;
        s.updQ() = randVector(nq); s.updU() = randVector(nu);
        sys.system.realize(s, Stage::Velocity);
        const Vector udot = randVector(nu);
        Vector_<SpatialVec> A_GB;
        sys.matter.calcBodyAccelerationFromUDot(s, udot, A_GB);

        sx.setQ(toDual(s.getQ(), s.getQDot()));
        sx.setU(toDual(s.getU(), udot));
        sys.system.realize(sx, Stage::Velocity);
        for (MobilizedBodyIndex b(1); b < nb; ++b) {
            const MobilizedBody& mobod = sys.matter.getMobilizedBody(b);
            const Transform_<Dual>& X_GB = mobod.getBodyTransform(sx);
            const SpatialVec_<Dual>& V_GB = mobod.getBodyVelocity(sx);
            const SpatialVec& V = mobod.getBodyVelocity(s);
            // d/dt R_GB = crossMat(w_GB) R_GB, and d/dt p_GB = v_GB.
            Mat33 R, RDot; Vec3 pDot;
            for (int i=0; i < 3; ++i) {
                pDot[i] = X_GB.p()[i].deriv();
                for (int j=0; j < 3; ++j) {
                    R(i,j) = X_GB.R().asMat33()(i,j).value();
                    RDot(i,j) = X_GB.R().asMat33()(i,j).deriv();
                }
            }
            SimTK_TEST_EQ_TOL(RDot, crossMat(V[0])*R, 1e-10);
            SimTK_TEST_EQ_TOL(pDot, V[1], 1e-10);
            for (int k=0; k < 2; ++k)
                for (int i=0; i < 3; ++i)
                    SimTK_TEST_EQ_TOL(V_GB[k][i].deriv(), A_GB[b][k][i],
                                      1e-10);
        }
    }
}

void testChain() {compareDualToSimbody(Chain);}
void testTree()  {compareDualToSimbody(Tree);}

// Each supported mobilizer type, forward and reversed, in a tree.
void testMobilizers() {
    std::vector<std::string> kinds;
    for (const std::string& kind : SupportedMobilizers) {
        kinds.push_back(kind); kinds.push_back(kind);
    }
    std::vector<int> parents;
    for (int i=0; i < (int)kinds.size(); ++i) parents.push_back(i/2);
    compareDualToSimbody(parents, kinds);
    compareDualToSimbody(parents, kinds, true); // Euler angles
    checkKinematicsByDifferentiation(parents, kinds, false);
    checkKinematicsByDifferentiation(parents, kinds, true);
}

// Stage tracking, copying, and misuse.
void testStages() {
    TestSystem sys(Chain);
    ScalarState<Dual> sx = sys.system.createScalarState<Dual>(sys.state);
    SimTK_TEST(sx.getSystemStage() == Stage::Model);
    // Operators check the stage.
    std::vector<Dual> Mv;
    SimTK_TEST_MUST_THROW(sys.matter.multiplyByM(sx, sx.getU(), Mv));
    SimTK_TEST_MUST_THROW(sys.matter.getMobilizedBody(MobilizedBodyIndex(1))
                              .getBodyTransform(sx));
    // Realizing Instance creates the matter subsystem's cache entry.
    sys.system.realize(sx, Stage::Instance);
    SimTK_TEST(sx.getSystemStage() == Stage::Instance);
    SimTK_TEST(sx.getNumCacheEntries(sys.matter.getMySubsystemIndex()) == 1);
    SimTK_TEST_MUST_THROW(sys.matter.multiplyByM(sx, sx.getU(), Mv));
    sys.system.realize(sx, Stage::Position);
    SimTK_TEST(sx.getSystemStage() == Stage::Position);
    sys.matter.multiplyByM(sx, sx.getU(), Mv);
    SimTK_TEST_MUST_THROW(sys.matter.calcKineticEnergy(sx));
    // Forces don't support other scalar types.
    SimTK_TEST_MUST_THROW(sys.system.realize(sx, Stage::Dynamics));
    sys.system.realize(sx);
    SimTK_TEST(sx.getSystemStage() == Stage::Velocity);
    // Changing u invalidates Velocity, q Position, and time Time; the
    // matter subsystem's cache entry is kept.
    sx.updU();
    SimTK_TEST(sx.getSystemStage() == Stage::Position);
    sys.system.realize(sx);
    sx.updQ();
    SimTK_TEST(sx.getSystemStage() == Stage::Time);
    sx.setTime(Dual(1));
    SimTK_TEST(sx.getSystemStage() == Stage::Instance);
    sys.matter.realizePositionKinematics(sx);
    SimTK_TEST(sx.getSystemStage() == Stage::Position);
    SimTK_TEST(sx.getNumCacheEntries(sys.matter.getMySubsystemIndex()) == 1);
    // Copies are independent.
    sys.system.realize(sx);
    ScalarState<Dual> copy(sx);
    SimTK_TEST(copy.getSystemStage() == Stage::Velocity);
    copy.updQ()[0] = Dual(1.5, 1);
    SimTK_TEST(copy.getSystemStage() == Stage::Time);
    SimTK_TEST(sx.getSystemStage() == Stage::Velocity);
    SimTK_TEST(sx.getQ()[0].value() != 1.5);
    sys.system.realize(copy);
    const MobilizedBody& body1 = 
        sys.matter.getMobilizedBody(MobilizedBodyIndex(1));
    SimTK_TEST(body1.getBodyTransform(copy).p()[0].value()
               != body1.getBodyTransform(sx).p()[0].value());
    // Initial time, q, and u come from the State.
    State s = sys.state;
    s.updQ()[2] = 0.25;
    s.setTime(3);
    const ScalarState<Dual> fromS = sys.system.createScalarState<Dual>(s);
    SimTK_TEST_EQ(fromS.getQ()[2].value(), 0.25);
    SimTK_TEST_EQ(fromS.getTime().value(), 3.);
    // A ScalarState can't be used with a different system, even an
    // identical one, and even before it's been realized.
    TestSystem other(Chain);
    SimTK_TEST_MUST_THROW(other.system.realize(sx));
    SimTK_TEST_MUST_THROW(other.matter.calcKineticEnergy(sx));
    ScalarState<Dual> fresh = sys.system.createScalarState<Dual>(sys.state);
    SimTK_TEST_MUST_THROW(other.system.realize(fresh, Stage::Instance));
    SimTK_TEST(fresh.getSystemStage() == Stage::Model);
    // The variables' lengths are fixed by the system.
    SimTK_TEST(fresh.updQ().size() == sys.state.getNQ());
    SimTK_TEST_MUST_THROW(fresh.setQ(std::vector<Dual>(sys.state.getNQ()-1)));
    SimTK_TEST_MUST_THROW(fresh.setU(std::vector<Dual>(sys.state.getNU()+1)));
    fresh.setQ(std::vector<Dual>(sys.state.getNQ(), Dual(0.1)));
    sys.system.realize(fresh);
    SimTK_TEST(fresh.getSystemStage() == Stage::Velocity);
}

// The mass matrix operators can be used in place (output = input).
void testInPlace() {
    TestSystem sys(Tree);
    const int nu = sys.state.getNU();
    State s = sys.state;
    s.updQ() = randVector(s.getNQ());
    sys.system.realize(s, Stage::Position);
    const Vector v = randVector(nu), dv = randVector(nu);
    Vector Mv, MInvV, inPlace = v;
    sys.matter.multiplyByM(s, v, Mv);
    sys.matter.multiplyByM(s, inPlace, inPlace);
    SimTK_TEST_EQ(inPlace, Mv);
    inPlace = v;
    sys.matter.multiplyByMInv(s, v, MInvV);
    sys.matter.multiplyByMInv(s, inPlace, inPlace);
    SimTK_TEST_EQ(inPlace, MInvV);

    ScalarState<Dual> sx = sys.system.createScalarState<Dual>(s);
    sys.system.realize(sx, Stage::Position);
    const std::vector<Dual> vx = toDual(v, dv);
    std::vector<Dual> Mvx, MInvVx, inPlaceX = vx;
    sys.matter.multiplyByM(sx, vx, Mvx);
    sys.matter.multiplyByM(sx, inPlaceX, inPlaceX);
    for (int i=0; i < nu; ++i) {
        SimTK_TEST_EQ(inPlaceX[i].value(), Mvx[i].value());
        SimTK_TEST_EQ(inPlaceX[i].deriv(), Mvx[i].deriv());
        SimTK_TEST_EQ(Mvx[i].value(), Mv[i]);
    }
    inPlaceX = vx;
    sys.matter.multiplyByMInv(sx, vx, MInvVx);
    sys.matter.multiplyByMInv(sx, inPlaceX, inPlaceX);
    for (int i=0; i < nu; ++i) {
        SimTK_TEST_EQ(inPlaceX[i].value(), MInvVx[i].value());
        SimTK_TEST_EQ(inPlaceX[i].deriv(), MInvVx[i].deriv());
        SimTK_TEST_EQ(MInvVx[i].value(), MInvV[i]);
    }
}

// A user-written Custom mobilizer (a pin about z), whose Real-only methods
// can't be evaluated with other scalar types.
class CustomPin : public MobilizedBody::Custom::Implementation {
public:
    explicit CustomPin(SimbodyMatterSubsystem& matter)
    :   Implementation(matter, 1, 1, 1) {}
    Implementation* clone() const override {return new CustomPin(*this);}
    Transform calcMobilizerTransformFromQ(const State&, int,
                                          const Real* q) const override
    {   return Transform(Rotation(q[0], ZAxis), Vec3(0)); }
    SpatialVec multiplyByHMatrix(const State&, int,
                                 const Real* u) const override
    {   return SpatialVec(Vec3(0, 0, u[0]), Vec3(0)); }
    void multiplyByHTranspose(const State&, const SpatialVec& F, int,
                              Real* f) const override {f[0] = F[0][2];}
    SpatialVec multiplyByHDotMatrix(const State&, int,
                                    const Real*) const override
    {   return SpatialVec(Vec3(0), Vec3(0)); }
    void multiplyByHDotTranspose(const State&, const SpatialVec&, int,
                                 Real* f) const override {f[0] = 0;}
};

void testUnsupported() {
    MultibodySystem system;
    SimbodyMatterSubsystem matter(system);
    MobilizedBody::Custom(matter.Ground(), new CustomPin(matter),
                          Body::Rigid(MassProperties(1, Vec3(0), Inertia(1))));
    const State state = system.realizeTopology();
    system.realize(state, Stage::Instance);
    ScalarState<Dual> sx = system.createScalarState<Dual>(state);
    SimTK_TEST_MUST_THROW(system.realize(sx, Stage::Instance));
}

// A Rod whose endpoints coincide has no direction. Real uses F's z axis, and
// other scalar types (which can't branch) get the same direction, finite
// forces, and finite velocity-dependent terms.
void testCoincidentRodEndpoints() {
    RodConstraintKernel rod;
    rod.p_FSf = Vec3(0.1, 0.2, 0.3); rod.p_BSb = Vec3(-0.2, 0.1, 0.4);
    rod.length = 1;
    const Transform X_AF(Rotation(0.7, UnitVec3(1, 2, 3)), Vec3(0.5, -0.2, 1));
    const Rotation R_AB(-0.4, UnitVec3(0, 1, 1));
    // Place B so that Sb is at Sf.
    const Transform X_AB(R_AB, X_AF*rod.p_FSf - R_AB*rod.p_BSb);
    const SpatialVec V_AF(Vec3(0.3, -0.1, 0.2), Vec3(1, 2, 3));
    const SpatialVec V_AB(Vec3(-0.2, 0.4, 0.1), Vec3(0.5, 0, -1));

    RodConstraintKernel::PositionCache<Real> pc;
    RodConstraintKernel::VelocityCache<Real> vc;
    rod.calcPositionCache(X_AF, X_AB, pc);
    rod.calcVelocityCache(pc, V_AF, V_AB, vc);
    SimTK_TEST(pc.isSingular);
    const Real lambda = 2.5;
    SpatialVec FF(Vec3(0), Vec3(0)), FB(Vec3(0), Vec3(0));
    rod.addInPositionConstraintForces(pc, &lambda, FF, FB);

    auto toDualVec = [](const Vec3& v)
    {   return Vec<3,Dual>(Dual(v[0], 1), Dual(v[1]), Dual(v[2])); };
    Transform_<Dual> X_AFx = sbCast<Dual>(X_AF), X_ABx = sbCast<Dual>(X_AB);
    X_ABx.updP() = toDualVec(X_AB.p()); // a direction of differentiation
    const SpatialVec_<Dual> V_AFx(sbCast<Dual>(V_AF[0]), sbCast<Dual>(V_AF[1]));
    const SpatialVec_<Dual> V_ABx(sbCast<Dual>(V_AB[0]), sbCast<Dual>(V_AB[1]));
    RodConstraintKernel::PositionCache<Dual> pcx;
    RodConstraintKernel::VelocityCache<Dual> vcx;
    rod.calcPositionCache(X_AFx, X_ABx, pcx);
    rod.calcVelocityCache(pcx, V_AFx, V_ABx, vcx);
    const Dual lambdax(lambda);
    SpatialVec_<Dual> FFx(Vec<3,Dual>(Dual(0)), Vec<3,Dual>(Dual(0)));
    SpatialVec_<Dual> FBx = FFx;
    rod.addInPositionConstraintForces(pcx, &lambdax, FFx, FBx);
    for (int i=0; i < 3; ++i) {
        SimTK_TEST_EQ(pcx.Cz_A[i].value(), pc.Cz_A[i]);
        SimTK_TEST(std::isfinite(pcx.Cz_A[i].deriv()));
        SimTK_TEST(std::isfinite(vcx.Czd_A[i].value()));
        for (int k=0; k < 2; ++k) {
            SimTK_TEST_EQ(FFx[k][i].value(), FF[k][i]);
            SimTK_TEST_EQ(FBx[k][i].value(), FB[k][i]);
        }
    }
}

// Disabled constraints (such as OpenSim's coordinate locks) are allowed, and
// so are enabled ones of the supported types, but not enabled ones of other
// types.
void testConstraints() {
    compareConstraintsDualToSimbody();

    MultibodySystem system;
    SimbodyMatterSubsystem matter(system);
    MobilizedBody::Pin body(matter.Ground(), Transform(),
        Body::Rigid(MassProperties(1, Vec3(0, -1, 0), Inertia(1))),
        Transform());
    Constraint::ConstantSpeed lock(body, 0.);
    lock.setDisabledByDefault(true);
    State state = system.realizeTopology();
    system.realize(state, Stage::Instance);
    ScalarState<Dual> sx = system.createScalarState<Dual>(state);
    system.realize(sx, Stage::Velocity);
    SimTK_TEST(sx.getSystemStage() == Stage::Velocity);

    lock.enable(state);
    system.realize(state, Stage::Instance);
    ScalarState<Dual> enabled = system.createScalarState<Dual>(state);
    SimTK_TEST_MUST_THROW(system.realize(enabled, Stage::Instance));
}

int main() {
    SimTK_START_TEST("TestMultibodyScalarDual");
        SimTK_SUBTEST(testChain);
        SimTK_SUBTEST(testTree);
        SimTK_SUBTEST(testMobilizers);
        SimTK_SUBTEST(testStages);
        SimTK_SUBTEST(testInPlace);
        SimTK_SUBTEST(testUnsupported);
        SimTK_SUBTEST(testConstraints);
        SimTK_SUBTEST(testCoincidentRodEndpoints);
    SimTK_END_TEST();
}
