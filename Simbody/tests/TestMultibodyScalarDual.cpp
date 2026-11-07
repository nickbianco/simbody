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
// Jacobian products, and kinetic energy. Also, force elements written once as
// a template on the scalar type, in a GeneralForceSubsystem_<Dual>, must agree
// with the same elements in Simbody's GeneralForceSubsystem.

// The scalar traits for Dual must be defined before any other SimTKcommon
// header is included.
#include "SimTKcommon/Scalar.h"
#include "SimTKcommon/internal/RealScalarType.h"

#include <cmath>
#include <iostream>
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
#include "simbody/internal/tree/MultibodyScalarImpl.h"
#include "simbody/internal/tree/ForceScalarImpl.h"
#include "SimTKcommon/Testing.h"

using namespace SimTK;
using SimTKDualTest::Dual;

// Instantiate everything so we catch compilation problems.
SimTK_INSTANTIATE_MULTIBODY_SCALAR(Dual);
SimTK_INSTANTIATE_FORCE_SCALAR(Dual);

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
struct TestSystem {
    MultibodySystem           system;
    SimbodyMatterSubsystem    matter;
    GeneralForceSubsystem     forces;
    Force::Gravity            gravity;
    Force::DiscreteForces     discreteForces;
    State                     state;

    explicit TestSystem(const std::vector<int>& parents)
    :   matter(system), forces(system),
        gravity(forces, matter, Gravity),
        discreteForces(forces, matter) {
        for (int i=0; i < (int)parents.size(); ++i) {
            MobilizedBody& parent =
                matter.updMobilizedBody(MobilizedBodyIndex(parents[i]));
            const Body::Rigid body(randMassProperties());
            const Transform X_PF = Test::randTransform();
            const Transform X_BM = Test::randTransform();
            const MobilizedBody::Direction dir = (i % 4 == 1)
                ? MobilizedBody::Reverse : MobilizedBody::Forward;
            if      (i == 3 || i == 5)
                MobilizedBody::Planar(parent, X_PF, body, X_BM, dir);
            else if (i % 5 == 4) MobilizedBody::Weld  (parent, X_PF, body, X_BM);
            else if (i % 3 == 2) MobilizedBody::Slider(parent, X_PF, body, X_BM, dir);
            else                 MobilizedBody::Pin   (parent, X_PF, body, X_BM, dir);
        }
        state = system.realizeTopology();
        system.realize(state, Stage::Instance);
    }
};

// Gravity body forces for a ScalarState, computed as Force::Gravity does.
// The built-in force elements are Real only, so these are computed by the
// caller.
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

void compareDualToSimbody(const std::vector<int>& parents) {
    TestSystem sys(parents);
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
    ScalarState<Dual> sx(sys.state);
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

        SimTK_TEST(y.size() == y0.size());
        for (size_t i=0; i < y0.size(); ++i) {
            SimTK_TEST_EQ(y[i].value(), y0[i]);
            SimTK_TEST_EQ_TOL(y[i].deriv(), (yp[i] - ym[i])/(2*h), 1e-6);
        }
    }
}

// Body i+1 has parent parents[i].
const std::vector<int> Chain{0, 1, 2, 3, 4, 5, 6, 7};
const std::vector<int> Tree{0, 1, 1, 0, 4, 2, 2, 5, 3};

//==============================================================================
//                        FORCE ELEMENTS FOR ANY SCALAR
//==============================================================================
template <class T>
Vec<3,T> toT(const Vec3& v) {return Vec<3,T>(T(v[0]), T(v[1]), T(v[2]));}

// A mobilized body's first q and its u index, from either kind of state.
Real getFirstQ(const SimbodyMatterSubsystem&, const MobilizedBody& mobod,
               const State& s)
{   return mobod.getOneQ(s, MobilizerQIndex(0)); }
template <class T>
T getFirstQ(const SimbodyMatterSubsystem& matter, const MobilizedBody& mobod,
            const ScalarState<T>& s) {
    return s.getQ()[s.getQStart(matter.getMySubsystemIndex())
                    + mobod.getFirstQIndex(s.getRealState())];
}
const State& getRealState(const State& s) {return s;}
template <class T>
const State& getRealState(const ScalarState<T>& s) {return s.getRealState();}

// A linear spring and damper between stations on two bodies, and a torsional
// spring on another body's first mobility, written once for any scalar type.
template <class T>
class SpringDamper : public Force_<T>::Custom::Implementation {
    using Super = typename Force_<T>::Custom::Implementation;
public:
    using typename Super::StateType;
    using typename Super::BodyForceVector;
    using typename Super::ParticleForceVector;
    using typename Super::MobilityForceVector;

    SpringDamper(const SimbodyMatterSubsystem& matter,
                 MobilizedBodyIndex body1, const Vec3& station1,
                 MobilizedBodyIndex body2, const Vec3& station2,
                 Real k, Real c, Real x0, MobilizedBodyIndex jointBody,
                 Real kq)
    :   matter(matter), body1(body1), body2(body2), station1(station1),
        station2(station2), k(k), c(c), x0(x0), jointBody(jointBody),
        kq(kq) {}

    void calcForce(const StateType& s, BodyForceVector& bodyForces,
                   ParticleForceVector&,
                   MobilityForceVector& mobilityForces) const override {
        const MobilizedBody& mobod1 = matter.getMobilizedBody(body1);
        const MobilizedBody& mobod2 = matter.getMobilizedBody(body2);
        const Transform_<T>&  X_GB1 = mobod1.getBodyTransform(s);
        const Transform_<T>&  X_GB2 = mobod2.getBodyTransform(s);
        const SpatialVec_<T>& V_GB1 = mobod1.getBodyVelocity(s);
        const SpatialVec_<T>& V_GB2 = mobod2.getBodyVelocity(s);
        const Vec<3,T> r1 = X_GB1.R()*toT<T>(station1);
        const Vec<3,T> r2 = X_GB2.R()*toT<T>(station2);
        const Vec<3,T> d = (X_GB2.p() + r2) - (X_GB1.p() + r1);
        const Vec<3,T> v = (V_GB2[1] + V_GB2[0] % r2)
                         - (V_GB1[1] + V_GB1[0] % r1);
        const T length = sqrt(~d*d);
        const Vec<3,T> e = d/length;
        const T tension = T(k)*(length - T(x0)) + T(c)*(~v*e);
        const Vec<3,T> f1 = tension*e; // on body 1, toward body 2
        bodyForces[body1] += SpatialVec_<T>(r1 % f1, f1);
        bodyForces[body2] -= SpatialVec_<T>(r2 % f1, f1);

        const MobilizedBody& joint = matter.getMobilizedBody(jointBody);
        mobilityForces[joint.getFirstUIndex(getRealState(s))] -=
            T(kq)*getFirstQ(matter, joint, s);
    }

    T calcPotentialEnergy(const StateType& s) const override {
        const Transform_<T>& X_GB1 =
            matter.getMobilizedBody(body1).getBodyTransform(s);
        const Transform_<T>& X_GB2 =
            matter.getMobilizedBody(body2).getBodyTransform(s);
        const Vec<3,T> d = (X_GB2.p() + X_GB2.R()*toT<T>(station2))
                         - (X_GB1.p() + X_GB1.R()*toT<T>(station1));
        const T stretch = sqrt(~d*d) - T(x0);
        const T q = getFirstQ(matter, matter.getMobilizedBody(jointBody), s);
        return T(0.5*k)*stretch*stretch + T(0.5*kq)*q*q;
    }

private:
    const SimbodyMatterSubsystem& matter;
    MobilizedBodyIndex body1, body2;
    Vec3 station1, station2;
    Real k, c, x0;
    MobilizedBodyIndex jointBody;
    Real kq;
};

// A chain of pin, slider, and planar bodies with the same force elements in
// Simbody's GeneralForceSubsystem and in a GeneralForceSubsystem_<Dual>. The
// second spring is disabled by default.
struct ForceTestSystem {
    MultibodySystem                 system;
    SimbodyMatterSubsystem          matter;
    GeneralForceSubsystem           forces;
    GeneralForceSubsystem_<Dual>    dualForces;
    Force::Custom                   spring, disabledSpring;
    Force_<Dual>::Custom            dualSpring, dualDisabledSpring;
    State                           state;

    ForceTestSystem() : matter(system), forces(system), dualForces(system) {
        MobilizedBody::Pin b1(matter.Ground(), Test::randTransform(),
                              Body::Rigid(randMassProperties()),
                              Test::randTransform());
        MobilizedBody::Slider b2(b1, Test::randTransform(),
                                 Body::Rigid(randMassProperties()),
                                 Test::randTransform());
        MobilizedBody::Planar b3(b2, Test::randTransform(),
                                 Body::Rigid(randMassProperties()),
                                 Test::randTransform(), MobilizedBody::Reverse);
        MobilizedBody::Pin b4(b3, Test::randTransform(),
                              Body::Rigid(randMassProperties()),
                              Test::randTransform());
        spring = Force::Custom(forces, new SpringDamper<Real>(matter,
            b1, Vec3(0.1, 0.2, 0.3), b4, Vec3(-0.2, 0.1, 0), 50, 2, 0.5,
            b2, 7));
        dualSpring = Force_<Dual>::Custom(dualForces, new SpringDamper<Dual>(
            matter, b1, Vec3(0.1, 0.2, 0.3), b4, Vec3(-0.2, 0.1, 0), 50, 2,
            0.5, b2, 7));
        disabledSpring = Force::Custom(forces, new SpringDamper<Real>(matter,
            b2, Vec3(0), b3, Vec3(0.3, 0, 0), 1000, 10, 0, b4, 100));
        dualDisabledSpring = Force_<Dual>::Custom(dualForces,
            new SpringDamper<Dual>(matter, b2, Vec3(0), b3, Vec3(0.3, 0, 0),
                                   1000, 10, 0, b4, 100));
        disabledSpring.setDisabledByDefault(true);
        dualDisabledSpring.setDisabledByDefault(true);
        state = system.realizeTopology();
        system.realize(state, Stage::Instance);
    }
};

Array_<ForceIndex> allForces(int n) {
    Array_<ForceIndex> all;
    for (int i=0; i < n; ++i) all.push_back(ForceIndex(i));
    return all;
}

} // anonymous namespace

void testChain() {compareDualToSimbody(Chain);}
void testTree()  {compareDualToSimbody(Tree);}

// Stage tracking, copying, and misuse.
void testStages() {
    TestSystem sys(Chain);
    ScalarState<Dual> sx(sys.state);
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
    SimTK_TEST_EQ(ScalarState<Dual>(s).getQ()[2].value(), 0.25);
    SimTK_TEST_EQ(ScalarState<Dual>(s).getTime().value(), 3.);
    // A ScalarState can't be used with a different system.
    TestSystem other(Chain);
    SimTK_TEST_MUST_THROW(other.system.realize(sx));
    SimTK_TEST_MUST_THROW(other.matter.calcKineticEnergy(sx));
}

void testUnsupported() {
    MultibodySystem system;
    SimbodyMatterSubsystem matter(system);
    MobilizedBody::Ball(matter.Ground(), Transform(),
                        Body::Rigid(MassProperties(1, Vec3(0), Inertia(1))),
                        Transform());
    const State state = system.realizeTopology();
    system.realize(state, Stage::Instance);
    ScalarState<Dual> sx(state);
    SimTK_TEST_MUST_THROW(system.realize(sx, Stage::Instance));
}

// Disabled constraints (such as OpenSim's coordinate locks) are allowed, and
// enabled ones aren't.
void testConstraints() {
    MultibodySystem system;
    SimbodyMatterSubsystem matter(system);
    MobilizedBody::Pin body(matter.Ground(), Transform(),
        Body::Rigid(MassProperties(1, Vec3(0, -1, 0), Inertia(1))),
        Transform());
    Constraint::ConstantSpeed lock(body, 0.);
    lock.setDisabledByDefault(true);
    State state = system.realizeTopology();
    system.realize(state, Stage::Instance);
    ScalarState<Dual> sx(state);
    system.realize(sx, Stage::Velocity);
    SimTK_TEST(sx.getSystemStage() == Stage::Velocity);

    lock.enable(state);
    system.realize(state, Stage::Instance);
    ScalarState<Dual> enabled(state);
    SimTK_TEST_MUST_THROW(system.realize(enabled, Stage::Instance));
}

// A force element written once as a template on the scalar type, evaluated
// with Duals in a GeneralForceSubsystem_<Dual>: its forces, its potential
// energy, and the accelerations they cause must equal Simbody's for the same
// element in a GeneralForceSubsystem, and their derivatives must match
// central differences.
void testForces() {
    ForceTestSystem sys;
    const int nq = sys.state.getNQ(), nu = sys.state.getNU();
    const Array_<ForceIndex> all = allForces(2);

    auto simbody = [&](const Vector& q, const Vector& u) {
        State s = sys.state;
        s.updQ() = q; s.updU() = u;
        sys.system.realize(s, Stage::Acceleration);
        Vector_<SpatialVec> F; Vector f;
        sys.forces.calcForceContributionsSum(s, all, F, f);
        std::vector<Real> out;
        append(out, F); append(out, f);
        append(out, sys.spring.calcPotentialEnergyContribution(s));
        // The GeneralForceSubsystem_<Dual> adds nothing to the System's
        // potential energy or forces, so this is the Real spring's alone.
        append(out, sys.system.calcPotentialEnergy(s));
        append(out, s.getUDot());
        return out;
    };

    ScalarState<Dual> sx(sys.state);
    auto dual = [&](const std::vector<Dual>& q, const std::vector<Dual>& u) {
        sx.setQ(q); sx.setU(u);
        sys.system.realize(sx, Stage::Velocity);
        std::vector<SpatialVec_<Dual>> F; std::vector<Dual> f;
        sys.dualForces.calcForceContributionsSum(sx, all, F, f);
        std::vector<Dual> out;
        append(out, F); append(out, f);
        const Dual pe = sys.dualSpring.calcPotentialEnergyContribution(sx);
        append(out, pe); append(out, pe);
        std::vector<Dual> udot;
        std::vector<SpatialVec_<Dual>> A_GB;
        sys.matter.calcAccelerationIgnoringConstraints(sx, f, F, udot, A_GB);
        append(out, udot);
        return out;
    };

    for (int trial=0; trial < 5; ++trial) {
        const Vector q = randVector(nq), u = randVector(nu);
        const Vector dq = randVector(nq), du = randVector(nu);
        const std::vector<Dual> y = dual(toDual(q, dq), toDual(u, du));
        const Real h = 1e-6;
        const std::vector<Real> y0 = simbody(q, u);
        const std::vector<Real> yp = simbody(q + h*dq, u + h*du);
        const std::vector<Real> ym = simbody(q - h*dq, u - h*du);
        SimTK_TEST(y.size() == y0.size());
        for (size_t i=0; i < y0.size(); ++i) {
            SimTK_TEST_EQ(y[i].value(), y0[i]);
            SimTK_TEST_EQ_TOL(y[i].deriv(), (yp[i] - ym[i])/(2*h), 1e-5);
        }
    }
}

// Handles, downcasts, and enabling and disabling (which use the Real State).
void testForceBookkeeping() {
    ForceTestSystem sys;
    SimTK_TEST(sys.dualForces.getNumForces() == 2);
    SimTK_TEST(sys.dualSpring.getForceIndex() == ForceIndex(0));
    SimTK_TEST(sys.dualDisabledSpring.getForceIndex() == ForceIndex(1));
    SimTK_TEST(sys.dualSpring.getForceSubsystem().getNumForces() == 2);
    SimTK_TEST(Force_<Dual>::Custom::isInstanceOf(
        sys.dualForces.getForce(ForceIndex(1))));
    SimTK_TEST(GeneralForceSubsystem::isInstanceOf(sys.forces));
    SimTK_TEST(!GeneralForceSubsystem::isInstanceOf(sys.dualForces));
    SimTK_TEST(GeneralForceSubsystem_<Dual>::isInstanceOf(sys.dualForces));
    SimTK_TEST(!GeneralForceSubsystem_<Dual>::isInstanceOf(sys.forces));

    SimTK_TEST(!sys.dualSpring.isDisabled(sys.state));
    SimTK_TEST(sys.dualDisabledSpring.isDisabled(sys.state));
    State s = sys.state;
    s.updQ() = randVector(s.getNQ()); s.updU() = randVector(s.getNU());
    sys.dualSpring.disable(s);
    sys.dualDisabledSpring.enable(s);
    sys.disabledSpring.enable(s);
    SimTK_TEST(sys.dualSpring.isDisabled(s));
    SimTK_TEST(!sys.dualDisabledSpring.isDisabled(s));
    sys.system.realize(s, Stage::Velocity);
    ScalarState<Dual> sx(s);
    sys.system.realize(sx, Stage::Velocity);

    // A disabled element contributes nothing.
    std::vector<SpatialVec_<Dual>> F;
    std::vector<Vec<3,Dual>> particleForces;
    std::vector<Dual> f;
    sys.dualSpring.calcForceContribution(sx, F, particleForces, f);
    SimTK_TEST(F.size() == (size_t)sys.matter.getNumBodies());
    SimTK_TEST(f.size() == (size_t)sys.matter.getNumMobilities());
    for (const auto& Fb : F) for (int k=0; k < 2; ++k) for (int i=0; i < 3; ++i)
        SimTK_TEST(Fb[k][i].value() == 0);
    for (const auto& fi : f) SimTK_TEST(fi.value() == 0);
    SimTK_TEST(sys.dualSpring.calcPotentialEnergyContribution(sx).value()
               == 0);

    // The enabled one equals its Real twin.
    Vector_<SpatialVec> FReal; Vector_<Vec3> particleReal; Vector fReal;
    sys.disabledSpring.calcForceContribution(s, FReal, particleReal, fReal);
    sys.dualDisabledSpring.calcForceContribution(sx, F, particleForces, f);
    for (int b=0; b < FReal.size(); ++b)
        for (int k=0; k < 2; ++k) for (int i=0; i < 3; ++i)
            SimTK_TEST_EQ(F[b][k][i].value(), FReal[b][k][i]);
    for (int i=0; i < fReal.size(); ++i) SimTK_TEST_EQ(f[i].value(), fReal[i]);
    SimTK_TEST_EQ(
        sys.dualDisabledSpring.calcPotentialEnergyContribution(sx).value(),
        sys.disabledSpring.calcPotentialEnergyContribution(s));
}

int main() {
    SimTK_START_TEST("TestMultibodyScalarDual");
        SimTK_SUBTEST(testChain);
        SimTK_SUBTEST(testTree);
        SimTK_SUBTEST(testStages);
        SimTK_SUBTEST(testUnsupported);
        SimTK_SUBTEST(testConstraints);
        SimTK_SUBTEST(testForces);
        SimTK_SUBTEST(testForceBookkeeping);
    SimTK_END_TEST();
}
