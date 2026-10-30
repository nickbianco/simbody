/* -------------------------------------------------------------------------- *
 *                               Simbody(tm)                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the SimTK biosimulation toolkit originating from           *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org/home/simbody.  *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 * Authors: Nicholas Bianco                                                   *
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

/* Tests for
SimbodyMatterSubsystem::calcResidualForceDerivativesIgnoringConstraints(), the
analytic first-order partials of inverse dynamics with respect to q and u.

Most of the checks compare against central finite differences of
calcResidualForceIgnoringConstraints() itself, so the operator is held to the
same definition of the residual that Simbody already exposes. Two checks need no
perturbation at all and so are much tighter: an Euler sum rule on dtau/du, and
invariance of the derivatives to where the model sits relative to the Ground
origin. */

#include "SimTKsimbody.h"

#include <iostream>
using namespace SimTK;

static const Real GravityMag = 9.81;


//==============================================================================
//                                 HELPERS
//==============================================================================

// Arbitrary but reproducible inputs, so a failure is always reproducible.
static Vector makeUdot(int nu) {
    Vector udot(nu);
    for (int i=0; i < nu; ++i)
        udot[i] = 0.35 + 0.27*i - 0.11*(i%3) - 0.05*(i%5);
    return udot;
}
static Vector makeMobilityForces(int nu) {
    Vector f(nu);
    for (int i=0; i < nu; ++i) f[i] = 0.9 - 0.31*i + 0.17*(i%4);
    return f;
}
static Vector_<SpatialVec> makeBodyForces(int nb) {
    Vector_<SpatialVec> F(nb, SpatialVec(Vec3(0), Vec3(0)));
    for (MobilizedBodyIndex bx(1); bx < nb; ++bx)
        F[bx] = SpatialVec(Vec3(0.3*(int)bx, -0.2, 0.1*(int)bx),
                           Vec3(-0.4, 0.5*(int)bx, 0.25));
    return F;
}

// Compare the analytic Jacobians against central differences of the residual.
// The differencing error on a tau derivative is set by eps*|tau|/h rather than
// by the size of the derivative itself, so the tolerance is scaled to the
// residual.
static void checkAgainstFiniteDifferences(const MultibodySystem& system,
                                          const State&           state,
                                          Real                   relTol = 1e-6)
{
    const SimbodyMatterSubsystem& matter = system.getMatterSubsystem();
    const int nu = state.getNU();
    const int nb = matter.getNumBodies();

    const Vector              udot       = makeUdot(nu);
    const Vector              mobForces  = makeMobilityForces(nu);
    const Vector_<SpatialVec> bodyForces = makeBodyForces(nb);

    Matrix dTauDQ, dTauDU;
    matter.calcResidualForceDerivativesIgnoringConstraints(
        state, bodyForces, udot, dTauDQ, dTauDU);
    SimTK_TEST(dTauDQ.nrow()==nu && dTauDQ.ncol()==nu);
    SimTK_TEST(dTauDU.nrow()==nu && dTauDU.ncol()==nu);

    Vector tau;
    matter.calcResidualForceIgnoringConstraints(state, mobForces, bodyForces,
                                                udot, tau);
    Real tauScale = 1;
    for (int i=0; i < nu; ++i) tauScale = std::max(tauScale, std::abs(tau[i]));
    const Real tol = relTol * tauScale;

    const Real h = 1e-5;
    for (int j=0; j < nu; ++j) {
        Vector e(nu, Real(0)); e[j] = 1;

        // q perturbation runs along the mobility direction, mapped through N
        // since qdot = N u.
        Vector Ne; matter.multiplyByN(state, false, e, Ne);
        State qp = state, qm = state;
        qp.updQ() += h*Ne;  qm.updQ() -= h*Ne;
        system.realize(qp, Stage::Velocity);
        system.realize(qm, Stage::Velocity);
        Vector tp, tm;
        matter.calcResidualForceIgnoringConstraints(qp, mobForces, bodyForces,
                                                    udot, tp);
        matter.calcResidualForceIgnoringConstraints(qm, mobForces, bodyForces,
                                                    udot, tm);
        SimTK_TEST_EQ_TOL(dTauDQ(j), (tp-tm)/(2*h), tol);

        State up = state, um = state;
        up.updU()[j] += h;  um.updU()[j] -= h;
        system.realize(up, Stage::Velocity);
        system.realize(um, Stage::Velocity);
        matter.calcResidualForceIgnoringConstraints(up, mobForces, bodyForces,
                                                    udot, tp);
        matter.calcResidualForceIgnoringConstraints(um, mobForces, bodyForces,
                                                    udot, tm);
        SimTK_TEST_EQ_TOL(dTauDU(j), (tp-tm)/(2*h), tol);
    }
}

static MassProperties cylinderProps(Real m, Vec3 com, Real halfLen, Real rad) {
    const Inertia Ic = m*Inertia::cylinderAlongY(rad, halfLen);
    return MassProperties(m, com, Ic.shiftFromMassCenter(-com, m));
}


//==============================================================================
//                                  TESTS
//==============================================================================

// Simplest possible case: one revolute joint. Catches gross sign errors before
// any of the multi-body or multi-DoF machinery is involved.
void testSinglePin() {
    MultibodySystem system;
    SimbodyMatterSubsystem matter(system);
    Body::Rigid body(cylinderProps(2.0, Vec3(0.03,-0.02,0.05), 0.3, 0.05));
    MobilizedBody::Pin pin(matter.Ground(),
        Transform(Rotation(0.2, XAxis), Vec3(0.05,0.1,-0.02)), body,
        Transform(Rotation(0.15, YAxis), Vec3(0,0.3,0)));

    State state = system.realizeTopology();
    pin.setAngle(state, 0.6);  pin.setRate(state, -1.3);
    system.realize(state, Stage::Velocity);
    checkAgainstFiniteDifferences(system, state);
}

// A serial chain exercises propagation of the transmitted-force sensitivity
// from a perturbed joint up through all of its ancestors.
void testSerialPinChain() {
    MultibodySystem system;
    SimbodyMatterSubsystem matter(system);
    Body::Rigid body(cylinderProps(1.1, Vec3(0.02,0.01,-0.03), 0.2, 0.04));

    MobilizedBody prev = matter.Ground();
    for (int i=0; i < 5; ++i)
        prev = MobilizedBody::Pin(prev,
            Transform(Rotation(0.1*(i+1), XAxis), Vec3(0.02*i,-0.2,0.01)),
            body, Transform(Rotation(0.07*(i+1), ZAxis), Vec3(0,0.2,0)));

    State state = system.realizeTopology();
    const int nu = state.getNU();
    for (int i=0; i < nu; ++i) {
        state.updQ()[i] = 0.3 - 0.17*i;
        state.updU()[i] = -0.8 + 0.23*i;
    }
    system.realize(state, Stage::Velocity);
    checkAgainstFiniteDifferences(system, state);
}

// A floating base, a multi-DoF interior joint and a branch, with mass centers
// off the body origins and skew mobilizer frames so no term drops out by
// accident of topology. The multi-DoF rotational modes are what exercise the
// Simbody-specific self-column term of dH/dq, which vanishes identically on a
// Pin-only model.
void testBranchedMultiDof() {
    MultibodySystem system;
    SimbodyMatterSubsystem matter(system);
    Body::Rigid body1(cylinderProps(2.0, Vec3( 0.03,-0.02, 0.05), 0.30, 0.05));
    Body::Rigid body2(cylinderProps(1.4, Vec3(-0.02, 0.04,-0.01), 0.25, 0.04));
    Body::Rigid body3(cylinderProps(0.8, Vec3( 0.01, 0.03, 0.02), 0.18, 0.03));
    Body::Rigid body4(cylinderProps(0.6, Vec3( 0.02,-0.01,-0.03), 0.15, 0.03));

    MobilizedBody::Free link1(matter.Ground(),
        Transform(Rotation(0.20, XAxis), Vec3(0.05, 0.10,-0.02)), body1,
        Transform(Rotation(0.15, YAxis), Vec3(0.00, 0.30, 0.00)));
    MobilizedBody::Ball link2(link1,
        Transform(Rotation(0.35, ZAxis), Vec3(0.00,-0.30, 0.04)), body2,
        Transform(Rotation(0.10, XAxis), Vec3(0.02, 0.25, 0.00)));
    MobilizedBody::Pin  link3(link2,
        Transform(Rotation(0.40, XAxis), Vec3(0.00,-0.25, 0.03)), body3,
        Transform(Rotation(0.25, ZAxis), Vec3(0.00, 0.18, 0.00)));
    MobilizedBody::Pin  link4(link1,
        Transform(Rotation(-0.30, YAxis), Vec3(0.12,-0.10,-0.05)), body4,
        Transform(Rotation(0.05, XAxis), Vec3(0.00, 0.15, 0.00)));

    State state = system.realizeTopology();
    // Set through the fitting methods so the quaternions stay normalized.
    link1.setQToFitTransform(state,
        Transform(Rotation(BodyRotationSequence, 0.30, XAxis, -0.45, YAxis,
                           0.25, ZAxis), Vec3(0.15, 0.40,-0.20)));
    link1.setUToFitVelocity(state,
        SpatialVec(Vec3(0.70,-1.10,0.50), Vec3(-0.40,0.90,1.30)));
    link2.setQToFitRotation(state,
        Rotation(BodyRotationSequence, -0.55, XAxis, 0.35, YAxis, 0.60, ZAxis));
    link2.setUToFitAngularVelocity(state, Vec3(1.20, 0.60,-0.85));
    link3.setAngle(state, 0.65);  link3.setRate(state,-1.45);
    link4.setAngle(state,-0.85);  link4.setRate(state, 0.95);
    system.realize(state, Stage::Velocity);

    checkAgainstFiniteDifferences(system, state);
}

// Purely translational mobilizers: every angular mode direction is zero, so all
// the cross-product terms drop out and the derivatives must still be right.
void testTranslationOnly() {
    MultibodySystem system;
    SimbodyMatterSubsystem matter(system);
    Body::Rigid body(cylinderProps(1.3, Vec3(0.04,-0.01,0.02), 0.2, 0.05));
    MobilizedBody::Translation t1(matter.Ground(), Transform(), body,
                                 Transform(Vec3(0.01,0.1,-0.02)));
    MobilizedBody::Slider      s1(t1,
        Transform(Rotation(0.3, YAxis), Vec3(0,-0.2,0)), body,
        Transform(Vec3(0,0.15,0)));

    State state = system.realizeTopology();
    const int nu = state.getNU();
    for (int i=0; i < nu; ++i) {
        state.updQ()[i] =  0.12 - 0.05*i;
        state.updU()[i] = -0.30 + 0.11*i;
    }
    system.realize(state, Stage::Velocity);
    checkAgainstFiniteDifferences(system, state);
}

// With udot and the applied loads zeroed, the residual is exactly the
// velocity-dependent generalized force, which is quadratic homogeneous in u.
// Euler's theorem for homogeneous functions then requires
//     sum_j (dtau_i/du_j) u_j = 2 tau_i.
// This needs no perturbation, so it holds to machine precision and is a far
// sharper check than the finite-difference comparisons above.
void testEulerSumRuleOnDTauDU() {
    MultibodySystem system;
    SimbodyMatterSubsystem matter(system);
    Body::Rigid body1(cylinderProps(2.0, Vec3( 0.03,-0.02, 0.05), 0.30, 0.05));
    Body::Rigid body2(cylinderProps(1.4, Vec3(-0.02, 0.04,-0.01), 0.25, 0.04));

    MobilizedBody::Free link1(matter.Ground(),
        Transform(Rotation(0.2, XAxis), Vec3(0.05,0.1,-0.02)), body1,
        Transform(Rotation(0.15, YAxis), Vec3(0,0.3,0)));
    MobilizedBody::Ball link2(link1,
        Transform(Rotation(0.35, ZAxis), Vec3(0,-0.3,0.04)), body2,
        Transform(Rotation(0.1, XAxis), Vec3(0.02,0.25,0)));

    State state = system.realizeTopology();
    link1.setQToFitTransform(state,
        Transform(Rotation(BodyRotationSequence, 0.3, XAxis, -0.45, YAxis,
                           0.25, ZAxis), Vec3(0.15,0.4,-0.2)));
    link1.setUToFitVelocity(state,
        SpatialVec(Vec3(0.7,-1.1,0.5), Vec3(-0.4,0.9,1.3)));
    link2.setQToFitRotation(state,
        Rotation(BodyRotationSequence, -0.55, XAxis, 0.35, YAxis, 0.6, ZAxis));
    link2.setUToFitAngularVelocity(state, Vec3(1.2,0.6,-0.85));
    system.realize(state, Stage::Velocity);

    const int nu = state.getNU();
    Matrix dTauDQ, dTauDU;
    matter.calcResidualForceDerivativesIgnoringConstraints(
        state, Vector_<SpatialVec>(), Vector(), dTauDQ, dTauDU);

    Vector tauCoriolis;
    matter.calcResidualForceIgnoringConstraints(
        state, Vector(), Vector_<SpatialVec>(), Vector(), tauCoriolis);

    const Vector& u = state.getU();
    Vector lhs(nu, Real(0));
    for (int i=0; i < nu; ++i)
        for (int j=0; j < nu; ++j) lhs[i] += dTauDU(i,j)*u[j];

    SimTK_TEST_EQ_TOL(lhs, 2.0*tauCoriolis, 1e-10);
}

// The derivatives describe relative motion only, so translating the whole model
// away from the Ground origin must not change them at all. This guards the
// property that every quantity in the implementation is referred to a body's
// own origin rather than to a distant common point.
void testInvarianceToModelPlacement() {
    Matrix dq[2], du[2];
    const Real offsets[2] = {0, 1000};

    for (int which=0; which < 2; ++which) {
        MultibodySystem system;
        SimbodyMatterSubsystem matter(system);
        Body::Rigid b1(cylinderProps(2.0, Vec3(0.03,-0.02,0.05), 0.3, 0.05));
        Body::Rigid b2(cylinderProps(1.4, Vec3(-0.02,0.04,-0.01), 0.25, 0.04));

        MobilizedBody::Free link1(matter.Ground(),
            Transform(Rotation(0.2, XAxis),
                      Vec3(0.05 + offsets[which], 0.1, -0.02)), b1,
            Transform(Rotation(0.15, YAxis), Vec3(0,0.3,0)));
        MobilizedBody::Pin link2(link1,
            Transform(Rotation(0.35, ZAxis), Vec3(0,-0.3,0.04)), b2,
            Transform(Rotation(0.1, XAxis), Vec3(0.02,0.25,0)));

        State state = system.realizeTopology();
        link1.setQToFitTransform(state,
            Transform(Rotation(BodyRotationSequence, 0.3, XAxis, -0.45, YAxis,
                               0.25, ZAxis), Vec3(0.15,0.4,-0.2)));
        link1.setUToFitVelocity(state,
            SpatialVec(Vec3(0.7,-1.1,0.5), Vec3(-0.4,0.9,1.3)));
        link2.setAngle(state, 0.65);  link2.setRate(state, -1.45);
        system.realize(state, Stage::Velocity);

        matter.calcResidualForceDerivativesIgnoringConstraints(
            state, Vector_<SpatialVec>(), makeUdot(state.getNU()),
            dq[which], du[which]);
    }

    SimTK_TEST_EQ_TOL(dq[0], dq[1], 1e-12);
    SimTK_TEST_EQ_TOL(du[0], du[1], 1e-12);
}

// Zero-length inputs must be treated as all-zero, and the applied body forces
// must not influence dtau/du at all: they enter the residual additively and
// carry no u dependence.
void testInputHandling() {
    MultibodySystem system;
    SimbodyMatterSubsystem matter(system);
    Body::Rigid body(cylinderProps(1.0, Vec3(0.02,0,0), 0.2, 0.04));
    MobilizedBody::Pin  p1(matter.Ground(), Transform(), body,
                           Transform(Vec3(0,0.2,0)));
    MobilizedBody::Ball b1(p1, Transform(), body, Transform(Vec3(0,0.2,0)));

    State state = system.realizeTopology();
    b1.setQToFitRotation(state, Rotation(0.4, XAxis));
    p1.setAngle(state, 0.3);
    state.updU() = 0.21;
    system.realize(state, Stage::Velocity);

    const int nu = state.getNU(), nb = matter.getNumBodies();

    Matrix dqZero, duZero;
    matter.calcResidualForceDerivativesIgnoringConstraints(
        state, Vector_<SpatialVec>(), Vector(), dqZero, duZero);
    SimTK_TEST(dqZero.nrow()==nu && dqZero.ncol()==nu);

    Matrix dqExplicit, duExplicit;
    matter.calcResidualForceDerivativesIgnoringConstraints(
        state, Vector_<SpatialVec>(nb, SpatialVec(Vec3(0),Vec3(0))),
        Vector(nu, Real(0)), dqExplicit, duExplicit);
    SimTK_TEST_EQ(dqZero, dqExplicit);
    SimTK_TEST_EQ(duZero, duExplicit);

    // Body forces change dtau/dq but must leave dtau/du untouched.
    Matrix dqF, duF;
    matter.calcResidualForceDerivativesIgnoringConstraints(
        state, makeBodyForces(nb), Vector(), dqF, duF);
    SimTK_TEST_EQ(duF, duZero);

    // Wrong-sized inputs must be rejected rather than silently misread.
    Matrix junkQ, junkU;
    SimTK_TEST_MUST_THROW(
        matter.calcResidualForceDerivativesIgnoringConstraints(
            state, Vector_<SpatialVec>(), Vector(nu+1, Real(0)),
            junkQ, junkU));
    SimTK_TEST_MUST_THROW(
        matter.calcResidualForceDerivativesIgnoringConstraints(
            state, Vector_<SpatialVec>(nb+1, SpatialVec(Vec3(0),Vec3(0))),
            Vector(), junkQ, junkU));
}

// A reversed mobilizer builds its H_FM by post-processing the as-defined H_MF
// with a q-dependent transform, so its partials are not the reversal of the
// as-defined ones. That is a separate rejection path from the q-dependent-H_FM
// one above, and it must also refuse rather than return a wrong answer.
void testReversedMobilizerIsRejected() {
    MultibodySystem system;
    SimbodyMatterSubsystem matter(system);
    Body::Rigid body(cylinderProps(1.0, Vec3(0.01,0,0), 0.2, 0.04));
    MobilizedBody::Pin pin(matter.Ground(), Transform(), body,
                           Transform(Vec3(0,0.2,0)));
    // Same Pin, but with q and u defined outboard-to-inboard.
    MobilizedBody::Pin rev(pin, Transform(), body, Transform(Vec3(0,0.2,0)),
                           MobilizedBody::Reverse);

    State state = system.realizeTopology();
    system.realize(state, Stage::Velocity);

    Matrix dq, du;
    SimTK_TEST_MUST_THROW(
        matter.calcResidualForceDerivativesIgnoringConstraints(
            state, Vector_<SpatialVec>(), Vector(), dq, du));
}

// Exercise every mobilizer type that claims to support the operator. Each gets
// a two-body model -- an inboard Pin so the mobilizer under test has a moving
// parent, then the mobilizer itself -- and is checked against central finite
// differences. This is what actually validates each mobilizer's dH_FM/dq and
// dHDot_FM/dq implementations; the formulas are short but easy to get subtly
// wrong, and several of the terms vanish on simpler models.
template <class MobodType>
static void checkMobilizer(const char* name, bool expectSupported = true) {
    MultibodySystem system;
    SimbodyMatterSubsystem matter(system);
    Body::Rigid body(cylinderProps(1.3, Vec3(0.03,-0.02,0.04), 0.22, 0.045));

    MobilizedBody::Pin inboard(matter.Ground(),
        Transform(Rotation(0.23, XAxis), Vec3(0.04,0.09,-0.03)), body,
        Transform(Rotation(0.17, YAxis), Vec3(0,0.26,0)));
    MobodType tested(inboard,
        Transform(Rotation(0.31, ZAxis), Vec3(0.01,-0.24,0.05)), body,
        Transform(Rotation(0.13, XAxis), Vec3(0.02,0.21,0)));

    State state = system.realizeTopology();
    // Realize once so Instance is valid, then set q. Changing q invalidates
    // Position and above but leaves Instance realized, which is what
    // normalizeQuaternions() needs.
    system.realize(state, Stage::Velocity);
    const int nq = state.getNQ(), nu = state.getNU();
    for (int i=0; i < nq; ++i) state.updQ()[i] = 0.21 - 0.09*i;
    matter.normalizeQuaternions(state);
    for (int i=0; i < nu; ++i) state.updU()[i] = -0.37 + 0.14*i;
    system.realize(state, Stage::Velocity);

    Matrix dq, du;
    if (!expectSupported) {
        SimTK_TEST_MUST_THROW(
            matter.calcResidualForceDerivativesIgnoringConstraints(
                state, Vector_<SpatialVec>(), Vector(), dq, du));
        return;
    }
    std::cout << "    " << name << " (nq=" << nq << ", nu=" << nu << ")\n";
    checkAgainstFiniteDifferences(system, state);
}

// Screw, Ellipsoid and CantileverFreeBeam take extra construction arguments, so
// they get their own setup rather than the uniform template above.
static void checkWithExtraArg(const char* name, int which) {
    MultibodySystem system;
    SimbodyMatterSubsystem matter(system);
    Body::Rigid body(cylinderProps(1.3, Vec3(0.03,-0.02,0.04), 0.22, 0.045));
    const Transform X_PF(Rotation(0.31, ZAxis), Vec3(0.01,-0.24,0.05));
    const Transform X_BM(Rotation(0.13, XAxis), Vec3(0.02,0.21,0));
    MobilizedBody::Pin inboard(matter.Ground(),
        Transform(Rotation(0.23, XAxis), Vec3(0.04,0.09,-0.03)), body,
        Transform(Rotation(0.17, YAxis), Vec3(0,0.26,0)));

    MobilizedBody tested;
    if (which == 0)
        tested = MobilizedBody::Screw(inboard, X_PF, body, X_BM, Real(0.37));
    else if (which == 1)   // deliberately unequal radii
        tested = MobilizedBody::Ellipsoid(inboard, X_PF, body, X_BM,
                                          Vec3(0.17, 0.23, 0.31));
    else
        tested = MobilizedBody::CantileverFreeBeam(inboard, X_PF, body, X_BM,
                                                   Real(0.42));

    State state = system.realizeTopology();
    system.realize(state, Stage::Velocity);
    for (int i=0; i < state.getNQ(); ++i) state.updQ()[i] = 0.21 - 0.09*i;
    matter.normalizeQuaternions(state);
    for (int i=0; i < state.getNU(); ++i) state.updU()[i] = -0.37 + 0.14*i;
    system.realize(state, Stage::Velocity);
    std::cout << "    " << name << " (nq=" << state.getNQ()
              << ", nu=" << state.getNU() << ")\n";
    checkAgainstFiniteDifferences(system, state);
}

void testAllSupportedMobilizers() {
    // H_FM constant in F, so every partial vanishes.
    checkMobilizer<MobilizedBody::Pin>              ("Pin");
    checkMobilizer<MobilizedBody::Slider>           ("Slider");
    checkMobilizer<MobilizedBody::Cylinder>         ("Cylinder");
    checkMobilizer<MobilizedBody::Planar>           ("Planar");
    checkMobilizer<MobilizedBody::Translation>      ("Translation");
    checkMobilizer<MobilizedBody::Ball>             ("Ball");
    checkMobilizer<MobilizedBody::Free>             ("Free");
    checkMobilizer<MobilizedBody::Weld>             ("Weld");
    checkWithExtraArg("Screw", 0);

    // H_FM depends on q, so these rely on both dH_FM/dq and dHDot_FM/dq; the
    // second is what makes the Coriolis term come out right.
    checkMobilizer<MobilizedBody::Universal>        ("Universal");
    checkMobilizer<MobilizedBody::LineOrientation>("LineOrientation");
    checkMobilizer<MobilizedBody::FreeLine>       ("FreeLine");
    checkMobilizer<MobilizedBody::Gimbal>           ("Gimbal");
    checkMobilizer<MobilizedBody::Bushing>          ("Bushing");
    checkWithExtraArg("Ellipsoid", 1);
    checkWithExtraArg("CantileverFreeBeam", 2);

    // HDot_FM depends on the linear part of the cross-mobilizer velocity too,
    // so these exercise the v_d path through both partials.
    checkMobilizer<MobilizedBody::SphericalCoords>  ("SphericalCoords");
    checkMobilizer<MobilizedBody::BendStretch>      ("BendStretch");
}

//==============================================================================
//                           CUSTOM MOBILIZERS
//==============================================================================

// A 2-DoF Custom mobilizer deliberately built so that H_FM really does depend
// on q: the first mode rotates about F's x axis and the second about M's y
// axis, which is exactly the Universal mobilizer's definition. Reimplementing
// it as a Custom lets us check the Custom partial interface against a mobilizer
// whose analytic answer we already trust.
class UniversalAsCustom : public MobilizedBody::Custom::Implementation {
public:
    explicit UniversalAsCustom(SimbodyMatterSubsystem& matter, bool withPartials)
    :   Implementation(matter, 2, 2, 0), m_withPartials(withPartials) {}

    Implementation* clone() const override
    {   return new UniversalAsCustom(*this); }

    Transform calcMobilizerTransformFromQ(const State&, int nq,
                                          const Real* q) const override
    {   return Transform(Rotation(BodyRotationSequence, q[0], XAxis,
                                                        q[1], YAxis)); }

    SpatialVec multiplyByHMatrix(const State& s, int nu,
                                 const Real* u) const override
    {   const Rotation R_FM = getMobilizerTransform(s).R();
        return SpatialVec(u[0]*Vec3(1,0,0) + u[1]*Vec3(R_FM.y()), Vec3(0)); }

    void multiplyByHTranspose(const State& s, const SpatialVec& F, int nu,
                              Real* f) const override
    {   const Rotation R_FM = getMobilizerTransform(s).R();
        f[0] = dot(Vec3(1,0,0),   F[0]);
        f[1] = dot(Vec3(R_FM.y()), F[0]); }

    SpatialVec multiplyByHDotMatrix(const State& s, int nu,
                                    const Real* u) const override
    {   const Rotation R_FM = getMobilizerTransform(s).R();
        const Vec3 w_FM = getMobilizerVelocity(s)[0];
        return SpatialVec(u[1]*(w_FM % Vec3(R_FM.y())), Vec3(0)); }

    void multiplyByHDotTranspose(const State& s, const SpatialVec& F, int nu,
                                 Real* f) const override
    {   const Rotation R_FM = getMobilizerTransform(s).R();
        const Vec3 w_FM = getMobilizerVelocity(s)[0];
        f[0] = 0;
        f[1] = dot(w_FM % Vec3(R_FM.y()), F[0]); }

    // dH/dq_d. Column 0 is fixed in F and contributes nothing; column 1 is
    // My_F, which rotates with the mobilizer.
    bool multiplyByHPartialMatrix(const State& s, int d, int nu, const Real* u,
                                  SpatialVec& out) const override
    {   if (!m_withPartials) return false;
        const Rotation R_FM = getMobilizerTransform(s).R();
        const Vec3 My_F = Vec3(R_FM.y());
        const Vec3 w_d  = (d==0) ? Vec3(1,0,0) : My_F;
        out = SpatialVec(u[1]*(w_d % My_F), Vec3(0));
        return true; }

    // dHDot/dq_d at fixed u. With w_FM = u0*x + u1*My_F and only column 1
    // varying, dw_FM/dq_d = u1*(w_d x My_F).
    bool multiplyByHDotPartialMatrix(const State& s, int d, int nu,
                                     const Real* u,
                                     SpatialVec& out) const override
    {   if (!m_withPartials) return false;
        const Rotation R_FM = getMobilizerTransform(s).R();
        const Vec3 My_F = Vec3(R_FM.y());
        const Vec3 w_FM = getMobilizerVelocity(s)[0];
        const Vec3 w_d  = (d==0) ? Vec3(1,0,0) : My_F;
        const Vec3 dMy  = w_d % My_F;
        // dw_FM/dq_d at fixed u; only the second column varies.
        const Vec3 dw   = dMy * getU(s)[1];
        out = SpatialVec(u[1]*(dw % My_F + w_FM % dMy), Vec3(0));
        return true; }

private:
    bool m_withPartials;
};

static void buildCustomSystem(MultibodySystem& system,
                              SimbodyMatterSubsystem& matter,
                              bool withPartials, State& state) {
    Body::Rigid body(cylinderProps(1.3, Vec3(0.03,-0.02,0.04), 0.22, 0.045));
    MobilizedBody::Pin inboard(matter.Ground(),
        Transform(Rotation(0.23, XAxis), Vec3(0.04,0.09,-0.03)), body,
        Transform(Rotation(0.17, YAxis), Vec3(0,0.26,0)));
    MobilizedBody::Custom tested(inboard,
        new UniversalAsCustom(matter, withPartials),
        Transform(Rotation(0.31, ZAxis), Vec3(0.01,-0.24,0.05)), body,
        Transform(Rotation(0.13, XAxis), Vec3(0.02,0.21,0)));

    state = system.realizeTopology();
    system.realize(state, Stage::Velocity);
    for (int i=0; i < state.getNQ(); ++i) state.updQ()[i] = 0.21 - 0.09*i;
    for (int i=0; i < state.getNU(); ++i) state.updU()[i] = -0.37 + 0.14*i;
    system.realize(state, Stage::Velocity);
}

// A Custom mobilizer that supplies both partials is supported, and its answer
// must match finite differences just like any built-in mobilizer.
void testCustomMobilizerWithPartials() {
    MultibodySystem system;
    SimbodyMatterSubsystem matter(system);
    State state;
    buildCustomSystem(system, matter, true, state);
    std::cout << "    Custom (with partials)\n";
    checkAgainstFiniteDifferences(system, state);
}

// A Custom mobilizer that does not supply them -- which is every existing user
// Implementation, since the new methods default to returning false -- must make
// the operator refuse rather than return a wrong answer.
void testCustomMobilizerWithoutPartials() {
    MultibodySystem system;
    SimbodyMatterSubsystem matter(system);
    State state;
    buildCustomSystem(system, matter, false, state);

    Matrix dq, du;
    SimTK_TEST_MUST_THROW(
        matter.calcResidualForceDerivativesIgnoringConstraints(
            state, Vector_<SpatialVec>(), Vector(), dq, du));
}

int main() {
    SimTK_START_TEST("TestInverseDynamicsDerivatives");
        SimTK_SUBTEST(testSinglePin);
        SimTK_SUBTEST(testSerialPinChain);
        SimTK_SUBTEST(testBranchedMultiDof);
        SimTK_SUBTEST(testTranslationOnly);
        SimTK_SUBTEST(testEulerSumRuleOnDTauDU);
        SimTK_SUBTEST(testInvarianceToModelPlacement);
        SimTK_SUBTEST(testInputHandling);
        SimTK_SUBTEST(testAllSupportedMobilizers);
        SimTK_SUBTEST(testCustomMobilizerWithPartials);
        SimTK_SUBTEST(testCustomMobilizerWithoutPartials);
        SimTK_SUBTEST(testReversedMobilizerIsRejected);
    SimTK_END_TEST();
}
