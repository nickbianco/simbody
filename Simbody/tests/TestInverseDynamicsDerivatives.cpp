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

The checks compare against central finite differences of
calcResidualForceIgnoringConstraints() itself, so the operator is held to the
same definition of the residual that Simbody already exposes. */

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

    Vector residual;
    Matrix dTdq, dTdu;
    matter.calcResidualForceDerivativesIgnoringConstraints(
            state, mobForces, bodyForces, udot,
            residual, dTdq, dTdu);
    SimTK_TEST(dTdq.nrow()==nu && dTdq.ncol()==nu);
    SimTK_TEST(dTdu.nrow()==nu && dTdu.ncol()==nu);
    SimTK_TEST_EQ(residual.size(), nu);

    // The residual that comes back for free must be exactly what the sibling
    // operator produces; it is built from the same composite force.
    Vector tau;
    matter.calcResidualForceIgnoringConstraints(state, mobForces, bodyForces,
                                                udot, tau);
    SimTK_TEST_EQ(residual, tau);
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
        SimTK_TEST_EQ_TOL(dTdq(j), (tp-tm)/(2*h), tol);

        State up = state, um = state;
        up.updU()[j] += h;  um.updU()[j] -= h;
        system.realize(up, Stage::Velocity);
        system.realize(um, Stage::Velocity);
        matter.calcResidualForceIgnoringConstraints(up, mobForces, bodyForces,
                                                    udot, tp);
        matter.calcResidualForceIgnoringConstraints(um, mobForces, bodyForces,
                                                    udot, tm);
        SimTK_TEST_EQ_TOL(dTdu(j), (tp-tm)/(2*h), tol);
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

int main() {
    SimTK_START_TEST("TestInverseDynamicsDerivatives");
        SimTK_SUBTEST(testSinglePin);
        SimTK_SUBTEST(testSerialPinChain);
    SimTK_END_TEST();
}
