/* -------------------------------------------------------------------------- *
 *                               Simbody(tm)                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the SimTK biosimulation toolkit originating from           *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org/home/simbody.  *
 *                                                                            *
 * Portions copyright (c) 2008-26 Stanford University and the Authors.        *
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

#include "SimTKsimbody.h"

using namespace SimTK;

////////////////
// MOBILIZERS //
////////////////

// Scale only the translational component of a Transform.
Transform scaleTranslation(const Transform& X, const Vec3& scales) {
    Transform result = X;
    result.updP() = X.p().elementwiseMultiply(scales);
    return result;
}

// Pin mobilizer.
MobilizedBody::Pin addPinMobilizer(
        MobilizedBody& parent,
        const Transform& X_PF, const Vec3& s_P,
        const Transform& X_BM, const Vec3& s_B) {
    Body::Rigid body(MassProperties(1.0, Vec3(0), UnitInertia(1)));
    return MobilizedBody::Pin(parent,
                              scaleTranslation(X_PF, s_P),
                              body,
                              scaleTranslation(X_BM, s_B));
}

// Ellipsoid mobilizer.
const Vec3 ellipsoidRadii(0.1, 0.2, 0.3);
MobilizedBody::Ellipsoid addEllipsoidMobilizer(
        MobilizedBody& parent,
        const Transform& X_PF, const Vec3& s_P,
        const Transform& X_BM, const Vec3& s_B) {
    Body::Rigid body(MassProperties(1.0, Vec3(0),
                     UnitInertia::ellipsoid(ellipsoidRadii)));
    body.addDecoration(Transform(), DecorativeSphere(0.1));
    return MobilizedBody::Ellipsoid(parent,
                                    scaleTranslation(X_PF, s_P),
                                    body,
                                    scaleTranslation(X_BM, s_B),
                                    ellipsoidRadii);
}

// Cantilever-free beam mobilizer. 
const Real cantileverFreeBeamLength = 1.23;
MobilizedBody::CantileverFreeBeam addCantileverFreeBeamMobilizer(
        MobilizedBody& parent,
        const Transform& X_PF, const Vec3& s_P,
        const Transform& X_BM, const Vec3& s_B) {
    Body::Rigid body(MassProperties(1.0, Vec3(0), UnitInertia(1)));
    return MobilizedBody::CantileverFreeBeam(
                parent,
                scaleTranslation(X_PF, s_P),
                body,
                scaleTranslation(X_BM, s_B),
                cantileverFreeBeamLength);
}

/////////////////////
// PENDULUM SYSTEM //
/////////////////////

// A chain of bodies connected by one each of Pin, Ellipsoid, and
// CantileverFreeBeam mobilizers. The constructor takes of set of per-body
// XYZ scale factors and converts them into mobilizer parameters: the inboard
// and outboard frames, ellipsoid radii, beam length, and function translation
// scales.
class PendulumSystem {
public:
    enum MobilizerType {Ground=0, Pin=1, Ellipsoid=2, CantileverFreeBeam=3};

    PendulumSystem(const Vector_<Vec3>& scales) :
            m_matter(m_system), m_forces(m_system),
            m_gravity(m_forces, m_matter, -YAxis, 9.8) {

        m_pin = addPinMobilizer(m_matter.Ground(),
                X_PF[Pin], scales[Ground],
                X_BM[Pin], scales[Pin]);

        m_ellipsoid = addEllipsoidMobilizer(m_pin,
                X_PF[Ellipsoid], scales[Pin],
                X_BM[Ellipsoid], scales[Ellipsoid]);

        m_cantileverFreeBeam = addCantileverFreeBeamMobilizer(m_ellipsoid,
                X_PF[CantileverFreeBeam], scales[Ellipsoid],
                X_BM[CantileverFreeBeam], scales[CantileverFreeBeam]);
    }

    void loadDefaultState(State& state) {
        for (int i = 0; i < state.getNQ(); ++i) {
            state.updQ()[i] = 0.1 * (i+1);
        }
        for (int i = 0; i < state.getNU(); ++i) {
            state.updU()[i] = 1.0 * (i+1);
        }
    }

    // Override every mobilizer's inboard and outboard frame for this State
    // based on a set of per-body XYZ scale factors. Mobilizer-specific
    // parameters (radii, length, translation scale) are independent of body
    // scales and are left at their topology defaults.
    void setParametersFromScales(State& state,
                                 const Vector_<Vec3>& scales) const {
        m_pin.setInboardFrame(state,
            scaleTranslation(X_PF[Pin], scales[Ground]));
        m_pin.setOutboardFrame(state,
            scaleTranslation(X_BM[Pin], scales[Pin]));

        m_ellipsoid.setInboardFrame(state,
            scaleTranslation(X_PF[Ellipsoid], scales[Pin]));
        m_ellipsoid.setOutboardFrame(state,
            scaleTranslation(X_BM[Ellipsoid], scales[Ellipsoid]));

        m_cantileverFreeBeam.setInboardFrame(state,
            scaleTranslation(X_PF[CantileverFreeBeam], scales[Ellipsoid]));
        m_cantileverFreeBeam.setOutboardFrame(state,
            scaleTranslation(X_BM[CantileverFreeBeam],
                             scales[CantileverFreeBeam]));
    }

    // Calculate how mobilizer inboard and outboard frames shift based on
    // changes in body scale factors.
    void calcMobilizerFrameShiftsFromBodyScaleDelta(
            const State& state,
            const Vector_<Vec3>& delta,
            Vector_<Vec3>& dp_PF, Vector_<Vec3>& dp_BM) const {
        const int nb = m_matter.getNumBodies();
        dp_PF.resize(nb); dp_PF = Vec3(0);
        dp_BM.resize(nb); dp_BM = Vec3(0);

        const MobilizedBody mobs[3] = { m_pin, m_ellipsoid,
                                        m_cantileverFreeBeam };
        for (int m = 0; m < 3; ++m) {
            const int parentIdx = (int)mobs[m].getParentMobilizedBody()
                                              .getMobilizedBodyIndex();
            const int thisIdx   = (int)mobs[m].getMobilizedBodyIndex();
            const Transform X_PF_state = mobs[m].getInboardFrame(state);
            const Transform X_BM_state = mobs[m].getOutboardFrame(state);

            for (int i = 0; i < 3; ++i)
                dp_PF[thisIdx][i] = X_PF_state.p()[i] * delta[parentIdx][i];
            for (int i = 0; i < 3; ++i)
                dp_BM[thisIdx][i] = X_BM_state.p()[i] * delta[thisIdx][i];
        }
    }

    MultibodySystem        m_system;
    SimbodyMatterSubsystem m_matter;
    GeneralForceSubsystem  m_forces;
    Force::Gravity         m_gravity;

    MobilizedBody::Pin                m_pin;
    MobilizedBody::Ellipsoid          m_ellipsoid;
    MobilizedBody::CantileverFreeBeam m_cantileverFreeBeam;

private:
    const Array_<Transform> X_PF = {
        Transform(),                              // [0] Ground
        Transform(Rotation(BodyRotationSequence,  // [1] Pin
                           Pi/8, XAxis,
                           Pi/7, YAxis),
                  Vec3(-0.4, 0.5, -0.6)),
        Transform(Rotation(BodyRotationSequence,  // [2] Ellipsoid
                           Pi/4, ZAxis,
                           Pi/5, XAxis),
                  Vec3(0.1, -0.2, 0.3)),
        Transform(Rotation(BodyRotationSequence,  // [3] CantileverFreeBeam
                           -Pi/3, XAxis,
                           Pi/5, ZAxis),
                  Vec3(0.6, -0.7, 0.8))};
    const Array_<Transform> X_BM = {
        Transform(),                              // [0] Ground
        Transform(Rotation(BodyRotationSequence,  // [1] Pin
                           Pi/5, ZAxis,
                           Pi/6, YAxis),
                  Vec3(-0.10, 0.11, -0.12)),
        Transform(Rotation(BodyRotationSequence,  // [2] Ellipsoid
                           -Pi/4, XAxis,
                           Pi/3, YAxis),
                  Vec3(0.7, -0.8, 0.9)),
        Transform(Rotation(BodyRotationSequence,  // [3] CantileverFreeBeam
                           -Pi/6, ZAxis,
                           Pi/4, XAxis),
                  Vec3(-0.16, 0.17, -0.18))};
};

/////////////
// HELPERS //
/////////////

Vector_<Vec3> getScales() {
    Vector_<Vec3> scales(4);
    scales[PendulumSystem::Ground]             = Vec3(1.0);
    scales[PendulumSystem::Pin]                = Vec3(1.5, 0.5, 2.0);
    scales[PendulumSystem::Ellipsoid]          = Vec3(2.0, 3.0, 4.0);
    scales[PendulumSystem::CantileverFreeBeam] = Vec3(3.0, 4.0, 5.0);
    return scales;
}

Vector_<Vec3> getUnityScales() {
    Vector_<Vec3> scales(5, Vec3(1.0));
    return scales;
}

struct ScaledFixture {
    Vector_<Vec3> scales = getScales();
    PendulumSystem unscaledSystem{getUnityScales()};
    PendulumSystem scaledSystem{scales};
    State unscaledState;
    State scaledState;

    ScaledFixture() {
        unscaledState = unscaledSystem.m_system.realizeTopology();
        unscaledSystem.loadDefaultState(unscaledState);
        unscaledSystem.setParametersFromScales(unscaledState, scales);
        unscaledSystem.m_system.realize(unscaledState, Stage::Position);

        scaledState = scaledSystem.m_system.realizeTopology();
        scaledSystem.loadDefaultState(scaledState);
        scaledSystem.m_system.realize(scaledState, Stage::Position);
    }
};

///////////
// TESTS //
///////////

// Verify J*u for both systems matches.
void testMultiplyByScaledSystemJacobian() {
    ScaledFixture f;
    const Vector u = f.unscaledState.getU();

    Vector_<SpatialVec> Ju_unscaled, Ju_scaled;
    f.unscaledSystem.m_matter.multiplyBySystemJacobian(
            f.unscaledState, u, Ju_unscaled);
    f.scaledSystem.m_matter.multiplyBySystemJacobian(
            f.scaledState, u, Ju_scaled);

    SimTK_TEST_EQ(Ju_unscaled, Ju_scaled);
}

// Verify ~J*F for both systems matches.
void testMultiplyByScaledSystemJacobianTranspose() {
    ScaledFixture f;
    const int nb = f.unscaledSystem.m_matter.getNumBodies();

    // Build a spatial-force-like input vector, one entry per body.
    Vector_<SpatialVec> F(nb);
    for (int b = 0; b < nb; ++b) {
        F[b] = SpatialVec(Vec3(0.1*(b+1), -0.2*(b+1), 0.3*(b+1)),
                          Vec3(-0.4*(b+1), 0.5*(b+1), -0.6*(b+1)));
    }

    Vector JtF_unscaled, JtF_scaled;
    f.unscaledSystem.m_matter.multiplyBySystemJacobianTranspose(
            f.unscaledState, F, JtF_unscaled);
    f.scaledSystem.m_matter.multiplyBySystemJacobianTranspose(
            f.scaledState, F, JtF_scaled);

    SimTK_TEST_EQ(JtF_unscaled, JtF_scaled);
}

// Verify the station and frame Jacobian operators match between the two
// systems. The two systems have equivalent geometry, so the same station
// offset (in body B) is the same physical point in both — we pass the same
// offsets to both Jacobian calls.
void testMultiplyByScaledStationAndFrameJacobians() {
    ScaledFixture f;
    const int nb = f.unscaledSystem.m_matter.getNumBodies();
    const int nt = nb - 1;
    const Vector u = f.unscaledState.getU();

    // One station offset per non-Ground body.
    Array_<MobilizedBodyIndex> bodies;
    Array_<Vec3> stations;
    for (int b = 1; b < nb; ++b) {
        bodies.push_back(MobilizedBodyIndex(b));
        stations.push_back(Vec3(0.1*b, -0.2*b, 0.3*b));
    }

    // Station Jacobian J_S * u.
    {
        Vector_<Vec3> JSu_unscaled, JSu_scaled;
        f.unscaledSystem.m_matter.multiplyByStationJacobian(
                f.unscaledState, bodies, stations, u, JSu_unscaled);
        f.scaledSystem.m_matter.multiplyByStationJacobian(
                f.scaledState, bodies, stations, u, JSu_scaled);
        SimTK_TEST_EQ(JSu_unscaled, JSu_scaled);
    }

    // Station Jacobian transpose ~J_S * f_S.
    {
        Vector_<Vec3> taskForces(nt);
        for (int b = 1; b < nb; ++b) {
            taskForces[b-1] = Vec3(0.1*b, -0.2*b, 0.3*b);
        }

        Vector f_unscaled, f_scaled;
        f.unscaledSystem.m_matter.multiplyByStationJacobianTranspose(
                f.unscaledState, bodies, stations, taskForces, f_unscaled);
        f.scaledSystem.m_matter.multiplyByStationJacobianTranspose(
                f.scaledState, bodies, stations, taskForces, f_scaled);
        SimTK_TEST_EQ(f_unscaled, f_scaled);
    }

    // Frame Jacobian J_F * u.
    {
        Vector_<SpatialVec> JFu_unscaled, JFu_scaled;
        f.unscaledSystem.m_matter.multiplyByFrameJacobian(
                f.unscaledState, bodies, stations, u, JFu_unscaled);
        f.scaledSystem.m_matter.multiplyByFrameJacobian(
                f.scaledState, bodies, stations, u, JFu_scaled);
        SimTK_TEST_EQ(JFu_unscaled, JFu_scaled);
    }

    // Frame Jacobian transpose ~J_F * F_F.
    {
        Vector_<SpatialVec> spatialForces(nt);
        for (int b = 1; b < nb; ++b) {
            spatialForces[b-1] = SpatialVec(Vec3(0.1*b, -0.2*b, 0.3*b),
                                            Vec3(-0.4*b, 0.5*b, -0.6*b));
        }

        Vector ff_unscaled, ff_scaled;
        f.unscaledSystem.m_matter.multiplyByFrameJacobianTranspose(
                f.unscaledState, bodies, stations, spatialForces, ff_unscaled);
        f.scaledSystem.m_matter.multiplyByFrameJacobianTranspose(
                f.scaledState, bodies, stations, spatialForces, ff_scaled);
        SimTK_TEST_EQ(ff_unscaled, ff_scaled);
    }
}

void testScaledStationPosition() {
    ScaledFixture f;
    const int nb = f.unscaledSystem.m_matter.getNumBodies();

    Array_<MobilizedBodyIndex> bodies;
    Array_<Vec3> stationsInB;
    for (int b = 1; b < nb; ++b) {
        bodies.push_back(MobilizedBodyIndex(b));
        stationsInB.push_back(Vec3(0.1*b, 0.2*b, 0.3*b));
    }
    const int nt = (int)bodies.size();

    for (int task = 0; task < nt; ++task) {
        const MobilizedBodyIndex mobodx = bodies[task];
        const Vec3& p_BS = stationsInB[task];

        const Transform& X_unscaled =
            f.unscaledSystem.m_matter.getMobilizedBody(mobodx)
                            .getBodyTransform(f.unscaledState);
        const Vec3 p_GS_unscaled = X_unscaled.p() + X_unscaled.R() * p_BS;

        const Transform& X_scaled =
            f.scaledSystem.m_matter.getMobilizedBody(mobodx)
                          .getBodyTransform(f.scaledState);
        const Vec3 p_GS_scaled = X_scaled.p() + X_scaled.R() * p_BS;

        SimTK_TEST_EQ_TOL(p_GS_unscaled, p_GS_scaled, 1e-10);
    }
}

// Forward inboard-frame: compare operator output against finite differences
// applied via setInboardFrame on every mobilizer simultaneously.
void testMultiplyByPositionJacobianWrtInboardFramePositions() {
    PendulumSystem sys(getUnityScales());
    State state = sys.m_system.realizeTopology();
    sys.loadDefaultState(state);
    sys.m_system.realize(state, Stage::Position);

    const int nb = sys.m_matter.getNumBodies();
    const Real h = 1e-5;
    const Real tol = 1e-4;

    Vector_<Vec3> delta(nb);
    for (int b = 0; b < nb; ++b) {
        delta[b] = Vec3(0.5*(b+1), -0.3*(b+1), 0.7*(b+1));
    }

    Vector_<Vec3> dp_PF, dp_BM;
    sys.calcMobilizerFrameShiftsFromBodyScaleDelta(state, delta, dp_PF, dp_BM);

    Vector_<Vec3> dp_GB_analytic;
    sys.m_matter.multiplyByPositionJacobianWrtInboardFramePositions(
            state, dp_PF, dp_GB_analytic);

    State pert = state;
    MobilizedBody mobs[3] = { sys.m_pin, sys.m_ellipsoid, 
                              sys.m_cantileverFreeBeam };
    for (int m = 0; m < 3; ++m) {
        const MobilizedBodyIndex bIdx = mobs[m].getMobilizedBodyIndex();
        Transform X_PF = mobs[m].getInboardFrame(pert);
        X_PF.updP() += h * dp_PF[bIdx];
        mobs[m].setInboardFrame(pert, X_PF);
    }
    sys.m_system.realize(pert, Stage::Position);

    for (int ib = 0; ib < nb; ++ib) {
        const Vec3 p0 = sys.m_matter.getMobilizedBody(MobilizedBodyIndex(ib))
                                    .getBodyTransform(state).p();
        const Vec3 p1 = sys.m_matter.getMobilizedBody(MobilizedBodyIndex(ib))
                                    .getBodyTransform(pert).p();
        SimTK_TEST_EQ_TOL(dp_GB_analytic[ib], (p1 - p0) / h, tol);
    }
}

// Transpose inboard-frame: verify <dp_GB, J * dp_PF> = <dp_PF, ~J * dp_GB>.
void testMultiplyByPositionJacobianWrtInboardFramePositionsTranspose() {
    PendulumSystem sys(getUnityScales());
    State state = sys.m_system.realizeTopology();
    sys.loadDefaultState(state);
    sys.m_system.realize(state, Stage::Position);

    const int nb = sys.m_matter.getNumBodies();

    Vector_<Vec3> dp_PF(nb), dp_GB_in(nb);
    for (int b = 0; b < nb; ++b) {
        dp_PF[b]    = Vec3( 0.1*(b+1), -0.2*(b+1),  0.3*(b+1));
        dp_GB_in[b] = Vec3(-0.5*(b+1),  0.7*(b+1), -0.9*(b+1));
    }
    dp_PF[0] = Vec3(0);  dp_GB_in[0] = Vec3(0);

    Vector_<Vec3> J_dp_PF, JT_dp_GB;
    sys.m_matter.multiplyByPositionJacobianWrtInboardFramePositions(
            state, dp_PF, J_dp_PF);
    sys.m_matter.multiplyByPositionJacobianWrtInboardFramePositionsTranspose(
            state, dp_GB_in, JT_dp_GB);

    Real lhs = 0, rhs = 0;
    for (int b = 0; b < nb; ++b) {
        lhs += dot(dp_GB_in[b], J_dp_PF[b]);
        rhs += dot(dp_PF[b],    JT_dp_GB[b]);
    }
    SimTK_TEST_EQ_TOL(lhs, rhs, 1e-10);
}

// Forward outboard-frame.
void testMultiplyByPositionJacobianWrtOutboardFramePositions() {
    PendulumSystem sys(getUnityScales());
    State state = sys.m_system.realizeTopology();
    sys.loadDefaultState(state);
    sys.m_system.realize(state, Stage::Position);

    const int nb = sys.m_matter.getNumBodies();
    const Real h = 1e-5;
    const Real tol = 1e-4;

    Vector_<Vec3> delta(nb);
    for (int b = 0; b < nb; ++b) {
        delta[b] = Vec3(0.5*(b+1), -0.3*(b+1), 0.7*(b+1));
    }

    Vector_<Vec3> dp_PF, dp_BM;
    sys.calcMobilizerFrameShiftsFromBodyScaleDelta(state, delta, dp_PF, dp_BM);

    Vector_<Vec3> dp_GB_analytic;
    sys.m_matter.multiplyByPositionJacobianWrtOutboardFramePositions(
            state, dp_BM, dp_GB_analytic);

    State pert = state;
    MobilizedBody mobs[3] = { sys.m_pin, sys.m_ellipsoid, 
                              sys.m_cantileverFreeBeam };
    for (int m = 0; m < 3; ++m) {
        const MobilizedBodyIndex bIdx = mobs[m].getMobilizedBodyIndex();
        Transform X_BM = mobs[m].getOutboardFrame(pert);
        X_BM.updP() += h * dp_BM[bIdx];
        mobs[m].setOutboardFrame(pert, X_BM);
    }
    sys.m_system.realize(pert, Stage::Position);

    for (int ib = 0; ib < nb; ++ib) {
        const Vec3 p0 = sys.m_matter.getMobilizedBody(MobilizedBodyIndex(ib))
                                    .getBodyTransform(state).p();
        const Vec3 p1 = sys.m_matter.getMobilizedBody(MobilizedBodyIndex(ib))
                                    .getBodyTransform(pert).p();
        SimTK_TEST_EQ_TOL(dp_GB_analytic[ib], (p1 - p0) / h, tol);
    }
}

// Transpose outboard-frame.
void testMultiplyByPositionJacobianWrtOutboardFramePositionsTranspose() {
    PendulumSystem sys(getUnityScales());
    State state = sys.m_system.realizeTopology();
    sys.loadDefaultState(state);
    sys.m_system.realize(state, Stage::Position);

    const int nb = sys.m_matter.getNumBodies();

    Vector_<Vec3> dp_BM(nb), dp_GB_in(nb);
    for (int b = 0; b < nb; ++b) {
        dp_BM[b]    = Vec3( 0.2*(b+1), -0.4*(b+1),  0.6*(b+1));
        dp_GB_in[b] = Vec3(-0.3*(b+1),  0.9*(b+1), -0.5*(b+1));
    }
    dp_BM[0] = Vec3(0);  dp_GB_in[0] = Vec3(0);

    Vector_<Vec3> J_dp_BM, JT_dp_GB;
    sys.m_matter.multiplyByPositionJacobianWrtOutboardFramePositions(
            state, dp_BM, J_dp_BM);
    sys.m_matter.multiplyByPositionJacobianWrtOutboardFramePositionsTranspose(
            state, dp_GB_in, JT_dp_GB);

    Real lhs = 0, rhs = 0;
    for (int b = 0; b < nb; ++b) {
        lhs += dot(dp_GB_in[b], J_dp_BM[b]);
        rhs += dot(dp_BM[b],    JT_dp_GB[b]);
    }
    SimTK_TEST_EQ_TOL(lhs, rhs, 1e-10);
}

// Forward Ellipsoid radii.
void testMultiplyByPositionJacobianWrtRadii() {
    PendulumSystem sys(getUnityScales());
    State state = sys.m_system.realizeTopology();
    sys.loadDefaultState(state);

    // Rotate the inboard frame away from its topology default so that the
    // Jacobian is only correct if R_PF is read from the State's Instance
    // variables rather than the topology-time transform.
    sys.m_ellipsoid.setInboardFrame(state,
        Transform(Rotation(BodyRotationSequence, Pi/7, XAxis, -Pi/9, ZAxis),
                  Vec3(0.05, -0.15, 0.25)));

    sys.m_system.realize(state, Stage::Position);

    const int nb = sys.m_matter.getNumBodies();
    const Real h = 1e-5;
    const Real tol = 1e-4;

    const Vec3 dr(0.13, -0.21, 0.34);

    const Mat33 J = sys.m_ellipsoid.calcPositionJacobianWrtRadii(state);
    Vector_<Vec3> dp_GB_analytic;
    sys.m_matter.multiplyByPositionJacobianWrtMobilizerTranslation(
            state, sys.m_ellipsoid.getMobilizedBodyIndex(),
            J * dr, dp_GB_analytic);

    State pert = state;
    sys.m_ellipsoid.setRadii(pert, sys.m_ellipsoid.getRadii(pert) + h * dr);
    sys.m_system.realize(pert, Stage::Position);

    for (int ib = 0; ib < nb; ++ib) {
        const Vec3 p0 = sys.m_matter.getMobilizedBody(MobilizedBodyIndex(ib))
                                    .getBodyTransform(state).p();
        const Vec3 p1 = sys.m_matter.getMobilizedBody(MobilizedBodyIndex(ib))
                                    .getBodyTransform(pert).p();
        SimTK_TEST_EQ_TOL(dp_GB_analytic[ib], (p1 - p0) / h, tol);
    }
}

// Transpose Ellipsoid radii.
void testMultiplyByPositionJacobianWrtRadiiTranspose() {
    PendulumSystem sys(getUnityScales());
    State state = sys.m_system.realizeTopology();
    sys.loadDefaultState(state);
    sys.m_system.realize(state, Stage::Position);

    const int nb = sys.m_matter.getNumBodies();

    const Vec3 dr(0.13, -0.21, 0.34);
    Vector_<Vec3> dp_GB_in(nb);
    for (int b = 0; b < nb; ++b)
        dp_GB_in[b] = Vec3(0.1*(b+1), -0.3*(b+1), 0.5*(b+1));
    dp_GB_in[0] = Vec3(0);

    const Mat33 J = sys.m_ellipsoid.calcPositionJacobianWrtRadii(state);
    Vector_<Vec3> J_dr;
    sys.m_matter.multiplyByPositionJacobianWrtMobilizerTranslation(
            state, sys.m_ellipsoid.getMobilizedBodyIndex(),
            J * dr, J_dr);
    const Vec3 sum = sys.m_matter
            .multiplyByPositionJacobianWrtMobilizerTranslationTranspose(
                    state, sys.m_ellipsoid.getMobilizedBodyIndex(),
                    dp_GB_in);
    const Vec3 JT_dp_GB = ~J * sum;

    Real lhs = 0;
    for (int b = 0; b < nb; ++b) lhs += dot(dp_GB_in[b], J_dr[b]);
    const Real rhs = dot(dr, JT_dp_GB);
    SimTK_TEST_EQ_TOL(lhs, rhs, 1e-10);
}

// Forward CantileverFreeBeam length.
void testMultiplyByPositionJacobianWrtLength() {
    PendulumSystem sys(getUnityScales());
    State state = sys.m_system.realizeTopology();
    sys.loadDefaultState(state);

    // See testMultiplyByPositionJacobianWrtRadii(): the inboard frame rotation
    // must come from the State, not from topology.
    sys.m_cantileverFreeBeam.setInboardFrame(state,
        Transform(Rotation(BodyRotationSequence, -Pi/8, YAxis, Pi/11, ZAxis),
                  Vec3(-0.25, 0.35, 0.45)));

    sys.m_system.realize(state, Stage::Position);

    const int nb = sys.m_matter.getNumBodies();
    const Real h = 1e-5;
    const Real tol = 1e-4;

    const Real dL = 0.42;

    const Vec3 J = sys.m_cantileverFreeBeam.calcPositionJacobianWrtLength(state);
    Vector_<Vec3> dp_GB_analytic;
    sys.m_matter.multiplyByPositionJacobianWrtMobilizerTranslation(
            state, sys.m_cantileverFreeBeam.getMobilizedBodyIndex(),
            J * dL, dp_GB_analytic);

    State pert = state;
    sys.m_cantileverFreeBeam.setLength(pert,
            sys.m_cantileverFreeBeam.getLength(pert) + h * dL);
    sys.m_system.realize(pert, Stage::Position);

    for (int ib = 0; ib < nb; ++ib) {
        const Vec3 p0 = sys.m_matter.getMobilizedBody(MobilizedBodyIndex(ib))
                                    .getBodyTransform(state).p();
        const Vec3 p1 = sys.m_matter.getMobilizedBody(MobilizedBodyIndex(ib))
                                    .getBodyTransform(pert).p();
        SimTK_TEST_EQ_TOL(dp_GB_analytic[ib], (p1 - p0) / h, tol);
    }
}

// Transpose CantileverFreeBeam length.
void testMultiplyByPositionJacobianWrtLengthTranspose() {
    PendulumSystem sys(getUnityScales());
    State state = sys.m_system.realizeTopology();
    sys.loadDefaultState(state);
    sys.m_system.realize(state, Stage::Position);

    const int nb = sys.m_matter.getNumBodies();

    const Real dL = 0.42;
    Vector_<Vec3> dp_GB_in(nb);
    for (int b = 0; b < nb; ++b)
        dp_GB_in[b] = Vec3(0.2*(b+1), -0.1*(b+1), 0.4*(b+1));
    dp_GB_in[0] = Vec3(0);

    const Vec3 J = sys.m_cantileverFreeBeam.calcPositionJacobianWrtLength(state);
    Vector_<Vec3> J_dL;
    sys.m_matter.multiplyByPositionJacobianWrtMobilizerTranslation(
            state, sys.m_cantileverFreeBeam.getMobilizedBodyIndex(),
            J * dL, J_dL);
    const Vec3 sum = sys.m_matter
            .multiplyByPositionJacobianWrtMobilizerTranslationTranspose(
                    state,
                    sys.m_cantileverFreeBeam.getMobilizedBodyIndex(),
                    dp_GB_in);
    const Real JT_dp_GB = dot(J, sum);

    Real lhs = 0;
    for (int b = 0; b < nb; ++b) lhs += dot(dp_GB_in[b], J_dL[b]);
    const Real rhs = dL * JT_dp_GB;
    SimTK_TEST_EQ_TOL(lhs, rhs, 1e-10);
}

// Use the transpose of mobilizer parameter Jacobians to assemble the gradient
// of a position error on a single station offset:
//
//   E(x) = (1/2) || p_GS(x) - p_target ||^2
//
// where x is a flat variables vector containing the body scales, beam length,
// and ellipsoid radii.
//
// The full analytic gradient is compared element-by-element against an
// independent finite-difference estimate that perturbs each variable one
// at a time through the mobilizer parameter setters.
void testPositionErrorGradientWrtCombinedVariables() {

    // Construct an unscaled pendulum system.
    PendulumSystem sys(getUnityScales());
    State state = sys.m_system.realizeTopology();
    sys.loadDefaultState(state);
    sys.m_system.realize(state, Stage::Position);
    const int nb = sys.m_matter.getNumBodies();

    // Position-error: the station on the end of the cantilever beam mobiilzer
    // should match the target.
    const MobilizedBodyIndex targetMobodIndx =
            sys.m_cantileverFreeBeam.getMobilizedBodyIndex();
    const Vec3 p_BS(0.1, 0.2, 0.3);
    const Vec3 p_target(1.0, 2.0, 3.0);
    auto computeError = [&](const State& s) -> Real {
        const Transform& X = sys.m_matter.getMobilizedBody(targetMobodIndx)
                                         .getBodyTransform(s);
        const Vec3 p_GS = X.p() + X.R() * p_BS;
        return 0.5 * (p_GS - p_target).normSqr();
    };

    // Unperturbed system error.
    const Real E0 = computeError(state);

    // Flat-variables layout.
    const int idxBodyScales = 0;
    const int idxLength     = 3 * nb;
    const int idxRadii      = idxLength + 1;
    const int nVars         = idxRadii  + 3;

    // Calculate the analytic gradient using the mobilizer Jacobian methods.
    // ---------------------------------------------------------------------
    // Build dE/dp_GB, the derivative of the position error cost with respect 
    // to body positions expressed in ground. The only non-zero element is the 
    // slot associated with the cantilever free beam mobilizer.
    Vector_<Vec3> dE_dp_GB(nb, Vec3(0));
    const Transform& X = sys.m_matter.getMobilizedBody(targetMobodIndx)
                                    .getBodyTransform(state);
    const Vec3 p_GS = X.p() + X.R() * p_BS;
    dE_dp_GB[targetMobodIndx] = p_GS - p_target;

    // Per-mobilizer frame-translation gradients. Chain rule with dE/dp_GB.
    Vector_<Vec3> dE_dp_PF, dE_dp_BM;
    sys.m_matter.multiplyByPositionJacobianWrtInboardFramePositionsTranspose(
            state, dE_dp_GB, dE_dp_PF);
    sys.m_matter.multiplyByPositionJacobianWrtOutboardFramePositionsTranspose(
            state, dE_dp_GB, dE_dp_BM);

    // Per-mobilizer rigid-body parameter gradients. Compose the matter-
    // subsystem subtree-sum operator with each mobilizer's local
    // Jacobian column:  dE/dParam = ~J_local * subtree-sum(dE/dp_GB).
    const Vec3 cfbSum = sys.m_matter
            .multiplyByPositionJacobianWrtMobilizerTranslationTranspose(
                    state,
                    sys.m_cantileverFreeBeam.getMobilizedBodyIndex(),
                    dE_dp_GB);
    const Real dE_dLength = dot(
            sys.m_cantileverFreeBeam.calcPositionJacobianWrtLength(state),
            cfbSum);

    const Vec3 ellSum = sys.m_matter
            .multiplyByPositionJacobianWrtMobilizerTranslationTranspose(
                    state, sys.m_ellipsoid.getMobilizedBodyIndex(),
                    dE_dp_GB);
    const Vec3 dE_dRadii =
            ~sys.m_ellipsoid.calcPositionJacobianWrtRadii(state) * ellSum;

    // Body-scale gradients. Use chaing rule to compute dE_dscales from dE_dp_PF
    // and dE_dp_BM.
    Vector dE_dscales(3 * nb, 0.0);
    const MobilizedBody mobs[3] = { sys.m_pin, sys.m_ellipsoid,
                                    sys.m_cantileverFreeBeam };
    for (int m = 0; m < 3; ++m) {
        const int parentIdx = (int)mobs[m].getParentMobilizedBody()
                                          .getMobilizedBodyIndex();
        const int thisIdx   = (int)mobs[m].getMobilizedBodyIndex();
        const Transform X_PF_state = mobs[m].getInboardFrame(state);
        const Transform X_BM_state = mobs[m].getOutboardFrame(state);
        for (int i = 0; i < 3; ++i) {
            dE_dscales[parentIdx*3 + i] +=
                    dE_dp_PF[thisIdx][i] * X_PF_state.p()[i];
            dE_dscales[thisIdx*3 + i]   +=
                    dE_dp_BM[thisIdx][i] * X_BM_state.p()[i];
        }
    }

    // Assemble the flat analytic gradient vector.
    Vector grad_analytic(nVars, 0.0);
    for (int i = 0; i < 3*nb; ++i) {
        grad_analytic[idxBodyScales + i] = dE_dscales[i];
    }
    grad_analytic[idxLength] = dE_dLength;
    for (int i = 0; i < 3; ++i) {
        grad_analytic[idxRadii + i] = dE_dRadii[i];
    }

    // Calculate the gradient via finite differences.
    // ----------------------------------------------
    Vector grad_fd(nVars, 0.0);
    const Real h   = 1e-5;
    const Real tol = 1e-4;

    // Body scales: perturb the scale factors and update the inboard and 
    // outboard frames using setParametersFromScales().
    for (int k = 0; k < nb; ++k) {
        for (int axis = 0; axis < 3; ++axis) {
            State pert = state;
            Vector_<Vec3> scales = getUnityScales();
            scales[k][axis] += h;
            sys.setParametersFromScales(pert, scales);
            sys.m_system.realize(pert, Stage::Position);
            grad_fd[idxBodyScales + k*3 + axis] =
                    (computeError(pert) - E0) / h;
        }
    }

    // CantileverFreeBeam length.
    {
        State pert = state;
        sys.m_cantileverFreeBeam.setLength(pert,
                sys.m_cantileverFreeBeam.getLength(pert) + h);
        sys.m_system.realize(pert, Stage::Position);
        grad_fd[idxLength] = (computeError(pert) - E0) / h;
    }

    // Ellipsoid radii.
    for (int i = 0; i < 3; ++i) {
        State pert = state;
        Vec3 r = sys.m_ellipsoid.getRadii(pert);
        r[i] += h;
        sys.m_ellipsoid.setRadii(pert, r);
        sys.m_system.realize(pert, Stage::Position);
        grad_fd[idxRadii + i] = (computeError(pert) - E0) / h;
    }

    // Compare analytic and finite-differenced gradients.
    // --------------------------------------------------
    for (int i = 0; i < nVars; ++i) {
        SimTK_TEST_EQ_TOL(grad_analytic[i], grad_fd[i], tol);
    }
}


int main() {
    SimTK_START_TEST("TestScaledSystemJacobian");
        SimTK_SUBTEST(testMultiplyByScaledSystemJacobian);
        SimTK_SUBTEST(testMultiplyByScaledSystemJacobianTranspose);
        SimTK_SUBTEST(testMultiplyByScaledStationAndFrameJacobians);
        SimTK_SUBTEST(testScaledStationPosition);
        SimTK_SUBTEST(testMultiplyByPositionJacobianWrtInboardFramePositions);
        SimTK_SUBTEST(
            testMultiplyByPositionJacobianWrtInboardFramePositionsTranspose);
        SimTK_SUBTEST(testMultiplyByPositionJacobianWrtOutboardFramePositions);
        SimTK_SUBTEST(
            testMultiplyByPositionJacobianWrtOutboardFramePositionsTranspose);
        SimTK_SUBTEST(testMultiplyByPositionJacobianWrtRadii);
        SimTK_SUBTEST(testMultiplyByPositionJacobianWrtRadiiTranspose);
        SimTK_SUBTEST(testMultiplyByPositionJacobianWrtLength);
        SimTK_SUBTEST(testMultiplyByPositionJacobianWrtLengthTranspose);
        SimTK_SUBTEST(testPositionErrorGradientWrtCombinedVariables);
    SimTK_END_TEST();
}
