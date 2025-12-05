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
///////////////

// Utility for scaling the position component of a Transform by given scale
// factors.
Transform scaleTranslation(const Transform& X, const Vec3& scales) {
    Transform result = X;
    result.updP() = X.p().elementwiseMultiply(scales);
    return result;
}

// Pin mobilizer.
MobilizedBody addPinMobilizer(
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
MobilizedBody addEllipsoidMobilizer(
        MobilizedBody& parent,
        const Transform& X_PF, const Vec3& s_P,
        const Transform& X_BM, const Vec3& s_B) {
    const Vec3 radii = Vec3(0.1, 0.2, 0.3);
    Vec3 radiiScaled;
    for (int i = 0; i < 3; ++i) {
        const Vec3 f_i = Vec3(X_PF.R().col(i));
        radiiScaled[i] = radii[i] * s_P.elementwiseMultiply(f_i).norm();
    }
    Body::Rigid body(MassProperties(1.0, Vec3(0),
                     UnitInertia::ellipsoid(radii)));
    body.addDecoration(Transform(), DecorativeSphere(0.1));
    return MobilizedBody::Ellipsoid(parent,
                                    scaleTranslation(X_PF, s_P),
                                    body,
                                    scaleTranslation(X_BM, s_B),
                                    radiiScaled);
}

// Function-based mobilizer.
class LinearFunction : public Function {
    Real m, b;
public:
    LinearFunction(Real slope = 1.0, Real intercept = 0.0) : m(slope),
                                                             b(intercept) {}
    Real calcValue(const Vector& x) const override { return m*x[0] + b; }
    Real calcDerivative(const Array_<int>& dc, const Vector&) const override {
        return dc.size() == 1 ? m : 0.0;
    }
    int getArgumentSize() const override { return 1; }
    int getMaxDerivativeOrder() const override { return 10; }
};

MobilizedBody addFunctionBasedMobilizer(
        MobilizedBody& parent,
        const Transform& X_PF, const Vec3& s_P,
        const Transform& X_BM, const Vec3& s_B) {

    const std::vector<Vec3> axes = {
        Vec3(1,    0,      0),   // rotation axis 0
        Vec3(0,    1,      0),   // rotation axis 1
        Vec3(0,    0,      1),   // rotation axis 2
        Vec3(1,    0,      0),   // translation axis 0
        Vec3(0,    1,      0),   // translation axis 1
        Vec3(0,    0,      1)    // translation axis 2
    };

    std::vector<std::vector<int>> coordIndices;
    std::vector<const Function*> functions;
    for (int i = 0; i < 6; ++i) {
        double slope = 1.0;
        if (i > 2) {
            const Vec3 f_i = Vec3(X_PF.R().col(i-3));
            slope = s_P.elementwiseMultiply(f_i).norm();
        }
        coordIndices.push_back({i});
        functions.push_back(new LinearFunction(slope));
    }
    Body::Rigid body(MassProperties(1.0, Vec3(0), UnitInertia(1)));
    return MobilizedBody::FunctionBased(parent,
                                        scaleTranslation(X_PF, s_P),
                                        body,
                                        scaleTranslation(X_BM, s_B),
                                        6, functions, coordIndices, axes);
}

// Cantilever-free beam mobilizer.
MobilizedBody addCantileverFreeBeamMobilizer(
        MobilizedBody& parent,
        const Transform& X_PF, const Vec3& s_P,
        const Transform& X_BM, const Vec3& s_B) {
    const Real length = 1.23;
    const Vec3 f_z = Vec3(X_PF.R().col(2));
    const Real s_z = s_P.elementwiseMultiply(f_z).norm();
    const Real lengthScaled = length * s_z;
    Body::Rigid body(MassProperties(1.0, Vec3(0), UnitInertia(1)));
    return MobilizedBody::CantileverFreeBeam(
                parent,
                scaleTranslation(X_PF, s_P),
                body,
                scaleTranslation(X_BM, s_B),
                lengthScaled);
}

/////////////////////
// PENDULUM SYSTEM //
/////////////////////

// Creates a pendulum-like system which is a chain of bodies connected by the
// various mobilizers defined above. The system can be scaled passing a vector
// of scale factors, one set of scale factors for each mobilized body (including
// ground).
class PendulumSystem {
public:
    enum MobilizerType {Ground=0, Pin=1, Ellipsoid=2, FunctionBased=3,
        CantileverFreeBeam=4};

    PendulumSystem(const Vector_<Vec3>& scales) :
            m_matter(m_system), m_forces(m_system),
            m_gravity(m_forces, m_matter, -YAxis, 9.8) {

        MobilizedBody pin = addPinMobilizer(m_matter.Ground(),
                X_PF[Pin], scales[Ground],
                X_BM[Pin], scales[Pin]);

        MobilizedBody ellipsoid = addEllipsoidMobilizer(pin,
                X_PF[Ellipsoid], scales[Pin],
                X_BM[Ellipsoid], scales[Ellipsoid]);

        MobilizedBody functionBased = addFunctionBasedMobilizer(ellipsoid,
                X_PF[FunctionBased], scales[Ellipsoid],
                X_BM[FunctionBased], scales[FunctionBased]);

        addCantileverFreeBeamMobilizer(functionBased,
                X_PF[CantileverFreeBeam], scales[FunctionBased],
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

    MultibodySystem        m_system;
    SimbodyMatterSubsystem m_matter;
    GeneralForceSubsystem  m_forces;
    Force::Gravity         m_gravity;

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
        Transform(Rotation(BodyRotationSequence,  // [3] FunctionBased
                           Pi/6, YAxis,
                           Pi/4, ZAxis),
                  Vec3(-0.3, 0.4, -0.5)),
        Transform(Rotation(BodyRotationSequence,  // [4] CantileverFreeBeam
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
        Transform(Rotation(BodyRotationSequence,  // [3] FunctionBased
                           Pi/3, YAxis,
                           -Pi/5, ZAxis),
                  Vec3(0.13, -0.14, 0.15)),
        Transform(Rotation(BodyRotationSequence,  // [4] CantileverFreeBeam
                           -Pi/6, ZAxis,
                           Pi/4, XAxis),
                  Vec3(-0.16, 0.17, -0.18))};
};

Vector_<Vec3> getScales() {
    Vector_<Vec3> scales(5);
    scales[PendulumSystem::Ground]             = Vec3(1.0);
    scales[PendulumSystem::Pin]                = Vec3(1.5, 0.5, 2.0);
    scales[PendulumSystem::Ellipsoid]          = Vec3(2.0, 3.0, 4.0);
    scales[PendulumSystem::FunctionBased]      = Vec3(2.5);
    scales[PendulumSystem::CantileverFreeBeam] = Vec3(3.0, 4.0, 5.0);
    return scales;
}

Vector_<Vec3> getUnityScales() {
    Vector_<Vec3> scales(5, Vec3(1.0));
    return scales;
}

// Verify J(s)*u == J_scaled*u by comparing the unscaled system with scale
// factors against the geometrically-scaled system.
void testMultiplyByScaledSystemJacobian() {

    // Unscaled system.
    Vector_<Vec3> unityScales = getUnityScales();
    PendulumSystem system(unityScales);
    State state = system.m_system.realizeTopology();
    system.loadDefaultState(state);
    system.m_system.realize(state, Stage::Position);

    // Scaled system.
    Vector_<Vec3> scales = getScales();
    PendulumSystem scaledSystem(scales);
    State scaledState = scaledSystem.m_system.realizeTopology();
    scaledSystem.loadDefaultState(scaledState);
    scaledSystem.m_system.realize(scaledState, Stage::Position);

    // The result of multiplyByScaledSystemJacobian on the unscaled system
    // should match multiplyBySystemJacobian on the scaled geometry.
    Vector u = state.getU();
    Vector_<SpatialVec> Ju_unscaled;
    system.m_matter.multiplyByScaledSystemJacobian(
            state, scales, u, Ju_unscaled);

    Vector_<SpatialVec> Ju_scaled;
    scaledSystem.m_matter.multiplyBySystemJacobian(scaledState, u, Ju_scaled);

    SimTK_TEST_EQ(Ju_unscaled, Ju_scaled);
}

// Verify ~J(s)*F == ~J_scaled*F by comparing the unscaled system with
// scale factors against the geometrically-scaled system.
void testMultiplyByScaledSystemJacobianTranspose() {

    // Unscaled system.
    Vector_<Vec3> unityScales = getUnityScales();
    PendulumSystem system(unityScales);
    State state = system.m_system.realizeTopology();
    system.loadDefaultState(state);
    system.m_system.realize(state, Stage::Position);

    // Scaled system.
    Vector_<Vec3> scales = getScales();
    PendulumSystem scaledSystem(scales);
    State scaledState = scaledSystem.m_system.realizeTopology();
    scaledSystem.loadDefaultState(scaledState);
    scaledSystem.m_system.realize(scaledState, Stage::Position);

    // Create a spatial-force-like input vector.
    const int nb = system.m_matter.getNumBodies();
    Vector_<SpatialVec> F(nb);
    for (int b = 0; b < nb; ++b) {
        F[b] = SpatialVec(Vec3(0.1*(b+1), -0.2*(b+1), 0.3*(b+1)),
                          Vec3(-0.4*(b+1), 0.5*(b+1), -0.6*(b+1)));
    }

    // Calculate the result from the unscaled system with the scaled Jacobian
    // operator.
    Vector JtF_unscaled;
    system.m_matter.multiplyByScaledSystemJacobianTranspose(
            state, scales, F, JtF_unscaled);

    // Calculate the result from the scaled system with the unscaled Jacobian
    // operator.
    Vector JtF_scaled;
    scaledSystem.m_matter.multiplyBySystemJacobianTranspose(
            scaledState, F, JtF_scaled);

    // Check that the two operations match.
    SimTK_TEST_EQ(JtF_unscaled, JtF_scaled);
}

// Verify the scaled station and frame Jacobian operators against the
// geometrically-scaled system.
void testMultiplyByScaledStationAndFrameJacobians() {

    // Unscaled system.
    Vector_<Vec3> unityScales = getUnityScales();
    PendulumSystem system(unityScales);
    State state = system.m_system.realizeTopology();
    system.loadDefaultState(state);
    system.m_system.realize(state, Stage::Position);

    // Scaled system.
    Vector_<Vec3> scales = getScales();
    PendulumSystem scaledSystem(scales);
    State scaledState = scaledSystem.m_system.realizeTopology();
    scaledSystem.loadDefaultState(scaledState);
    scaledSystem.m_system.realize(scaledState, Stage::Position);

    const int nb = system.m_matter.getNumBodies();
    const int nt = nb - 1;
    const Vector u = state.getU();

    // Body stations.
    Array_<MobilizedBodyIndex> bodies;
    Array_<Vec3> stations;
    Array_<Vec3> scaledStations;
    for (int b = 1; b < nb; ++b) {
        bodies.push_back(MobilizedBodyIndex(b));
        stations.push_back(Vec3(0.1*b, -0.2*b, 0.3*b));
        scaledStations.push_back(
            stations.back().elementwiseMultiply(scales[b]));
    }

    // Station Jacobian operator.
    {
        Vector_<Vec3> JSu_unscaled, JSu_scaled;
        system.m_matter.multiplyByScaledStationJacobian(
                state, scales, bodies, stations, u, JSu_unscaled);
        scaledSystem.m_matter.multiplyByStationJacobian(
                scaledState, bodies, scaledStations, u, JSu_scaled);
        SimTK_TEST_EQ(JSu_unscaled, JSu_scaled);
    }

    // Station Jacobian transpose operator.
    {
        Vector_<Vec3> taskForces(nt);
        for (int b = 1; b < nb; ++b) {
            taskForces[b-1] = Vec3(0.1*b, -0.2*b, 0.3*b);
        }

        Vector f_unscaled, f_scaled;
        system.m_matter.multiplyByScaledStationJacobianTranspose(
                state, scales, bodies, stations, taskForces, f_unscaled);
        scaledSystem.m_matter.multiplyByStationJacobianTranspose(
                scaledState, bodies, scaledStations, taskForces, f_scaled);
        SimTK_TEST_EQ(f_unscaled, f_scaled);
    }

    // Frame Jacobian operator.
    {
        Vector_<SpatialVec> JFu_unscaled, JFu_scaled;
        system.m_matter.multiplyByScaledFrameJacobian(
                state, scales, bodies, stations, u, JFu_unscaled);
        scaledSystem.m_matter.multiplyByFrameJacobian(
                scaledState, bodies, scaledStations, u, JFu_scaled);
        SimTK_TEST_EQ(JFu_unscaled, JFu_scaled);
    }

    // Frame Jacobian transpose operator.
    {
        Vector_<SpatialVec> spatialForces(nt);
        for (int b = 1; b < nb; ++b) {
            spatialForces[b-1] = SpatialVec(Vec3(0.1*b, -0.2*b, 0.3*b),
                                            Vec3(-0.4*b, 0.5*b, -0.6*b));
        }

        Vector ff_unscaled, ff_scaled;
        system.m_matter.multiplyByScaledFrameJacobianTranspose(
                state, scales, bodies, stations, spatialForces, ff_unscaled);
        scaledSystem.m_matter.multiplyByFrameJacobianTranspose(
                scaledState, bodies, scaledStations, spatialForces, ff_scaled);
        SimTK_TEST_EQ(ff_unscaled, ff_scaled);
    }
}

// Verify JP = d(p_GB)/d(s) via finite differences.
void testMultiplyByPositionJacobianWrtBodyScales() {

    // Unscaled system.
    Vector_<Vec3> unityScales = getUnityScales();
    PendulumSystem system(unityScales);
    State state = system.m_system.realizeTopology();
    system.loadDefaultState(state);
    system.m_system.realize(state, Stage::Position);

    // Compare the analytic scale Jacobian against finite differences of the
    // body origins in ground, which are directly affected by the scale factors.
    const int nb = system.m_matter.getNumBodies();
    const Real h = 1e-5;
    for (int b = 0; b < nb; ++b) {
        for (int j = 0; j < 3; ++j) {
            Vector_<Vec3> s(nb, Vec3(0));
            s[b][j] = 1.0;

            // Analytic position Jacobian.
            Vector_<Vec3> JPs_analytic;
            system.m_matter.multiplyByPositionJacobianWrtBodyScales(
                state, s, JPs_analytic);

            // Scale Jacobian via finite differences.
            Vector_<Vec3> scales_pert = unityScales;
            scales_pert[b][j] += h;
            PendulumSystem pertSystem(scales_pert);
            State pertState = pertSystem.m_system.realizeTopology();
            pertSystem.loadDefaultState(pertState);
            pertSystem.m_system.realize(pertState, Stage::Position);

            for (int ib = 0; ib < nb; ++ib) {
                const Vec3 p0 = system.m_matter.getMobilizedBody(
                        MobilizedBodyIndex(ib)).getBodyTransform(state).p();
                const Vec3 p_pert = pertSystem.m_matter.getMobilizedBody(
                        MobilizedBodyIndex(ib)).getBodyTransform(pertState).p();
                SimTK_TEST_EQ_TOL(JPs_analytic[ib], (p_pert - p0) / h, 1e-4);
            }
        }
    }
}

// Verify ds = ~JP*dp via finite differences: build JP explicitly column by
// column, then compute ~JP*dp as a matrix-vector product and compare against
// multiplyByPositionJacobianWrtBodyScalesTranspose.
void testMultiplyByPositionJacobianWrtBodyScalesTranspose() {

    // Unscaled system.
    Vector_<Vec3> unityScales = getUnityScales();
    PendulumSystem system(unityScales);
    State state = system.m_system.realizeTopology();
    system.loadDefaultState(state);
    system.m_system.realize(state, Stage::Position);

    const int nb = system.m_matter.getNumBodies();
    const Real h = 1e-5;

    // Unperturbed body-origin positions in ground.
    Vector_<Vec3> p_B_0(nb);
    for (int b = 0; b < nb; ++b) {
        p_B_0[b] = system.m_matter.getMobilizedBody(
                MobilizedBodyIndex(b)).getBodyTransform(state).p();
    }

    // Build JP via finite differences.
    Matrix K(3*nb, 3*nb, 0.0);
    for (int jb = 0; jb < nb; ++jb) {
        for (int js = 0; js < 3; ++js) {

            // For this body and scale factor, perturb the system.
            Vector_<Vec3> perturbScales = unityScales;
            perturbScales[jb][js] += h;
            PendulumSystem perturbSystem(perturbScales);
            State pertState = perturbSystem.m_system.realizeTopology();
            perturbSystem.loadDefaultState(pertState);
            perturbSystem.m_system.realize(pertState, Stage::Position);

            // Compute the perturbed body origin positions in ground and fill in
            // the appropriate entries of JP.
            for (int ib = 0; ib < nb; ++ib) {
                const Vec3 p_B_pert = perturbSystem.m_matter.getMobilizedBody(
                        MobilizedBodyIndex(ib)).getBodyTransform(pertState).p();
                for (int is = 0; is < 3; ++is) {
                    K[ib*3 + is][jb*3 + js] =
                        (p_B_pert[is] - p_B_0[ib][is]) / h;
                }
            }
        }
    }

    // Input vector dp.
    Vector_<Vec3> dp(nb);
    for (int b = 0; b < nb; ++b) {
        dp[b] = Vec3(0.1*(b+1), -0.2*(b+1), 0.3*(b+1));
    }

    // Flattened dp.
    Vector dp_flat(3*nb);
    for (int b = 0; b < nb; ++b) {
        for (int i = 0; i < 3; ++i) {
            dp_flat[b*3 + i] = dp[b][i];
        }
    }

    // Compute ~JP * dp via the explicit finite-difference matrix.
    const Vector JPtp_fd = ~K * dp_flat;

    // Compute ~JP * dp via the analytic operator.
    Vector_<Vec3> JPtp_analytic;
    system.m_matter.multiplyByPositionJacobianWrtBodyScalesTranspose(
        state, dp, JPtp_analytic);

    // Compare.
    for (int b = 0; b < nb; ++b) {
        for (int i = 0; i < 3; ++i) {
            SimTK_TEST_EQ_TOL(JPtp_analytic[b][i], JPtp_fd[b*3 + i], 1e-4);
        }
    }
}

// Verify SimbodyMatterSubsystem::multiplyByStationJacobianWrtBodyScales via
// finite differences.
void testMultiplyByStationJacobianWrtBodyScales() {

    // Unscaled system.
    Vector_<Vec3> unityScales = getUnityScales();
    PendulumSystem system(unityScales);
    State state = system.m_system.realizeTopology();
    system.loadDefaultState(state);
    system.m_system.realize(state, Stage::Position);

    const int nb = system.m_matter.getNumBodies();
    const Real h = 1e-5;

    // Use a non-trivial station offset on each body.
    Array_<MobilizedBodyIndex> bodies;
    Array_<Vec3> stationsInB;
    for (int b = 1; b < nb; ++b) {
        bodies.push_back(MobilizedBodyIndex(b));
        stationsInB.push_back(Vec3(0.1*b, 0.2*b, 0.3*b));
    }
    const int nt = (int)bodies.size();

    // Compare the analytic station Jacobian against finite differences of the
    // body origins in ground, which are directly affected by the scale factors.
    for (int b = 0; b < nb; ++b) {
        for (int j = 0; j < 3; ++j) {
            Vector_<Vec3> s(nb, Vec3(0));
            s[b][j] = 1.0;

            // Analytic station Jacobian.
            Vector_<Vec3> JSs;
            system.m_matter.multiplyByStationJacobianWrtBodyScales(
                    state, bodies, stationsInB, s, JSs);

            // Create a new system perturbed in the scale factor for this body
            // and compute the perturbed
            Vector_<Vec3> perturbScales = unityScales;
            perturbScales[b][j] += h;
            PendulumSystem perturbSystem(perturbScales);
            State perturbState = perturbSystem.m_system.realizeTopology();
            perturbSystem.loadDefaultState(perturbState);
            perturbSystem.m_system.realize(perturbState, Stage::Position);

            // For each station task, compute the perturbed station position in
            // ground and finite-difference Jacobian and compare against the
            // analytic Jacobian.
            for (int task = 0; task < nt; ++task) {
                const MobilizedBodyIndex mobodx = bodies[task];
                const Vec3& p_BS = stationsInB[task];

                // Unscaled station in ground: p_GB + R_GB * (p_BS ⊙ s0)
                const Transform& X0 = system.m_matter.getMobilizedBody(mobodx)
                                                     .getBodyTransform(state);
                const Vec3 p_GS0 = X0.p() + X0.R() * p_BS;

                // Perturbed station in ground:
                // p_GB_pert + R_GB * (p_BS ⊙ s_pert)
                const Transform& Xp =
                    perturbSystem.m_matter.getMobilizedBody(mobodx)
                                          .getBodyTransform(perturbState);
                const Vec3 p_GS_pert = Xp.p() +
                    Xp.R() * p_BS.elementwiseMultiply(perturbScales[mobodx]);

                // Finite-difference Jacobian: (p_GS_pert - p_GS0) / h.
                const Vec3 JSs_fd = (p_GS_pert - p_GS0) / h;

                // Compare against the analytic Jacobian.
                SimTK_TEST_EQ_TOL(JSs[task], JSs_fd, 1e-4);
            }
        }
    }
}


// Verify JStp = ~JS*p_GS via finite differences: build JS explicitly column
// by column, then compute ~JS*p_GS as a matrix-vector product and compare
// against multiplyByStationJacobianWrtBodyScalesTranspose.
void testMultiplyByStationJacobianWrtBodyScalesTranspose() {

    // Unscaled system.
    Vector_<Vec3> unityScales = getUnityScales();
    PendulumSystem system(unityScales);
    State state = system.m_system.realizeTopology();
    system.loadDefaultState(state);
    system.m_system.realize(state, Stage::Position);

    const int nb = system.m_matter.getNumBodies();
    const Real h = 1e-5;

    // Use a non-trivial station offset on each non-ground body.
    Array_<MobilizedBodyIndex> bodies;
    Array_<Vec3> stationsInB;
    for (int b = 1; b < nb; ++b) {
        bodies.push_back(MobilizedBodyIndex(b));
        stationsInB.push_back(Vec3(0.1*b, 0.2*b, 0.3*b));
    }
    const int nt = (int)bodies.size();

    // Unperturbed station positions in ground.
    Vector_<Vec3> p_GS0(nt);
    for (int task = 0; task < nt; ++task) {
        const MobilizedBodyIndex mobodx = bodies[task];
        const Vec3& p_BS = stationsInB[task];
        const Transform& X0 = system.m_matter.getMobilizedBody(mobodx)
                                             .getBodyTransform(state);
        p_GS0[task] = X0.p() + X0.R() * p_BS;
    }

    // Build JS via finite differences. JS is (nt*3) x (nb*3): rows are
    // station position components, columns are scale factor components.
    Matrix KS(3*nt, 3*nb, 0.0);
    for (int jb = 0; jb < nb; ++jb) {
        for (int js = 0; js < 3; ++js) {
            Vector_<Vec3> perturbScales = unityScales;
            perturbScales[jb][js] += h;
            PendulumSystem perturbSystem(perturbScales);
            State perturbState = perturbSystem.m_system.realizeTopology();
            perturbSystem.loadDefaultState(perturbState);
            perturbSystem.m_system.realize(perturbState, Stage::Position);

            for (int task = 0; task < nt; ++task) {
                const MobilizedBodyIndex mobodx = bodies[task];
                const Vec3& p_BS = stationsInB[task];
                const Transform& Xp = perturbSystem.m_matter
                    .getMobilizedBody(mobodx).getBodyTransform(perturbState);
                // Include the contribution from the station offset, which also
                // scales with the body.
                const Vec3 p_GS_pert = Xp.p() +
                    Xp.R() * p_BS.elementwiseMultiply(perturbScales[mobodx]);
                for (int is = 0; is < 3; ++is) {
                    KS[task*3 + is][jb*3 + js] =
                        (p_GS_pert[is] - p_GS0[task][is]) / h;
                }
            }
        }
    }

    // Input station force vector p_GS.
    Vector_<Vec3> p_GS(nt);
    for (int task = 0; task < nt; ++task) {
        p_GS[task] = Vec3(0.1*(task+1), -0.2*(task+1), 0.3*(task+1));
    }

    // Flattened p_GS for the matrix-vector product.
    Vector dp_flat(3*nt);
    for (int task = 0; task < nt; ++task) {
        for (int i = 0; i < 3; ++i) {
            dp_flat[task*3 + i] = p_GS[task][i];
        }
    }

    // Compute ~KS * p_GS via the explicit finite-difference matrix.
    const Vector JStp_fd = ~KS * dp_flat;

    // Compute ~KS * p_GS via the analytic operator.
    Vector_<Vec3> JStp_analytic;
    system.m_matter.multiplyByStationJacobianWrtBodyScalesTranspose(
            state, bodies, stationsInB, p_GS, JStp_analytic);

    // Compare.
    for (int b = 0; b < nb; ++b) {
        for (int i = 0; i < 3; ++i) {
            SimTK_TEST_EQ_TOL(JStp_analytic[b][i], JStp_fd[b*3 + i], 1e-4);
        }
    }
}

// Verify calcScaledStationPosition against a directly-built scaled system.
// The unscaled state is realized at s=1; applying bodyScales via the Jacobian
// should match the station positions obtained by building
// PendulumSystem(bodyScales).
void testScaledStationPosition() {

    // Unscaled system.
    Vector_<Vec3> unityScales = getUnityScales();
    PendulumSystem system(unityScales);
    State state = system.m_system.realizeTopology();
    system.loadDefaultState(state);
    system.m_system.realize(state, Stage::Position);
    const int nb = system.m_matter.getNumBodies();

    // Use a non-trivial station offset on each body.
    Array_<MobilizedBodyIndex> bodies;
    Array_<Vec3> stationsInB;
    for (int b = 1; b < nb; ++b) {
        bodies.push_back(MobilizedBodyIndex(b));
        stationsInB.push_back(Vec3(0.1*b, 0.2*b, 0.3*b));
    }
    const int nt = (int)bodies.size();

    // Non-trivial scale factors.
    Vector_<Vec3> bodyScales = getScales();

    // Calculate the scaled station positions in ground using the operator with
    // the unscaled system.
    Vector_<Vec3> p_GS;
    system.m_matter.calcScaledStationPosition(
            state, bodyScales, bodies, stationsInB, p_GS);

    // Build a new system with the scale factors already applied.
    PendulumSystem scaledSystem(bodyScales);
    State scaledState = scaledSystem.m_system.realizeTopology();
    scaledSystem.loadDefaultState(scaledState);
    scaledSystem.m_system.realize(scaledState, Stage::Position);

    // Check that the station positions in the scaled system match the result
    // from calcScaledStationPosition on the unscaled system.
    for (int task = 0; task < nt; ++task) {
        const MobilizedBodyIndex mobodx = bodies[task];
        const Vec3& p_BS = stationsInB[task];
        const Transform& X = scaledSystem.m_matter.getMobilizedBody(
                mobodx).getBodyTransform(scaledState);
        // In the scaled system the station offset also scales with the body.
        const Vec3 p_GS_ref = X.p() +
            X.R() * p_BS.elementwiseMultiply(bodyScales[mobodx]);
        SimTK_TEST_EQ_TOL(p_GS[task], p_GS_ref, 1e-10);
    }
}

int main() {
    SimTK_START_TEST("TestScaledSystemJacobian");
        SimTK_SUBTEST(testMultiplyByScaledSystemJacobian);
        SimTK_SUBTEST(testMultiplyByScaledSystemJacobianTranspose);
        SimTK_SUBTEST(testMultiplyByScaledStationAndFrameJacobians);
        SimTK_SUBTEST(testMultiplyByPositionJacobianWrtBodyScales);
        SimTK_SUBTEST(testMultiplyByPositionJacobianWrtBodyScalesTranspose);
        SimTK_SUBTEST(testMultiplyByStationJacobianWrtBodyScales);
        SimTK_SUBTEST(testMultiplyByStationJacobianWrtBodyScalesTranspose);
        SimTK_SUBTEST(testScaledStationPosition);
    SimTK_END_TEST();
}
