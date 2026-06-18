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

// Sanity-checks the State-parameterized mobilizer geometry against
// constructor-baked geometry across a chain of mobilizers. Each subtest
// builds two PendulumSystems: a "scaled" one whose mobilizer parameters
// are baked in at construction (via XYZ body scale factors that flow into
// X_PF translation, X_BM translation, Ellipsoid radii, FunctionBased
// translation function slopes, and CantileverFreeBeam length) and an
// "unscaled" one (built with unity scale factors) whose Instance-stage
// State variables are then overridden via the new setters
// (setInboardFrame, setOutboardFrame, setRadii, setTranslationScale,
// setLength) to reproduce the same effective geometry. The same
// (unscaled) Jacobian operator is then run on both systems and the
// outputs must match — that's the contract this test is checking.

#include "SimTKsimbody.h"

using namespace SimTK;


////////////////
// MOBILIZERS //
///////////////

// Scale only the translational component of a Transform; leave the rotation
// alone. Used to fold body-scale factors into mobilizer-frame placements.
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

// Compute the effective per-axis scale factor on the F frame given parent
// body scales s_P expressed in the parent frame P, projecting onto each F
// axis via the (unscaled) rotation R_PF.
//   s_F[i] = ||s_P ⊙ R_PF.col(i)||,  i = 0,1,2
// This is the natural mapping from body-frame XYZ scales into F-frame
// translation scales when X_PF.R() is non-identity.
Vec3 scaledParentInF(const Transform& X_PF, const Vec3& s_P) {
    Vec3 s_F;
    for (int i = 0; i < 3; ++i) {
        const Vec3 f_i = Vec3(X_PF.R().col(i));
        s_F[i] = s_P.elementwiseMultiply(f_i).norm();
    }
    return s_F;
}

// Ellipsoid mobilizer. The unscaled radii are baked here; under scaling
// each radius is multiplied by the projected parent scale on that F axis.
const Vec3 ellipsoidRadii(0.1, 0.2, 0.3);

MobilizedBody::Ellipsoid addEllipsoidMobilizer(
        MobilizedBody& parent,
        const Transform& X_PF, const Vec3& s_P,
        const Transform& X_BM, const Vec3& s_B) {
    const Vec3 s_F = scaledParentInF(X_PF, s_P);
    const Vec3 radiiScaled = ellipsoidRadii.elementwiseMultiply(s_F);
    Body::Rigid body(MassProperties(1.0, Vec3(0),
                     UnitInertia::ellipsoid(ellipsoidRadii)));
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

MobilizedBody::FunctionBased addFunctionBasedMobilizer(
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

    // Rotation functions stay unit-slope; translation functions get the
    // projected parent scale baked into their slopes. Unity scales produce
    // unit-slope translation functions, which is what the unscaled system
    // wants.
    const Vec3 s_F = scaledParentInF(X_PF, s_P);
    std::vector<std::vector<int>> coordIndices;
    std::vector<const Function*> functions;
    for (int i = 0; i < 6; ++i) {
        const Real slope = (i < 3) ? Real(1.0) : s_F[i-3];
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

// Cantilever-free beam mobilizer. The unscaled length is baked here; the
// scaled length is folded along the F-z component of the parent scales.
const Real cantileverFreeBeamLength = 1.23;

MobilizedBody::CantileverFreeBeam addCantileverFreeBeamMobilizer(
        MobilizedBody& parent,
        const Transform& X_PF, const Vec3& s_P,
        const Transform& X_BM, const Vec3& s_B) {
    const Vec3 s_F = scaledParentInF(X_PF, s_P);
    const Real lengthScaled = cantileverFreeBeamLength * s_F[2];
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

// A chain of bodies connected by one each of Pin, Ellipsoid, FunctionBased,
// and CantileverFreeBeam mobilizers. The constructor bakes a per-body Vec3
// of XYZ scale factors into the mobilizer parameters. Passing unity scales
// yields the unscaled reference geometry; passing non-trivial scales yields
// a geometrically distinct ("scaled") system. The applyScalesToState method
// goes the other direction: given an unscaled-baked system, it applies the
// equivalent scale via Instance-stage State overrides so the resulting
// kinematics match the directly-baked scaled system bit-for-bit.
class PendulumSystem {
public:
    enum MobilizerType {Ground=0, Pin=1, Ellipsoid=2, FunctionBased=3,
        CantileverFreeBeam=4};

    PendulumSystem(const Vector_<Vec3>& scales) :
            m_matter(m_system), m_forces(m_system),
            m_gravity(m_forces, m_matter, -YAxis, 9.8) {

        m_pin = addPinMobilizer(m_matter.Ground(),
                X_PF[Pin], scales[Ground],
                X_BM[Pin], scales[Pin]);

        m_ellipsoid = addEllipsoidMobilizer(m_pin,
                X_PF[Ellipsoid], scales[Pin],
                X_BM[Ellipsoid], scales[Ellipsoid]);

        m_functionBased = addFunctionBasedMobilizer(m_ellipsoid,
                X_PF[FunctionBased], scales[Ellipsoid],
                X_BM[FunctionBased], scales[FunctionBased]);

        m_cantileverFreeBeam = addCantileverFreeBeamMobilizer(m_functionBased,
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

    // Override each mobilizer's Instance-stage State so that an
    // unity-scale-baked PendulumSystem reproduces the kinematics of one
    // constructed directly with the given scales. The system must have
    // been built with unity scales for this mapping to be correct.
    //
    // For every mobilizer:
    //   X_PF.p() is scaled by the parent-body XYZ scale (s_P)
    //   X_BM.p() is scaled by this body's XYZ scale (s_B)
    // Mobilizer-specific:
    //   Ellipsoid radii are scaled by the projected parent scale in F
    //   FunctionBased translation outputs are scaled the same way
    //   CantileverFreeBeam length is scaled by the F-z parent-scale projection
    void applyScalesToState(State& state,
                            const Vector_<Vec3>& scales) const {
        // Pin: X_PF, X_BM only.
        m_pin.setInboardFrame(state,
            scaleTranslation(X_PF[Pin], scales[Ground]));
        m_pin.setOutboardFrame(state,
            scaleTranslation(X_BM[Pin], scales[Pin]));

        // Ellipsoid: X_PF, X_BM, radii.
        m_ellipsoid.setInboardFrame(state,
            scaleTranslation(X_PF[Ellipsoid], scales[Pin]));
        m_ellipsoid.setOutboardFrame(state,
            scaleTranslation(X_BM[Ellipsoid], scales[Ellipsoid]));
        const Vec3 s_F_ell = scaledParentInF(X_PF[Ellipsoid], scales[Pin]);
        m_ellipsoid.setRadii(state,
            ellipsoidRadii.elementwiseMultiply(s_F_ell));

        // FunctionBased: X_PF, X_BM, translation-output scale.
        m_functionBased.setInboardFrame(state,
            scaleTranslation(X_PF[FunctionBased], scales[Ellipsoid]));
        m_functionBased.setOutboardFrame(state,
            scaleTranslation(X_BM[FunctionBased], scales[FunctionBased]));
        m_functionBased.setTranslationScale(state,
            scaledParentInF(X_PF[FunctionBased], scales[Ellipsoid]));

        // CantileverFreeBeam: X_PF, X_BM, length.
        m_cantileverFreeBeam.setInboardFrame(state,
            scaleTranslation(X_PF[CantileverFreeBeam], scales[FunctionBased]));
        m_cantileverFreeBeam.setOutboardFrame(state,
            scaleTranslation(X_BM[CantileverFreeBeam],
                             scales[CantileverFreeBeam]));
        const Vec3 s_F_cfb =
            scaledParentInF(X_PF[CantileverFreeBeam], scales[FunctionBased]);
        m_cantileverFreeBeam.setLength(state,
            cantileverFreeBeamLength * s_F_cfb[2]);
    }

    MultibodySystem        m_system;
    SimbodyMatterSubsystem m_matter;
    GeneralForceSubsystem  m_forces;
    Force::Gravity         m_gravity;

    // Typed mobilizer handles so applyScalesToState can reach the
    // mobilizer-specific setters.
    MobilizedBody::Pin                m_pin;
    MobilizedBody::Ellipsoid          m_ellipsoid;
    MobilizedBody::FunctionBased      m_functionBased;
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

// Build the unscaled and scaled PendulumSystems and bring them to the same
// effective geometry — the unscaled one via applyScalesToState, the scaled
// one via constructor-baked scales. Used by every subtest below.
struct ScaledFixture {
    Vector_<Vec3> scales = getScales();
    PendulumSystem unscaledSystem{getUnityScales()};
    PendulumSystem scaledSystem{scales};
    State unscaledState;
    State scaledState;

    ScaledFixture() {
        unscaledState = unscaledSystem.m_system.realizeTopology();
        unscaledSystem.loadDefaultState(unscaledState);
        unscaledSystem.applyScalesToState(unscaledState, scales);
        unscaledSystem.m_system.realize(unscaledState, Stage::Position);

        scaledState = scaledSystem.m_system.realizeTopology();
        scaledSystem.loadDefaultState(scaledState);
        scaledSystem.m_system.realize(scaledState, Stage::Position);
    }
};

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

// // Verify JP = d(p_GB)/d(s) via finite differences.
// void testMultiplyByPositionJacobianWrtBodyScales() {

//     // Unscaled system.
//     Vector_<Vec3> unityScales = getUnityScales();
//     PendulumSystem system(unityScales);
//     State state = system.m_system.realizeTopology();
//     system.loadDefaultState(state);
//     system.m_system.realize(state, Stage::Position);

//     // Compare the analytic scale Jacobian against finite differences of the
//     // body origins in ground, which are directly affected by the scale factors.
//     const int nb = system.m_matter.getNumBodies();
//     const Real h = 1e-5;
//     for (int b = 0; b < nb; ++b) {
//         for (int j = 0; j < 3; ++j) {
//             Vector_<Vec3> s(nb, Vec3(0));
//             s[b][j] = 1.0;

//             // Analytic position Jacobian.
//             Vector_<Vec3> JPs_analytic;
//             system.m_matter.multiplyByPositionJacobianWrtBodyScales(
//                 state, s, JPs_analytic);

//             // Scale Jacobian via finite differences.
//             Vector_<Vec3> scales_pert = unityScales;
//             scales_pert[b][j] += h;
//             PendulumSystem pertSystem(scales_pert);
//             State pertState = pertSystem.m_system.realizeTopology();
//             pertSystem.loadDefaultState(pertState);
//             pertSystem.m_system.realize(pertState, Stage::Position);

//             for (int ib = 0; ib < nb; ++ib) {
//                 const Vec3 p0 = system.m_matter.getMobilizedBody(
//                         MobilizedBodyIndex(ib)).getBodyTransform(state).p();
//                 const Vec3 p_pert = pertSystem.m_matter.getMobilizedBody(
//                         MobilizedBodyIndex(ib)).getBodyTransform(pertState).p();
//                 SimTK_TEST_EQ_TOL(JPs_analytic[ib], (p_pert - p0) / h, 1e-4);
//             }
//         }
//     }
// }

// // Verify ds = ~JP*dp via finite differences: build JP explicitly column by
// // column, then compute ~JP*dp as a matrix-vector product and compare against
// // multiplyByPositionJacobianWrtBodyScalesTranspose.
// void testMultiplyByPositionJacobianWrtBodyScalesTranspose() {

//     // Unscaled system.
//     Vector_<Vec3> unityScales = getUnityScales();
//     PendulumSystem system(unityScales);
//     State state = system.m_system.realizeTopology();
//     system.loadDefaultState(state);
//     system.m_system.realize(state, Stage::Position);

//     const int nb = system.m_matter.getNumBodies();
//     const Real h = 1e-5;

//     // Unperturbed body-origin positions in ground.
//     Vector_<Vec3> p_B_0(nb);
//     for (int b = 0; b < nb; ++b) {
//         p_B_0[b] = system.m_matter.getMobilizedBody(
//                 MobilizedBodyIndex(b)).getBodyTransform(state).p();
//     }

//     // Build JP via finite differences.
//     Matrix K(3*nb, 3*nb, 0.0);
//     for (int jb = 0; jb < nb; ++jb) {
//         for (int js = 0; js < 3; ++js) {

//             // For this body and scale factor, perturb the system.
//             Vector_<Vec3> perturbScales = unityScales;
//             perturbScales[jb][js] += h;
//             PendulumSystem perturbSystem(perturbScales);
//             State pertState = perturbSystem.m_system.realizeTopology();
//             perturbSystem.loadDefaultState(pertState);
//             perturbSystem.m_system.realize(pertState, Stage::Position);

//             // Compute the perturbed body origin positions in ground and fill in
//             // the appropriate entries of JP.
//             for (int ib = 0; ib < nb; ++ib) {
//                 const Vec3 p_B_pert = perturbSystem.m_matter.getMobilizedBody(
//                         MobilizedBodyIndex(ib)).getBodyTransform(pertState).p();
//                 for (int is = 0; is < 3; ++is) {
//                     K[ib*3 + is][jb*3 + js] =
//                         (p_B_pert[is] - p_B_0[ib][is]) / h;
//                 }
//             }
//         }
//     }

//     // Input vector dp.
//     Vector_<Vec3> dp(nb);
//     for (int b = 0; b < nb; ++b) {
//         dp[b] = Vec3(0.1*(b+1), -0.2*(b+1), 0.3*(b+1));
//     }

//     // Flattened dp.
//     Vector dp_flat(3*nb);
//     for (int b = 0; b < nb; ++b) {
//         for (int i = 0; i < 3; ++i) {
//             dp_flat[b*3 + i] = dp[b][i];
//         }
//     }

//     // Compute ~JP * dp via the explicit finite-difference matrix.
//     const Vector JPtp_fd = ~K * dp_flat;

//     // Compute ~JP * dp via the analytic operator.
//     Vector_<Vec3> JPtp_analytic;
//     system.m_matter.multiplyByPositionJacobianWrtBodyScalesTranspose(
//         state, dp, JPtp_analytic);

//     // Compare.
//     for (int b = 0; b < nb; ++b) {
//         for (int i = 0; i < 3; ++i) {
//             SimTK_TEST_EQ_TOL(JPtp_analytic[b][i], JPtp_fd[b*3 + i], 1e-4);
//         }
//     }
// }

// // Verify SimbodyMatterSubsystem::multiplyByStationJacobianWrtBodyScales via
// // finite differences.
// void testMultiplyByStationJacobianWrtBodyScales() {

//     // Unscaled system.
//     Vector_<Vec3> unityScales = getUnityScales();
//     PendulumSystem system(unityScales);
//     State state = system.m_system.realizeTopology();
//     system.loadDefaultState(state);
//     system.m_system.realize(state, Stage::Position);

//     const int nb = system.m_matter.getNumBodies();
//     const Real h = 1e-5;

//     // Use a non-trivial station offset on each body.
//     Array_<MobilizedBodyIndex> bodies;
//     Array_<Vec3> stationsInB;
//     for (int b = 1; b < nb; ++b) {
//         bodies.push_back(MobilizedBodyIndex(b));
//         stationsInB.push_back(Vec3(0.1*b, 0.2*b, 0.3*b));
//     }
//     const int nt = (int)bodies.size();

//     // Compare the analytic station Jacobian against finite differences of the
//     // body origins in ground, which are directly affected by the scale factors.
//     for (int b = 0; b < nb; ++b) {
//         for (int j = 0; j < 3; ++j) {
//             Vector_<Vec3> s(nb, Vec3(0));
//             s[b][j] = 1.0;

//             // Analytic station Jacobian.
//             Vector_<Vec3> JSs;
//             system.m_matter.multiplyByStationJacobianWrtBodyScales(
//                     state, bodies, stationsInB, s, JSs);

//             // Create a new system perturbed in the scale factor for this body
//             // and compute the perturbed
//             Vector_<Vec3> perturbScales = unityScales;
//             perturbScales[b][j] += h;
//             PendulumSystem perturbSystem(perturbScales);
//             State perturbState = perturbSystem.m_system.realizeTopology();
//             perturbSystem.loadDefaultState(perturbState);
//             perturbSystem.m_system.realize(perturbState, Stage::Position);

//             // For each station task, compute the perturbed station position in
//             // ground and finite-difference Jacobian and compare against the
//             // analytic Jacobian.
//             for (int task = 0; task < nt; ++task) {
//                 const MobilizedBodyIndex mobodx = bodies[task];
//                 const Vec3& p_BS = stationsInB[task];

//                 // Unscaled station in ground: p_GB + R_GB * (p_BS ⊙ s0)
//                 const Transform& X0 = system.m_matter.getMobilizedBody(mobodx)
//                                                      .getBodyTransform(state);
//                 const Vec3 p_GS0 = X0.p() + X0.R() * p_BS;

//                 // Perturbed station in ground:
//                 // p_GB_pert + R_GB * (p_BS ⊙ s_pert)
//                 const Transform& Xp =
//                     perturbSystem.m_matter.getMobilizedBody(mobodx)
//                                           .getBodyTransform(perturbState);
//                 const Vec3 p_GS_pert = Xp.p() +
//                     Xp.R() * p_BS.elementwiseMultiply(perturbScales[mobodx]);

//                 // Finite-difference Jacobian: (p_GS_pert - p_GS0) / h.
//                 const Vec3 JSs_fd = (p_GS_pert - p_GS0) / h;

//                 // Compare against the analytic Jacobian.
//                 SimTK_TEST_EQ_TOL(JSs[task], JSs_fd, 1e-4);
//             }
//         }
//     }
// }


// // Verify JStp = ~JS*p_GS via finite differences: build JS explicitly column
// // by column, then compute ~JS*p_GS as a matrix-vector product and compare
// // against multiplyByStationJacobianWrtBodyScalesTranspose.
// void testMultiplyByStationJacobianWrtBodyScalesTranspose() {

//     // Unscaled system.
//     Vector_<Vec3> unityScales = getUnityScales();
//     PendulumSystem system(unityScales);
//     State state = system.m_system.realizeTopology();
//     system.loadDefaultState(state);
//     system.m_system.realize(state, Stage::Position);

//     const int nb = system.m_matter.getNumBodies();
//     const Real h = 1e-5;

//     // Use a non-trivial station offset on each non-ground body.
//     Array_<MobilizedBodyIndex> bodies;
//     Array_<Vec3> stationsInB;
//     for (int b = 1; b < nb; ++b) {
//         bodies.push_back(MobilizedBodyIndex(b));
//         stationsInB.push_back(Vec3(0.1*b, 0.2*b, 0.3*b));
//     }
//     const int nt = (int)bodies.size();

//     // Unperturbed station positions in ground.
//     Vector_<Vec3> p_GS0(nt);
//     for (int task = 0; task < nt; ++task) {
//         const MobilizedBodyIndex mobodx = bodies[task];
//         const Vec3& p_BS = stationsInB[task];
//         const Transform& X0 = system.m_matter.getMobilizedBody(mobodx)
//                                              .getBodyTransform(state);
//         p_GS0[task] = X0.p() + X0.R() * p_BS;
//     }

//     // Build JS via finite differences. JS is (nt*3) x (nb*3): rows are
//     // station position components, columns are scale factor components.
//     Matrix KS(3*nt, 3*nb, 0.0);
//     for (int jb = 0; jb < nb; ++jb) {
//         for (int js = 0; js < 3; ++js) {
//             Vector_<Vec3> perturbScales = unityScales;
//             perturbScales[jb][js] += h;
//             PendulumSystem perturbSystem(perturbScales);
//             State perturbState = perturbSystem.m_system.realizeTopology();
//             perturbSystem.loadDefaultState(perturbState);
//             perturbSystem.m_system.realize(perturbState, Stage::Position);

//             for (int task = 0; task < nt; ++task) {
//                 const MobilizedBodyIndex mobodx = bodies[task];
//                 const Vec3& p_BS = stationsInB[task];
//                 const Transform& Xp = perturbSystem.m_matter
//                     .getMobilizedBody(mobodx).getBodyTransform(perturbState);
//                 // Include the contribution from the station offset, which also
//                 // scales with the body.
//                 const Vec3 p_GS_pert = Xp.p() +
//                     Xp.R() * p_BS.elementwiseMultiply(perturbScales[mobodx]);
//                 for (int is = 0; is < 3; ++is) {
//                     KS[task*3 + is][jb*3 + js] =
//                         (p_GS_pert[is] - p_GS0[task][is]) / h;
//                 }
//             }
//         }
//     }

//     // Input station force vector p_GS.
//     Vector_<Vec3> p_GS(nt);
//     for (int task = 0; task < nt; ++task) {
//         p_GS[task] = Vec3(0.1*(task+1), -0.2*(task+1), 0.3*(task+1));
//     }

//     // Flattened p_GS for the matrix-vector product.
//     Vector dp_flat(3*nt);
//     for (int task = 0; task < nt; ++task) {
//         for (int i = 0; i < 3; ++i) {
//             dp_flat[task*3 + i] = p_GS[task][i];
//         }
//     }

//     // Compute ~KS * p_GS via the explicit finite-difference matrix.
//     const Vector JStp_fd = ~KS * dp_flat;

//     // Compute ~KS * p_GS via the analytic operator.
//     Vector_<Vec3> JStp_analytic;
//     system.m_matter.multiplyByStationJacobianWrtBodyScalesTranspose(
//             state, bodies, stationsInB, p_GS, JStp_analytic);

//     // Compare.
//     for (int b = 0; b < nb; ++b) {
//         for (int i = 0; i < 3; ++i) {
//             SimTK_TEST_EQ_TOL(JStp_analytic[b][i], JStp_fd[b*3 + i], 1e-4);
//         }
//     }
// }

// // Verify calcScaledStationPosition against a directly-built scaled system.
// // The unscaled state is realized at s=1; applying bodyScales via the Jacobian
// // should match the station positions obtained by building
// // PendulumSystem(bodyScales).
// void testScaledStationPosition() {

//     // Unscaled system.
//     Vector_<Vec3> unityScales = getUnityScales();
//     PendulumSystem system(unityScales);
//     State state = system.m_system.realizeTopology();
//     system.loadDefaultState(state);
//     system.m_system.realize(state, Stage::Position);
//     const int nb = system.m_matter.getNumBodies();

//     // Use a non-trivial station offset on each body.
//     Array_<MobilizedBodyIndex> bodies;
//     Array_<Vec3> stationsInB;
//     for (int b = 1; b < nb; ++b) {
//         bodies.push_back(MobilizedBodyIndex(b));
//         stationsInB.push_back(Vec3(0.1*b, 0.2*b, 0.3*b));
//     }
//     const int nt = (int)bodies.size();

//     // Non-trivial scale factors.
//     Vector_<Vec3> bodyScales = getScales();

//     // Calculate the scaled station positions in ground using the operator with
//     // the unscaled system.
//     Vector_<Vec3> p_GS;
//     system.m_matter.calcScaledStationPosition(
//             state, bodyScales, bodies, stationsInB, p_GS);

//     // Build a new system with the scale factors already applied.
//     PendulumSystem scaledSystem(bodyScales);
//     State scaledState = scaledSystem.m_system.realizeTopology();
//     scaledSystem.loadDefaultState(scaledState);
//     scaledSystem.m_system.realize(scaledState, Stage::Position);

//     // Check that the station positions in the scaled system match the result
//     // from calcScaledStationPosition on the unscaled system.
//     for (int task = 0; task < nt; ++task) {
//         const MobilizedBodyIndex mobodx = bodies[task];
//         const Vec3& p_BS = stationsInB[task];
//         const Transform& X = scaledSystem.m_matter.getMobilizedBody(
//                 mobodx).getBodyTransform(scaledState);
//         // In the scaled system the station offset also scales with the body.
//         const Vec3 p_GS_ref = X.p() +
//             X.R() * p_BS.elementwiseMultiply(bodyScales[mobodx]);
//         SimTK_TEST_EQ_TOL(p_GS[task], p_GS_ref, 1e-10);
//     }
// }

int main() {
    SimTK_START_TEST("TestScaledSystemJacobian");
        SimTK_SUBTEST(testMultiplyByScaledSystemJacobian);
        SimTK_SUBTEST(testMultiplyByScaledSystemJacobianTranspose);
        SimTK_SUBTEST(testMultiplyByScaledStationAndFrameJacobians);
        // SimTK_SUBTEST(testMultiplyByPositionJacobianWrtBodyScales);
        // SimTK_SUBTEST(testMultiplyByPositionJacobianWrtBodyScalesTranspose);
        // SimTK_SUBTEST(testMultiplyByStationJacobianWrtBodyScales);
        // SimTK_SUBTEST(testMultiplyByStationJacobianWrtBodyScalesTranspose);
        // SimTK_SUBTEST(testScaledStationPosition);
    SimTK_END_TEST();
}
