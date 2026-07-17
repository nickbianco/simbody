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

// Verifies that the system, station, and frame Jacobian operators agree with
// the realized body kinematics under every Instance-stage State override that
// mobilizers expose. Builds a small two-body chain ending in the
// mobilizer-under-test, sets q/u to nontrivial values, applies an override
// via the State setter, re-realizes through Velocity, and asserts:
//
//   (1)  multiplyBySystemJacobian(state, u, V) matches getBodyVelocity per body
//   (2)  multiplyByStationJacobian(state, u, ...) matches the station velocity
//        computed via cross-product on the realized body velocity
//   (3)  multiplyByFrameJacobian(state, u, ...) matches the frame velocity
//        computed via cross-product on the realized body velocity
//   (4)  Forward/transpose duality: F · (J*u) == u · (~J*F) for arbitrary F, u
//
// Covers: Pin (X_PF, X_BM), Ellipsoid (radii, X_PF, X_BM), CantileverFreeBeam
// (length, X_PF, X_BM), FunctionBased (functions, X_PF, X_BM), and a minimal
// Custom (X_PF, X_BM).

#include "SimTKsimbody.h"

#include <cmath>

using namespace SimTK;

namespace {

const Real kJacTol = 1e-12;  // operator-vs-realized agreement
const Real kDualTol = 1e-11; // forward/transpose duality

// Build a mobility-space vector with nontrivial but reproducible entries.
Vector makeU(int nu) {
    Vector u(nu);
    for (int i = 0; i < nu; ++i)
        u[i] = 0.1 + 0.07 * (i + 1) - 0.03 * (i * i);
    return u;
}

// Build a body-spatial-force vector with nontrivial but reproducible entries.
Vector_<SpatialVec> makeF(int nb) {
    Vector_<SpatialVec> F(nb);
    for (int i = 0; i < nb; ++i) {
        F[i] = SpatialVec(Vec3(0.3 - 0.11 * i, 0.2 * i, -0.05 * (i + 1)),
                          Vec3(0.4 * (i + 1), -0.25 * i, 0.18 - 0.06 * i));
    }
    return F;
}

// Compute body velocities from the realized State cache. Requires Stage::Velocity.
Vector_<SpatialVec> bodyVelocitiesFromCache(const MultibodySystem& sys,
                                            const State& s) {
    const SimbodyMatterSubsystem& matter = sys.getMatterSubsystem();
    Vector_<SpatialVec> V(matter.getNumBodies());
    for (MobilizedBodyIndex i(0); i < matter.getNumBodies(); ++i)
        V[i] = matter.getMobilizedBody(i).getBodyVelocity(s);
    return V;
}

Vector_<SpatialVec> bodyVelocitiesFromJacobian(const MultibodySystem& sys,
                                               const State& s,
                                               const Vector& u) {
    const SimbodyMatterSubsystem& matter = sys.getMatterSubsystem();
    Vector_<SpatialVec> V(matter.getNumBodies());
    matter.multiplyBySystemJacobian(s, u, V);
    return V;
}

void requireBodyVelocitiesAgree(const MultibodySystem& sys,
                                State& s,
                                const Vector& u,
                                const char* label) {
    s.updU() = u;
    sys.realize(s, Stage::Velocity);
    const auto Vcache = bodyVelocitiesFromCache(sys, s);
    const auto Vjac   = bodyVelocitiesFromJacobian(sys, s, u);
    for (int i = 0; i < Vcache.size(); ++i) {
        for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 3; ++col) {
                const Real a = Vcache[i][row][col];
                const Real b = Vjac[i][row][col];
                if (std::abs(a - b) > kJacTol) {
                    std::cerr << "[" << label << "] body " << i
                              << " row " << row << " col " << col
                              << " : cache=" << a << " jac=" << b
                              << " diff=" << (a - b) << "\n";
                    SimTK_TEST(false);
                }
            }
        }
    }
}

// Station velocity computed from the realized body velocity:
//     v_GS = v_GB + w_GB x (R_GB * p_BS)
Vec3 stationVelocityFromCache(const MultibodySystem& sys, const State& s,
                              MobilizedBodyIndex body, const Vec3& p_BS) {
    const MobilizedBody& mb = sys.getMatterSubsystem().getMobilizedBody(body);
    const SpatialVec V_GB = mb.getBodyVelocity(s);
    const Rotation& R_GB = mb.getBodyTransform(s).R();
    return V_GB[1] + V_GB[0] % (R_GB * p_BS);
}

void requireStationJacobianAgrees(const MultibodySystem& sys, State& s,
                                  const Vector& u,
                                  MobilizedBodyIndex body, const Vec3& p_BS,
                                  const char* label) {
    s.updU() = u;
    sys.realize(s, Stage::Velocity);
    const SimbodyMatterSubsystem& matter = sys.getMatterSubsystem();
    const Vec3 vRef = stationVelocityFromCache(sys, s, body, p_BS);
    const Vec3 vJac = matter.multiplyByStationJacobian(s, body, p_BS, u);
    for (int k = 0; k < 3; ++k) {
        if (std::abs(vRef[k] - vJac[k]) > kJacTol) {
            std::cerr << "[" << label
                      << "] station Jacobian mismatch on body " << body
                      << " row " << k << " : ref=" << vRef[k]
                      << " jac=" << vJac[k] << " diff=" << (vRef[k] - vJac[k])
                      << "\n";
            SimTK_TEST(false);
        }
    }
}

// Frame velocity at a body-fixed frame Ai with origin Ao=p_BA:
//   angular: w_GB
//   linear : v_GB + w_GB x (R_GB * p_BA)
SpatialVec frameVelocityFromCache(const MultibodySystem& sys, const State& s,
                                  MobilizedBodyIndex body, const Vec3& p_BA) {
    const MobilizedBody& mb = sys.getMatterSubsystem().getMobilizedBody(body);
    const SpatialVec V_GB = mb.getBodyVelocity(s);
    const Rotation& R_GB = mb.getBodyTransform(s).R();
    return SpatialVec(V_GB[0], V_GB[1] + V_GB[0] % (R_GB * p_BA));
}

void requireFrameJacobianAgrees(const MultibodySystem& sys, State& s,
                                const Vector& u,
                                MobilizedBodyIndex body, const Vec3& p_BA,
                                const char* label) {
    s.updU() = u;
    sys.realize(s, Stage::Velocity);
    const SimbodyMatterSubsystem& matter = sys.getMatterSubsystem();
    const SpatialVec VRef = frameVelocityFromCache(sys, s, body, p_BA);
    const SpatialVec VJac = matter.multiplyByFrameJacobian(s, body, p_BA, u);
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 3; ++col) {
            const Real a = VRef[row][col], b = VJac[row][col];
            if (std::abs(a - b) > kJacTol) {
                std::cerr << "[" << label
                          << "] frame Jacobian mismatch on body " << body
                          << " row " << row << " col " << col
                          << " : ref=" << a << " jac=" << b
                          << " diff=" << (a - b) << "\n";
                SimTK_TEST(false);
            }
        }
    }
}

// F · (J*u) must equal u · (~J*F) for any F, u (operator duality).
void requireJacobianTransposeDuality(const MultibodySystem& sys, State& s,
                                     const char* label) {
    const SimbodyMatterSubsystem& matter = sys.getMatterSubsystem();
    sys.realize(s, Stage::Position);

    const int nu = s.getNU();
    const int nb = matter.getNumBodies();
    const Vector u = makeU(nu);
    const Vector_<SpatialVec> F = makeF(nb);

    Vector_<SpatialVec> Ju(nb);
    matter.multiplyBySystemJacobian(s, u, Ju);
    Vector JtF(nu);
    matter.multiplyBySystemJacobianTranspose(s, F, JtF);

    Real fDotJu = 0, uDotJtF = 0;
    for (int i = 0; i < nb; ++i) {
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 3; ++c)
                fDotJu += F[i][r][c] * Ju[i][r][c];
    }
    for (int i = 0; i < nu; ++i)
        uDotJtF += u[i] * JtF[i];

    if (std::abs(fDotJu - uDotJtF) > kDualTol) {
        std::cerr << "[" << label << "] system Jacobian transpose duality: "
                  << "F.(J u) = " << fDotJu << " vs u.(~J F) = " << uDotJtF
                  << " diff=" << (fDotJu - uDotJtF) << "\n";
        SimTK_TEST(false);
    }
}

// Run a single override scenario: apply, realize, and check all four
// consistency conditions. `body` and `p_offset` choose where to anchor the
// station/frame Jacobian probes.
void exerciseOverride(MultibodySystem& sys, State& s,
                      MobilizedBodyIndex body, const Vec3& p_offset,
                      const char* label) {
    const int nu = s.getNU();
    const Vector u = makeU(nu);
    requireBodyVelocitiesAgree(sys, s, u, label);
    requireStationJacobianAgrees(sys, s, u, body, p_offset, label);
    requireFrameJacobianAgrees(sys, s, u, body, p_offset, label);
    requireJacobianTransposeDuality(sys, s, label);
}

// Reusable nontrivial frame perturbations for X_PF / X_BM tests.
Transform makeFramePerturbation(int seed) {
    return Transform(Rotation(BodyRotationSequence,
                              0.13 + 0.07 * seed, XAxis,
                              -0.21 - 0.05 * seed, YAxis,
                              0.34 * (seed + 1), ZAxis),
                     Vec3(0.11 - 0.04 * seed,
                          -0.07 + 0.03 * seed,
                          0.08 * (seed + 1)));
}

// A minimal Custom mobilizer (single-DOF translation along F-frame x-axis).
class CustomSlider : public MobilizedBody::Custom::Implementation {
public:
    explicit CustomSlider(SimbodyMatterSubsystem& matter)
        : Implementation(matter, 1, 1, 0) {}
    Implementation* clone() const override { return new CustomSlider(*this); }
    Transform calcMobilizerTransformFromQ(const State&, int nq,
                                          const Real* q) const override {
        return Transform(Vec3(q[0], 0, 0));
    }
    SpatialVec multiplyByHMatrix(const State&, int nu,
                                 const Real* u) const override {
        return SpatialVec(Vec3(0), Vec3(u[0], 0, 0));
    }
    void multiplyByHTranspose(const State&, const SpatialVec& F, int nu,
                              Real* f) const override {
        f[0] = F[1][0];
    }
    SpatialVec multiplyByHDotMatrix(const State&, int,
                                    const Real*) const override {
        return SpatialVec(Vec3(0), Vec3(0));
    }
    void multiplyByHDotTranspose(const State&, const SpatialVec&, int,
                                 Real* f) const override {
        f[0] = 0;
    }
    void setQToFitTransform(const State&, const Transform& X_FM, int,
                            Real* q) const override {
        q[0] = X_FM.p()[0];
    }
    void setUToFitVelocity(const State&, const SpatialVec& V_FM, int,
                           Real* u) const override {
        u[0] = V_FM[1][0];
    }
};

// Linear function f(q) = m*q + b used by FunctionBased.
class LinearFunction : public Function {
    Real m, b;
public:
    LinearFunction(Real slope = 1.0, Real intercept = 0.0)
        : m(slope), b(intercept) {}
    Real calcValue(const Vector& x) const override { return m * x[0] + b; }
    Real calcDerivative(const Array_<int>& dc, const Vector&) const override {
        return dc.size() == 1 ? m : 0.0;
    }
    int getArgumentSize() const override { return 1; }
    int getMaxDerivativeOrder() const override { return 10; }
};

// Set a reproducible nonzero pose used by all subtests.
void setNonTrivialState(State& s) {
    Vector& q = s.updQ();
    for (int i = 0; i < q.size(); ++i)
        q[i] = 0.13 + 0.07 * (i + 1);
}

} // anonymous namespace


//==============================================================================
// SUBTESTS
//==============================================================================

void testPinFrameOverrides() {
    MultibodySystem sys;
    SimbodyMatterSubsystem matter(sys);
    GeneralForceSubsystem forces(sys);
    Body::Rigid body(MassProperties(1.0, Vec3(0), UnitInertia(0.1)));
    MobilizedBody::Pin pin1(matter.Ground(),
                            Transform(Rotation(0.2, ZAxis), Vec3(0.1, 0, 0)),
                            body, Transform());
    MobilizedBody::Pin pin2(pin1,
                            Transform(Rotation(0.25, YAxis), Vec3(0, 0.2, 0)),
                            body,
                            Transform(Rotation(), Vec3(0, 0.05, 0)));
    sys.realizeTopology();
    State s = sys.getDefaultState();
    setNonTrivialState(s);

    exerciseOverride(sys, s, pin2.getMobilizedBodyIndex(), Vec3(0.02, 0.03, 0.04),
                     "Pin/default");

    pin2.setInboardFrame(s, makeFramePerturbation(0));
    exerciseOverride(sys, s, pin2.getMobilizedBodyIndex(), Vec3(0.02, 0.03, 0.04),
                     "Pin/X_PF");

    pin2.setOutboardFrame(s, makeFramePerturbation(1));
    exerciseOverride(sys, s, pin2.getMobilizedBodyIndex(), Vec3(0.02, 0.03, 0.04),
                     "Pin/X_PF+X_BM");
}

void testEllipsoidStateOverrides() {
    MultibodySystem sys;
    SimbodyMatterSubsystem matter(sys);
    GeneralForceSubsystem forces(sys);
    Body::Rigid body(MassProperties(1.0, Vec3(0),
                                    UnitInertia::ellipsoid(Vec3(0.1))));
    MobilizedBody::Pin pin(matter.Ground(), Transform(), body, Transform());
    MobilizedBody::Ellipsoid ell(pin,
        Transform(Rotation(0.18, ZAxis), Vec3(0.1, 0, 0)),
        body,
        Transform(Rotation(), Vec3(0, 0.05, 0)),
        Vec3(0.2, 0.3, 0.4));
    sys.realizeTopology();
    State s = sys.getDefaultState();
    setNonTrivialState(s);

    exerciseOverride(sys, s, ell.getMobilizedBodyIndex(), Vec3(0.0, 0.0, 0.0),
                     "Ellipsoid/default");

    ell.setRadii(s, Vec3(0.5, 0.7, 0.6));
    exerciseOverride(sys, s, ell.getMobilizedBodyIndex(), Vec3(0.0, 0.0, 0.0),
                     "Ellipsoid/radii");

    ell.setInboardFrame(s, makeFramePerturbation(2));
    exerciseOverride(sys, s, ell.getMobilizedBodyIndex(), Vec3(0.0, 0.0, 0.0),
                     "Ellipsoid/radii+X_PF");

    ell.setOutboardFrame(s, makeFramePerturbation(3));
    exerciseOverride(sys, s, ell.getMobilizedBodyIndex(), Vec3(0.0, 0.0, 0.0),
                     "Ellipsoid/radii+X_PF+X_BM");
}

void testCantileverFreeBeamStateOverrides() {
    MultibodySystem sys;
    SimbodyMatterSubsystem matter(sys);
    GeneralForceSubsystem forces(sys);
    Body::Rigid body(MassProperties(1.0, Vec3(0), UnitInertia(0.1)));
    MobilizedBody::Pin pin(matter.Ground(), Transform(), body, Transform());
    MobilizedBody::CantileverFreeBeam cfb(pin,
        Transform(Rotation(0.15, ZAxis), Vec3(0.0, 0, 0.1)),
        body,
        Transform(Rotation(), Vec3(0, 0.05, 0)),
        0.5);
    sys.realizeTopology();
    State s = sys.getDefaultState();
    setNonTrivialState(s);

    exerciseOverride(sys, s, cfb.getMobilizedBodyIndex(), Vec3(0.0, 0.0, 0.2),
                     "CFB/default");

    cfb.setLength(s, 1.2);
    exerciseOverride(sys, s, cfb.getMobilizedBodyIndex(), Vec3(0.0, 0.0, 0.2),
                     "CFB/length");

    cfb.setInboardFrame(s, makeFramePerturbation(4));
    exerciseOverride(sys, s, cfb.getMobilizedBodyIndex(), Vec3(0.0, 0.0, 0.2),
                     "CFB/length+X_PF");

    cfb.setOutboardFrame(s, makeFramePerturbation(5));
    exerciseOverride(sys, s, cfb.getMobilizedBodyIndex(), Vec3(0.0, 0.0, 0.2),
                     "CFB/length+X_PF+X_BM");
}

void testFunctionBasedStateOverrides() {
    Array_<Array_<int>> coordIndices(6);
    for (int i = 0; i < 6; ++i)
        coordIndices[i].push_back(i);
    const std::vector<Vec3> axesVec = {
        Vec3(1,0,0), Vec3(0,1,0), Vec3(0,0,1),
        Vec3(1,0,0), Vec3(0,1,0), Vec3(0,0,1)};
    const Array_<Vec3> axes(axesVec);

    Array_<const Function*> fns(6);
    for (int i = 0; i < 6; ++i) fns[i] = new LinearFunction(1.0);

    MultibodySystem sys;
    SimbodyMatterSubsystem matter(sys);
    GeneralForceSubsystem forces(sys);
    Body::Rigid body(MassProperties(1.0, Vec3(0), UnitInertia(0.1)));
    MobilizedBody::FunctionBased fb(matter.Ground(), Transform(),
                                    body, Transform(),
                                    6, fns, coordIndices, axes);
    sys.realizeTopology();
    State s = sys.getDefaultState();
    setNonTrivialState(s);

    exerciseOverride(sys, s, fb.getMobilizedBodyIndex(), Vec3(0.04, 0.02, -0.01),
                     "FunctionBased/default");

    fb.setInboardFrame(s, makeFramePerturbation(6));
    exerciseOverride(sys, s, fb.getMobilizedBodyIndex(), Vec3(0.04, 0.02, -0.01),
                     "FunctionBased/X_PF");

    fb.setOutboardFrame(s, makeFramePerturbation(7));
    exerciseOverride(sys, s, fb.getMobilizedBodyIndex(), Vec3(0.04, 0.02, -0.01),
                     "FunctionBased/X_PF+X_BM");
}

void testCustomMobilizerFrameOverrides() {
    MultibodySystem sys;
    SimbodyMatterSubsystem matter(sys);
    GeneralForceSubsystem forces(sys);
    Body::Rigid body(MassProperties(1.0, Vec3(0), UnitInertia(0.1)));
    MobilizedBody::Pin pin(matter.Ground(), Transform(), body, Transform());
    MobilizedBody::Custom cm(pin,
        new CustomSlider(matter),
        Transform(Rotation(0.22, ZAxis), Vec3(0.05, 0, 0.05)),
        body,
        Transform(Rotation(), Vec3(0, 0.04, 0)));
    sys.realizeTopology();
    State s = sys.getDefaultState();
    setNonTrivialState(s);

    exerciseOverride(sys, s, cm.getMobilizedBodyIndex(), Vec3(0.01, 0.02, 0.03),
                     "Custom/default");

    cm.setInboardFrame(s, makeFramePerturbation(8));
    exerciseOverride(sys, s, cm.getMobilizedBodyIndex(), Vec3(0.01, 0.02, 0.03),
                     "Custom/X_PF");

    cm.setOutboardFrame(s, makeFramePerturbation(9));
    exerciseOverride(sys, s, cm.getMobilizedBodyIndex(), Vec3(0.01, 0.02, 0.03),
                     "Custom/X_PF+X_BM");
}

// Sanity: overriding X_PF must actually change the system Jacobian (i.e. the
// new geometry flows through realizePosition rather than being silently
// ignored). This complements exerciseOverride which only verifies internal
// consistency.
void testFrameOverrideActuallyChangesJacobian() {
    MultibodySystem sys;
    SimbodyMatterSubsystem matter(sys);
    GeneralForceSubsystem forces(sys);
    Body::Rigid body(MassProperties(1.0, Vec3(0), UnitInertia(0.1)));
    MobilizedBody::Pin pin(matter.Ground(), Transform(), body, Transform());
    sys.realizeTopology();
    State s = sys.getDefaultState();
    setNonTrivialState(s);

    Vector u = makeU(s.getNU());
    Vector_<SpatialVec> Vbefore(matter.getNumBodies());
    sys.realize(s, Stage::Position);
    matter.multiplyBySystemJacobian(s, u, Vbefore);

    pin.setInboardFrame(s, makeFramePerturbation(11));
    sys.realize(s, Stage::Position);

    Vector_<SpatialVec> Vafter(matter.getNumBodies());
    matter.multiplyBySystemJacobian(s, u, Vafter);

    Real maxDiff = 0;
    for (int i = 0; i < Vbefore.size(); ++i)
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 3; ++c)
                maxDiff = std::max(maxDiff,
                    std::abs(Vbefore[i][r][c] - Vafter[i][r][c]));
    SimTK_TEST(maxDiff > 1e-4);
}

int main() {
    SimTK_START_TEST("TestStateParameterizedJacobian");
        SimTK_SUBTEST(testPinFrameOverrides);
        SimTK_SUBTEST(testEllipsoidStateOverrides);
        SimTK_SUBTEST(testCantileverFreeBeamStateOverrides);
        SimTK_SUBTEST(testFunctionBasedStateOverrides);
        SimTK_SUBTEST(testCustomMobilizerFrameOverrides);
        SimTK_SUBTEST(testFrameOverrideActuallyChangesJacobian);
    SimTK_END_TEST();
}
