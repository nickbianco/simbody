/* -------------------------------------------------------------------------- *
 *                               Simbody(tm)                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the SimTK biosimulation toolkit originating from           *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org/home/simbody.  *
 *                                                                            *
 * Portions copyright (c) 2025 Stanford University and the Authors.           *
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

/* Sandbox containing every calculation needed to perform inverse dynamics on a
double pendulum, i.e. to solve

        tau = M(q) udot + f_inertial(q,u) - f_applied

for the mobility forces tau. The calculations are grouped and named after the
corresponding RigidBodyNode/RigidBodyNodeSpec methods so they can be compared
side by side with Simbody's internals; only the Pin-specific pieces (calcX_FM,
calcAcrossJointVelocityJacobian, calcAcrossJointVelocityJacobianDot) are
specialized here. The result is checked against
SimbodyMatterSubsystem::calcResidualForceIgnoringConstraints(). */

#include "SimTKsimbody.h"

#include <cstdio>
#include <iostream>

using namespace SimTK;
using std::cout;
using std::endl;

static const Real Tol = 1e-10;

SpatialVec cross(const SpatialVec& v, const SpatialVec& u) {
    SpatialVec w;
    w[0] = v[0] % u[0];
    w[1] = v[1] % u[0] + v[0] % u[1];
}

SpatialVec crossStar(const SpatialVec& v, const SpatialVec& f) {
    SpatialVec c;
    c[0] = v[0] % f[0] + v[1] % f[1];
    c[1] = v[0] % f[1];
}

SpatialVec crossStarBar(const SpatialVec& f, const SpatialVec& v) {
    SpatialVec c;
    c[0] = -f[0] % v[0] - f[1] % v[1];
    c[1] = -f[1] % v[0];
}

//==============================================================================
//                              PENDULUM SYSTEM
//==============================================================================
// Two Pin-jointed links. The second joint axis is tilted out of the first
// joint's plane and each mass center is offset from its body origin so that no
// term in the algebra can vanish by accident.
class PendulumSystem {
public:
    static const Real L1, Mass1, Radius1;
    static const Real L2, Mass2, Radius2;
    static const Vec3 Com1, Com2;
    static const Real Tilt2;
    static const Vec3 GravityVec;

    PendulumSystem() : m_matter(m_system) {
        const Inertia Ic1 = Mass1 * Inertia::cylinderAlongY(Radius1, L1/2);
        const Inertia Ic2 = Mass2 * Inertia::cylinderAlongY(Radius2, L2/2);

        Body::Rigid body1(MassProperties(Mass1, Com1,
                                         Ic1.shiftFromMassCenter(-Com1, Mass1)));
        Body::Rigid body2(MassProperties(Mass2, Com2,
                                         Ic2.shiftFromMassCenter(-Com2, Mass2)));

        m_link1 = MobilizedBody::Pin(m_matter.Ground(),
                                     Transform(Rotation(), Vec3(0)),
                                     body1,
                                     Transform(Rotation(), Vec3(0, L1/2, 0)));

        m_link2 = MobilizedBody::Pin(m_link1,
                                     Transform(Rotation(Tilt2, XAxis),
                                               Vec3(0, -L1/2, 0)),
                                     body2,
                                     Transform(Rotation(), Vec3(0, L2/2, 0)));

        m_state = m_system.realizeTopology();
    }

    void setState(Real q1, Real q2, Real u1, Real u2) {
        m_link1.setAngle(m_state, q1);  m_link1.setRate(m_state, u1);
        m_link2.setAngle(m_state, q2);  m_link2.setRate(m_state, u2);
        m_system.realize(m_state, Stage::Velocity);
    }

    // One scalar per mobility, and one spatial force per body (moment about
    // Bo, force at Bo, expressed in Ground) with Ground as body zero.
    void calcAppliedForces(Real joint1Torque, const Vec3& tipForce_G,
                           Vector& mobilityForces,
                           Vector_<SpatialVec>& bodyForces) const {
        mobilityForces.resize(getNumMobilities());
        mobilityForces.setToZero();
        mobilityForces[m_link1.getFirstUIndex(m_state)] = joint1Torque;

        bodyForces.resize(getNumBodies());
        bodyForces.setTo(SpatialVec(Vec3(0), Vec3(0)));

        for (MobilizedBodyIndex mbx(1); mbx < getNumBodies(); ++mbx) {
            const MobilizedBody& mobod = m_matter.getMobilizedBody(mbx);
            const MassProperties& mp = mobod.getBodyMassProperties(m_state);
            const Vec3 p_BoBc_G =
                mobod.getBodyRotation(m_state) * mp.getMassCenter();
            const Vec3 fGrav_G = mp.getMass() * GravityVec;
            bodyForces[mbx] += SpatialVec(p_BoBc_G % fGrav_G, fGrav_G);
        }

        const Vec3 p_BoS_G =
            m_link2.getBodyRotation(m_state) * Vec3(0, -L2/2, 0);
        bodyForces[m_link2.getMobilizedBodyIndex()] +=
            SpatialVec(p_BoS_G % tipForce_G, tipForce_G);
    }

    Vector calcUDot(Real udot1, Real udot2) const {
        Vector udot(getNumMobilities(), Real(0));
        udot[m_link1.getFirstUIndex(m_state)] = udot1;
        udot[m_link2.getFirstUIndex(m_state)] = udot2;
        return udot;
    }

    const MultibodySystem&        getSystem() const {return m_system;}
    const SimbodyMatterSubsystem& getMatter() const {return m_matter;}
    const State&                  getState()  const {return m_state;}
    int getNumBodies()      const {return m_matter.getNumBodies();}
    int getNumMobilities()  const {return m_matter.getNumMobilities();}

private:
    MultibodySystem        m_system;
    SimbodyMatterSubsystem m_matter;
    MobilizedBody::Pin     m_link1, m_link2;
    State                  m_state;
};

const Real PendulumSystem::L1 = 0.8, PendulumSystem::Mass1 = 2.0,
           PendulumSystem::Radius1 = 0.03;
const Real PendulumSystem::L2 = 0.6, PendulumSystem::Mass2 = 1.0,
           PendulumSystem::Radius2 = 0.025;
const Vec3 PendulumSystem::Com1(0, -0.02, 0.03);
const Vec3 PendulumSystem::Com2(0,  0.03,-0.01);
const Real PendulumSystem::Tilt2 = 20 * Pi/180;
const Vec3 PendulumSystem::GravityVec(0, -9.81, 0);


//==============================================================================
//                              INVERSE DYNAMICS
//==============================================================================
// Reimplementation of Simbody's O(n) inverse dynamics for a chain of Pin
// mobilizers, using the same decomposition and method names as
// RigidBodyNode/RigidBodyNodeSpec. Each mobilizer has dof==1, so the hinge
// matrices H are single SpatialVec columns.
class InverseDynamics {
public:
    explicit InverseDynamics(const PendulumSystem& pendulum)
    :   m_matter(pendulum.getMatter()), m_cache(pendulum.getNumBodies()) {
        const State& state = pendulum.getState();
        for (MobilizedBodyIndex mbx(0); mbx < pendulum.getNumBodies(); ++mbx) {
            const MobilizedBody& mobod = m_matter.getMobilizedBody(mbx);
            MobodCache& mc = m_cache[mbx];
            if (mbx == 0) { mc.X_GB = Transform(); continue; }
            SimTK_ERRCHK1_ALWAYS(MobilizedBody::Pin::isInstanceOf(mobod),
                "InverseDynamics::InverseDynamics()",
                "Mobilized body %d is not a Pin.", (int)mbx);
            mc.parent = mobod.getParentMobilizedBody().getMobilizedBodyIndex();
            mc.X_PF   = mobod.getInboardFrame(state);
            mc.X_MB   = ~mobod.getOutboardFrame(state);
            const MassProperties& mp = mobod.getBodyMassProperties(state);
            mc.mass             = mp.getMass();
            mc.com_B            = mp.getMassCenter();
            mc.unitInertia_Bo_B = mp.getUnitInertia();
            mc.qx = mobod.getFirstQIndex(state);
            mc.ux = mobod.getFirstUIndex(state);
        }
    }

    void realizePosition(const Vector& q) {
        for (MobilizedBodyIndex mbx(1); mbx < m_cache.size(); ++mbx) {
            calcX_FM(mbx, q);
            calcBodyTransforms(mbx);
            calcAcrossJointVelocityJacobian(mbx);
            calcParentToChildVelocityJacobianInGround(mbx);
            calcJointIndependentKinematicsPos(mbx);
        }
    }

    void realizeVelocity(const Vector& u) {
        m_cache[MobilizedBodyIndex(0)].V_GB = SpatialVec(Vec3(0), Vec3(0));
        m_cache[MobilizedBodyIndex(0)].totalCoriolisAcceleration =
            SpatialVec(Vec3(0), Vec3(0));
        for (MobilizedBodyIndex mbx(1); mbx < m_cache.size(); ++mbx) {
            MobodCache& mc = m_cache[mbx];
            mc.V_FM   = mc.H_FM * u[mc.ux];
            mc.V_PB_G = mc.H_PB_G * u[mc.ux];
            calcAcrossJointVelocityJacobianDot(mbx);
            calcParentToChildVelocityJacobianInGroundDot(mbx);
            mc.VD_PB_G = mc.HDot_PB_G * u[mc.ux];
            calcJointIndependentKinematicsVel(mbx);
        }
    }

    // Requires realizePosition() and realizeVelocity() to have been called.
    void calcInverseDynamics(const Vector&              knownUdot,
                             const Vector&              mobilityForces,
                             const Vector_<SpatialVec>& bodyForces,
                             Vector&                    tau) {
        tau.resize(m_matter.getNumMobilities());
        m_cache[MobilizedBodyIndex(0)].A_GB = SpatialVec(Vec3(0), Vec3(0));
        for (MobilizedBodyIndex mbx(1); mbx < m_cache.size(); ++mbx)
            calcBodyAccelerationsFromUdotOutward(mbx, knownUdot);
        for (MobilizedBodyIndex mbx(0); mbx < m_cache.size(); ++mbx)
            m_cache[mbx].F = SpatialVec(Vec3(0), Vec3(0));
        for (MobilizedBodyIndex mbx(m_cache.size()-1); mbx >= 1; --mbx)
            calcInverseDynamicsPass2Inward(mbx, mobilityForces, bodyForces,
                                           tau);
    }

    const SpatialVec& getV_GB(MobilizedBodyIndex mbx) const
    {   return m_cache[mbx].V_GB; }
    const SpatialVec& getA_GB(MobilizedBodyIndex mbx) const
    {   return m_cache[mbx].A_GB; }
    const SpatialVec& getMobilizerCoriolisAcceleration
       (MobilizedBodyIndex mbx) const
    {   return m_cache[mbx].mobilizerCoriolisAcceleration; }
    const SpatialVec& getGyroscopicForce(MobilizedBodyIndex mbx) const
    {   return m_cache[mbx].gyroscopicForce; }
    const SpatialVec& getTotalCentrifugalForces(MobilizedBodyIndex mbx) const
    {   return m_cache[mbx].totalCentrifugalForces; }

private:
    struct MobodCache {
        // Instance.
        MobilizedBodyIndex  parent;
        Transform           X_PF, X_MB;
        Real                mass{NaN};
        Vec3                com_B{NaN, NaN, NaN};
        UnitInertia         unitInertia_Bo_B;
        QIndex              qx;
        UIndex              ux;

        // Position.
        Transform           X_FM, X_PB, X_GB;
        SpatialVec          H_FM, H_PB_G;
        PhiMatrix           Phi;
        Vec3                COM_G;
        SpatialInertia      Mk_G;

        // Velocity.
        SpatialVec          V_FM, V_PB_G, HDot_FM, HDot_PB_G, VD_PB_G, V_GB;
        SpatialVec          gyroscopicForce, mobilizerCoriolisAcceleration,
                            totalCoriolisAcceleration, totalCentrifugalForces;
        SpatialVec          Psidot_PB_G;

        // Acceleration.
        SpatialVec          A_GB, F;
    };

    static SpatialVec reexpress(const Rotation& R, const SpatialVec& H)
    {   return SpatialVec(R*H[0], R*H[1]); }

    //--------------------------------------------------------------------------
    // Pin-specific.
    //--------------------------------------------------------------------------
    void calcX_FM(MobilizedBodyIndex mbx, const Vector& q) {
        MobodCache& mc = m_cache[mbx];
        mc.X_FM = Transform(Rotation(q[mc.qx], ZAxis), Vec3(0));
    }

    void calcAcrossJointVelocityJacobian(MobilizedBodyIndex mbx) {
        m_cache[mbx].H_FM = SpatialVec(Vec3(0,0,1), Vec3(0));
    }

    void calcAcrossJointVelocityJacobianDot(MobilizedBodyIndex mbx) {
        m_cache[mbx].HDot_FM = SpatialVec(Vec3(0), Vec3(0));
    }

    //--------------------------------------------------------------------------
    // Same for all mobilizers.
    //--------------------------------------------------------------------------
    void calcBodyTransforms(MobilizedBodyIndex mbx) {
        MobodCache& mc = m_cache[mbx];
        const Transform X_FB = mc.X_FM * mc.X_MB;
        mc.X_PB = mc.X_PF * X_FB;
        mc.X_GB = m_cache[mc.parent].X_GB * mc.X_PB;
    }

    // H_PB_G maps u to the cross-body relative spatial velocity of B in P,
    // expressed in Ground and taken about Bo.
    void calcParentToChildVelocityJacobianInGround(MobilizedBodyIndex mbx) {
        MobodCache& mc = m_cache[mbx];
        const Rotation R_GF = m_cache[mc.parent].X_GB.R() * mc.X_PF.R();
        const Vec3 r_MB_F = mc.X_FM.R() * mc.X_MB.p();
        const SpatialVec H_MB_F(Vec3(0), -(r_MB_F % mc.H_FM[0]));
        mc.H_PB_G = reexpress(R_GF, mc.H_FM + H_MB_F);
    }

    void calcParentToChildVelocityJacobianInGroundDot(MobilizedBodyIndex mbx) {
        MobodCache& mc = m_cache[mbx];
        const Rotation R_GF = m_cache[mc.parent].X_GB.R() * mc.X_PF.R();
        const Vec3& w_GF = m_cache[mc.parent].V_GB[0];
        const Vec3 r_MB_F = mc.X_FM.R() * mc.X_MB.p();
        const Vec3& w_FM = mc.V_FM[0];
        const SpatialVec HDot_MB_F(Vec3(0),
                                   -(r_MB_F % mc.HDot_FM[0])
                                   - (w_FM % r_MB_F) % mc.H_FM[0]);
        mc.HDot_PB_G = reexpress(R_GF, mc.HDot_FM + HDot_MB_F)
                     + SpatialVec(w_GF % mc.H_PB_G[0], w_GF % mc.H_PB_G[1]);

        // Psidot_i = v_lambda(i) % S_i
        mc.Psidot_PB_G = cross(m_cache[mc.parent].V_GB, mc.H_PB_G);
    }

    // Phi and the spatial mass properties about Bo, expressed in Ground.
    void calcJointIndependentKinematicsPos(MobilizedBodyIndex mbx) {
        MobodCache& mc = m_cache[mbx];
        const Vec3 p_PB_G = m_cache[mc.parent].X_GB.R() * mc.X_PB.p();
        mc.Phi = PhiMatrix(p_PB_G);

        const Rotation& R_GB = mc.X_GB.R();
        const UnitInertia G_Bo_G = mc.unitInertia_Bo_B.reexpress(~R_GB);
        const Vec3 p_BBc_G = R_GB * mc.com_B;
        mc.COM_G = mc.X_GB.p() + p_BBc_G;
        mc.Mk_G  = SpatialInertia(mc.mass, p_BBc_G, G_Bo_G);
    }

    // V_GB, the gyroscopic force b, and the velocity-dependent remainder
    // Jdot*u split into this mobilizer's incremental contribution and the
    // running total.
    void calcJointIndependentKinematicsVel(MobilizedBodyIndex mbx) {
        MobodCache& mc = m_cache[mbx];
        const MobodCache& pc = m_cache[mc.parent];
        const PhiMatrixTranspose PhiT = ~mc.Phi;

        mc.V_GB = PhiT * pc.V_GB + mc.V_PB_G;

        const Vec3& w_GB = mc.V_GB[0];
        const Vec3& v_GB = mc.V_GB[1];
        mc.gyroscopicForce = mc.mass *
            SpatialVec(w_GB % (mc.Mk_G.getUnitInertia() * w_GB),
                       w_GB % (w_GB % mc.Mk_G.getMassCenter()));

        const Vec3& w_GP = pc.V_GB[0];
        const Vec3& v_GP = pc.V_GB[1];
        mc.mobilizerCoriolisAcceleration =
            SpatialVec(mc.VD_PB_G[0],
                       mc.VD_PB_G[1] + w_GP % (v_GB - v_GP));
        mc.totalCoriolisAcceleration = PhiT * pc.totalCoriolisAcceleration
                                     + mc.mobilizerCoriolisAcceleration;
        mc.totalCentrifugalForces = mc.Mk_G * mc.totalCoriolisAcceleration
                                  + mc.gyroscopicForce;
    }

    // A_GB = ~Phi*A_GP + H*udot + a. Base to tip.
    void calcBodyAccelerationsFromUdotOutward(MobilizedBodyIndex mbx,
                                              const Vector& knownUdot) {
        MobodCache& mc = m_cache[mbx];
        const SpatialVec A_GP = ~mc.Phi * m_cache[mc.parent].A_GB;
        mc.A_GB = A_GP + mc.H_PB_G * knownUdot[mc.ux]
                + mc.mobilizerCoriolisAcceleration;
    }

    // F = Mk_G*A_GB + b - F_applied + sum_children Phi*F_child, then
    // tau = ~H*F - f_applied. Tip to base.
    void calcInverseDynamicsPass2Inward(MobilizedBodyIndex         mbx,
                                        const Vector&              mobilityForces,
                                        const Vector_<SpatialVec>& bodyForces,
                                        Vector&                    tau) {
        MobodCache& mc = m_cache[mbx];
        mc.F += mc.Mk_G * mc.A_GB + mc.gyroscopicForce - bodyForces[mbx];
        tau[mc.ux] = dot(mc.H_PB_G, mc.F) - mobilityForces[mc.ux];
        m_cache[mc.parent].F += mc.Phi * mc.F;
    }

    const SimbodyMatterSubsystem&              m_matter;
    Array_<MobodCache, MobilizedBodyIndex>     m_cache;
};


//==============================================================================
//                                  MAIN
//==============================================================================
namespace {

Real maxAbsDiff(const Vector& a, const Vector& b) {
    Real e = 0;
    for (int i=0; i < a.size(); ++i) e = std::max(e, std::abs(a[i]-b[i]));
    return e;
}

Real maxAbsDiff(const SpatialVec& a, const SpatialVec& b) {
    Real e = 0;
    for (int i=0; i < 2; ++i)
        for (int j=0; j < 3; ++j) e = std::max(e, std::abs(a[i][j]-b[i][j]));
    return e;
}

bool report(const char* what, Real err) {
    const bool ok = err <= Tol;
    printf("  %-44s %-10.3e %s\n", what, err, ok ? "OK" : "*** MISMATCH ***");
    return ok;
}

void printVec(const char* label, const Vector& v) {
    printf("  %-22s [", label);
    for (int i=0; i < v.size(); ++i) printf(" % .12f", v[i]);
    printf(" ]\n");
}

} // anonymous namespace

int main() {
try {
    PendulumSystem pendulum;
    pendulum.setState(25*Pi/180, -40*Pi/180, 1.3, -2.1);

    const SimbodyMatterSubsystem& matter = pendulum.getMatter();
    const State& state = pendulum.getState();

    const Vector knownUdot = pendulum.calcUDot(0.7, 2.4);
    Vector              mobilityForces;
    Vector_<SpatialVec> bodyForces;
    pendulum.calcAppliedForces(1.75, Vec3(3.0, -1.5, 2.0),
                               mobilityForces, bodyForces);

    InverseDynamics id(pendulum);
    id.realizePosition(state.getQ());
    id.realizeVelocity(state.getU());

    Vector tau;
    id.calcInverseDynamics(knownUdot, mobilityForces, bodyForces, tau);

    Vector tauRef;
    matter.calcResidualForceIgnoringConstraints(state, mobilityForces,
                                               bodyForces, knownUdot, tauRef);

    printVec("q", state.getQ());
    printVec("u", state.getU());
    printVec("knownUdot", knownUdot);
    printVec("tau", tau);
    printVec("tau (Simbody)", tauRef);

    printf("\n  %-44s %-10s\n", "check", "max|diff|");
    bool allOk = true;
    for (MobilizedBodyIndex mbx(1); mbx < pendulum.getNumBodies(); ++mbx) {
        const MobilizedBody& mobod = matter.getMobilizedBody(mbx);
        char buf[80];
        snprintf(buf, sizeof(buf), "V_GB[%d]", (int)mbx);
        allOk &= report(buf, maxAbsDiff(id.getV_GB(mbx),
                                        mobod.getBodyVelocity(state)));
        snprintf(buf, sizeof(buf), "a_mobilizer[%d]", (int)mbx);
        allOk &= report(buf,
            maxAbsDiff(id.getMobilizerCoriolisAcceleration(mbx),
                       matter.getMobilizerCoriolisAcceleration(state, mbx)));
        snprintf(buf, sizeof(buf), "gyroscopic force[%d]", (int)mbx);
        allOk &= report(buf, maxAbsDiff(id.getGyroscopicForce(mbx),
                                        matter.getGyroscopicForce(state, mbx)));
        snprintf(buf, sizeof(buf), "total centrifugal forces[%d]", (int)mbx);
        allOk &= report(buf,
            maxAbsDiff(id.getTotalCentrifugalForces(mbx),
                       matter.getTotalCentrifugalForces(state, mbx)));
    }

    Vector_<SpatialVec> A_GB_ref;
    matter.calcBodyAccelerationFromUDot(state, knownUdot, A_GB_ref);
    for (MobilizedBodyIndex mbx(1); mbx < pendulum.getNumBodies(); ++mbx) {
        char buf[80];
        snprintf(buf, sizeof(buf), "A_GB[%d]", (int)mbx);
        allOk &= report(buf, maxAbsDiff(id.getA_GB(mbx), A_GB_ref[mbx]));
    }

    allOk &= report("tau vs Simbody", maxAbsDiff(tau, tauRef));

    // Round trip: applying tau in forward dynamics must reproduce knownUdot.
    {
        State dynState = state;
        pendulum.getSystem().realize(dynState, Stage::Dynamics);
        Vector udotFwd;
        Vector_<SpatialVec> A_GB_fwd;
        matter.calcAccelerationIgnoringConstraints(dynState,
                                                   mobilityForces + tau,
                                                   bodyForces,
                                                   udotFwd, A_GB_fwd);
        allOk &= report("forward dynamics round trip",
                        maxAbsDiff(udotFwd, knownUdot));
    }

    printf("\n%s\n\n", allOk ? "All checks passed."
                             : "*** SOME CHECKS FAILED ***");
    return allOk ? 0 : 1;

} catch (const std::exception& e) {
    cout << "EXCEPTION: " << e.what() << endl;
    return 1;
}
}
