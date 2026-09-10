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

/* Every step of Simbody's O(n) inverse dynamics, written out for a multibody
system with Pin, Ball and Free mobilizers, i.e. solving

        tau = M(q) udot + f_inertial(q,u) - f_applied

for the mobility forces tau. The calculations are grouped and named after the
corresponding RigidBodyNode/RigidBodyNodeSpec methods so they can be read side
by side with Simbody's internals. Only three pieces are mobilizer-specific --
calcX_FM(), calcAcrossJointVelocityJacobian() and its Dot form -- and each is
implemented for all three mobilizer types here, so the multi-DoF paths get
exercised rather than assumed.

Each intermediate is checked against the corresponding Simbody quantity, and
the resulting tau against calcResidualForceIgnoringConstraints(). */

#include "SimTKsimbody.h"

#include <cstdio>
#include <iostream>

using namespace SimTK;
using std::cout;
using std::endl;

static const Real Tol = 1e-10;

static const Vec3 GravityVec(0, -9.81, 0);


//==============================================================================
//                              PENDULUM SYSTEM
//==============================================================================
//     Ground --Free--> link1 --Ball--> link2 --Pin--> link3
//                            \--Pin--> link4
//
// A floating base, a multi-DoF interior joint, and a branch, so nothing about
// the algorithm below can be right by accident of topology. Mass centers are
// off their body origins and the mobilizer frames are skew, so no term drops
// out numerically either. nq = 13 (Free and Ball are quaternion-parameterized),
// nu = 11.
class PendulumSystem {
public:
    PendulumSystem() : m_matter(m_system) {
        Body::Rigid body1(massProps(2.0, Vec3( 0.03,-0.02, 0.05), 0.30, 0.05));
        Body::Rigid body2(massProps(1.4, Vec3(-0.02, 0.04,-0.01), 0.25, 0.04));
        Body::Rigid body3(massProps(0.8, Vec3( 0.01, 0.03, 0.02), 0.18, 0.03));
        Body::Rigid body4(massProps(0.6, Vec3( 0.02,-0.01,-0.03), 0.15, 0.03));

        m_link1 = MobilizedBody::Free(m_matter.Ground(),
            Transform(Rotation(0.20, XAxis), Vec3(0.05,  0.10, -0.02)),
            body1,
            Transform(Rotation(0.15, YAxis), Vec3(0.00,  0.30,  0.00)));

        m_link2 = MobilizedBody::Ball(m_link1,
            Transform(Rotation(0.35, ZAxis), Vec3(0.00, -0.30,  0.04)),
            body2,
            Transform(Rotation(0.10, XAxis), Vec3(0.02,  0.25,  0.00)));

        m_link3 = MobilizedBody::Pin(m_link2,
            Transform(Rotation(0.40, XAxis), Vec3(0.00, -0.25,  0.03)),
            body3,
            Transform(Rotation(0.25, ZAxis), Vec3(0.00,  0.18,  0.00)));

        m_link4 = MobilizedBody::Pin(m_link1,
            Transform(Rotation(-0.30, YAxis), Vec3(0.12, -0.10, -0.05)),
            body4,
            Transform(Rotation(0.05, XAxis), Vec3(0.00,  0.15,  0.00)));

        m_state = m_system.realizeTopology();
    }

    // Set through the mobilizer-level fitting methods rather than by writing q
    // directly, so the Free and Ball quaternions come out normalized.
    void setArbitraryState() {
        m_link1.setQToFitTransform(m_state,
            Transform(Rotation(BodyRotationSequence,
                               0.30, XAxis, -0.45, YAxis, 0.25, ZAxis),
                      Vec3(0.15, 0.40, -0.20)));
        m_link1.setUToFitVelocity(m_state,
            SpatialVec(Vec3(0.70, -1.10, 0.50), Vec3(-0.40, 0.90, 1.30)));

        m_link2.setQToFitRotation(m_state,
            Rotation(BodyRotationSequence,
                     -0.55, XAxis, 0.35, YAxis, 0.60, ZAxis));
        m_link2.setUToFitAngularVelocity(m_state, Vec3(1.20, 0.60, -0.85));

        m_link3.setAngle(m_state, 0.65);   m_link3.setRate(m_state, -1.45);
        m_link4.setAngle(m_state, -0.85);  m_link4.setRate(m_state,  0.95);

        m_system.realize(m_state, Stage::Velocity);
    }

    Vector calcArbitraryUDot() const {
        Vector udot(getNumMobilities());
        for (int i=0; i < udot.size(); ++i)
            udot[i] = 0.35 + 0.27*i - 0.11*(i%3) - 0.05*(i%5);
        return udot;
    }

    // One spatial force per body (moment about Bo, force at Bo, expressed in
    // Ground), Ground being body zero.
    void calcGravityBodyForces(const State& state,
                               Vector_<SpatialVec>& bodyForces) const {
        bodyForces.resize(getNumBodies());
        bodyForces.setTo(SpatialVec(Vec3(0), Vec3(0)));
        for (MobilizedBodyIndex mbx(1); mbx < getNumBodies(); ++mbx) {
            const MobilizedBody& mobod = m_matter.getMobilizedBody(mbx);
            const MassProperties& mp = mobod.getBodyMassProperties(state);
            const Vec3 p_BoBc_G =
                mobod.getBodyRotation(state) * mp.getMassCenter();
            const Vec3 fGrav_G = mp.getMass() * GravityVec;
            bodyForces[mbx] = SpatialVec(p_BoBc_G % fGrav_G, fGrav_G);
        }
    }

    // Gravity, a point force on the tip of link3, and a torque on every
    // mobility, so no applied-force path goes untested.
    void calcAppliedForces(const State& state, Vector& mobilityForces,
                           Vector_<SpatialVec>& bodyForces) const {
        mobilityForces.resize(getNumMobilities());
        for (int i=0; i < mobilityForces.size(); ++i)
            mobilityForces[i] = 0.9 - 0.31*i + 0.17*(i%4);

        calcGravityBodyForces(state, bodyForces);

        const Vec3 tipForce_G(3.0, -1.5, 2.0);
        const Vec3 p_BoS_G = m_link3.getBodyRotation(state) * Vec3(0, -0.18, 0);
        bodyForces[m_link3.getMobilizedBodyIndex()] +=
            SpatialVec(p_BoS_G % tipForce_G, tipForce_G);
    }

    const MultibodySystem&        getSystem() const {return m_system;}
    const SimbodyMatterSubsystem& getMatter() const {return m_matter;}
    const State&                  getState()  const {return m_state;}
    int getNumBodies()     const {return m_matter.getNumBodies();}
    int getNumMobilities() const {return m_matter.getNumMobilities();}

private:
    static MassProperties massProps(Real mass, const Vec3& com,
                                    Real halfLength, Real radius) {
        const Inertia Ic = mass * Inertia::cylinderAlongY(radius, halfLength);
        return MassProperties(mass, com, Ic.shiftFromMassCenter(-com, mass));
    }

    MultibodySystem        m_system;
    SimbodyMatterSubsystem m_matter;
    MobilizedBody::Free    m_link1;
    MobilizedBody::Ball    m_link2;
    MobilizedBody::Pin     m_link3, m_link4;
    State                  m_state;
};


//==============================================================================
//                              INVERSE DYNAMICS
//==============================================================================
// Reimplementation of Simbody's O(n) inverse dynamics using the same
// decomposition and method names as RigidBodyNode/RigidBodyNodeSpec. Hinge
// matrices are stored one SpatialVec per mobility, so a column of H for joint j
// is m_H[firstUIndex(j) + d].
class InverseDynamics {
public:
    explicit InverseDynamics(const PendulumSystem& pendulum)
    :   m_matter(pendulum.getMatter()), m_cache(pendulum.getNumBodies()) {
        const State& state = pendulum.getState();
        const int nu = pendulum.getNumMobilities();
        m_H_FM.resize(nu); m_H.resize(nu);
        m_HDot_FM.resize(nu); m_HDot.resize(nu);

        for (MobilizedBodyIndex mbx(0); mbx < pendulum.getNumBodies(); ++mbx) {
            const MobilizedBody& mobod = m_matter.getMobilizedBody(mbx);
            MobodCache& mc = m_cache[mbx];
            if (mbx == 0) { mc.X_GB = Transform(); continue; }

            if      (MobilizedBody::Pin::isInstanceOf(mobod))  mc.type = PinType;
            else if (MobilizedBody::Ball::isInstanceOf(mobod)) mc.type = BallType;
            else if (MobilizedBody::Free::isInstanceOf(mobod)) mc.type = FreeType;
            else SimTK_ERRCHK1_ALWAYS(false, "InverseDynamics()",
                "Mobilized body %d is not a Pin, Ball or Free.", (int)mbx);

            mc.parent = mobod.getParentMobilizedBody().getMobilizedBodyIndex();
            mc.X_PF   = mobod.getInboardFrame(state);
            mc.X_MB   = ~mobod.getOutboardFrame(state);
            const MassProperties& mp = mobod.getBodyMassProperties(state);
            mc.mass             = mp.getMass();
            mc.com_B            = mp.getMassCenter();
            mc.unitInertia_Bo_B = mp.getUnitInertia();
            mc.qx = mobod.getFirstQIndex(state);
            mc.ux = mobod.getFirstUIndex(state);
            mc.nu = mobod.getNumU(state);
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
        const SpatialVec zero(Vec3(0), Vec3(0));
        m_cache[MobilizedBodyIndex(0)].V_GB = zero;
        m_cache[MobilizedBodyIndex(0)].totalCoriolisAcceleration = zero;

        for (MobilizedBodyIndex mbx(1); mbx < m_cache.size(); ++mbx) {
            MobodCache& mc = m_cache[mbx];
            mc.V_FM = zero; mc.V_PB_G = zero;
            for (int d=0; d < mc.nu; ++d) {
                const UIndex ux(mc.ux+d);
                mc.V_FM   += m_H_FM[ux] * u[ux];
                mc.V_PB_G += m_H[ux]    * u[ux];
            }
            calcAcrossJointVelocityJacobianDot(mbx);
            calcParentToChildVelocityJacobianInGroundDot(mbx);
            mc.VD_PB_G = zero;
            for (int d=0; d < mc.nu; ++d)
                mc.VD_PB_G += m_HDot[UIndex(mc.ux+d)] * u[UIndex(mc.ux+d)];
            calcJointIndependentKinematicsVel(mbx);
        }
    }

    // Requires realizePosition() and realizeVelocity().
    void calcInverseDynamics(const Vector&              knownUdot,
                             const Vector&              mobilityForces,
                             const Vector_<SpatialVec>& bodyForces,
                             Vector&                    tau) {
        tau.resize(m_matter.getNumMobilities());
        const SpatialVec zero(Vec3(0), Vec3(0));
        m_cache[MobilizedBodyIndex(0)].A_GB = zero;
        for (MobilizedBodyIndex mbx(1); mbx < m_cache.size(); ++mbx)
            calcBodyAccelerationsFromUdotOutward(mbx, knownUdot);
        for (MobilizedBodyIndex mbx(0); mbx < m_cache.size(); ++mbx)
            m_cache[mbx].F = zero;
        for (MobilizedBodyIndex mbx(m_cache.size()-1); mbx >= 1; --mbx)
            calcInverseDynamicsPass2Inward(mbx, mobilityForces, bodyForces, tau);
    }

    const SpatialVec& getH(UIndex ux) const {return m_H[ux];}
    const SpatialVec& getV_GB(MobilizedBodyIndex mbx) const
    {   return m_cache[mbx].V_GB; }
    const SpatialVec& getA_GB(MobilizedBodyIndex mbx) const
    {   return m_cache[mbx].A_GB; }
    const SpatialVec& getMobilizerCoriolisAcceleration
       (MobilizedBodyIndex mbx) const
    {   return m_cache[mbx].mobilizerCoriolisAcceleration; }
    const SpatialVec& getTotalCoriolisAcceleration
       (MobilizedBodyIndex mbx) const
    {   return m_cache[mbx].totalCoriolisAcceleration; }
    const SpatialVec& getGyroscopicForce(MobilizedBodyIndex mbx) const
    {   return m_cache[mbx].gyroscopicForce; }
    const SpatialVec& getTotalCentrifugalForces(MobilizedBodyIndex mbx) const
    {   return m_cache[mbx].totalCentrifugalForces; }

private:
    enum MobilizerType {PinType, BallType, FreeType};

    struct MobodCache {
        // Instance.
        MobilizedBodyIndex  parent;
        MobilizerType       type{PinType};
        Transform           X_PF, X_MB;
        Real                mass{NaN};
        Vec3                com_B{NaN, NaN, NaN};
        UnitInertia         unitInertia_Bo_B;
        QIndex              qx;
        UIndex              ux;
        int                 nu{0};

        // Position.
        Transform           X_FM, X_PB, X_GB;
        PhiMatrix           Phi;
        Vec3                COM_G;
        SpatialInertia      Mk_G;

        // Velocity.
        SpatialVec          V_FM, V_PB_G, VD_PB_G, V_GB;
        SpatialVec          gyroscopicForce, mobilizerCoriolisAcceleration,
                            totalCoriolisAcceleration, totalCentrifugalForces;

        // Acceleration.
        SpatialVec          A_GB, F;
    };

    static SpatialVec reexpress(const Rotation& R, const SpatialVec& H)
    {   return SpatialVec(R*H[0], R*H[1]); }

    static Vec3 unitVec(int i) {Vec3 e(0); e[i] = 1; return e;}

    //--------------------------------------------------------------------------
    // Mobilizer-specific. These are the only three methods that need to know
    // which kind of mobilizer this is.
    //--------------------------------------------------------------------------
    // Ball and Free are quaternion-parameterized: q = [quat(4)] and
    // q = [quat(4), p_FM(3)] respectively, with p_FM along the F axes.
    void calcX_FM(MobilizedBodyIndex mbx, const Vector& q) {
        MobodCache& mc = m_cache[mbx];
        const QIndex qx = mc.qx;
        switch (mc.type) {
        case PinType:
            mc.X_FM = Transform(Rotation(q[qx], ZAxis), Vec3(0));
            break;
        case BallType:
            mc.X_FM = Transform(Rotation(Quaternion(
                          Vec4(q[qx], q[qx+1], q[qx+2], q[qx+3]))), Vec3(0));
            break;
        case FreeType:
            mc.X_FM = Transform(Rotation(Quaternion(
                          Vec4(q[qx], q[qx+1], q[qx+2], q[qx+3]))),
                      Vec3(q[qx+4], q[qx+5], q[qx+6]));
            break;
        }
    }

    // H_FM maps u to the cross-mobilizer velocity V_FM = (w_FM, v_FMo),
    // expressed in F and referred to Mo. For all three of these mobilizers the
    // generalized speeds ARE the measure numbers of w_FM and v_FMo in F, so
    // H_FM is a constant selection matrix -- which is also why HDot_FM below is
    // zero. A mobilizer with q-dependent H_FM would need more.
    void calcAcrossJointVelocityJacobian(MobilizedBodyIndex mbx) {
        const MobodCache& mc = m_cache[mbx];
        switch (mc.type) {
        case PinType:
            m_H_FM[mc.ux] = SpatialVec(Vec3(0,0,1), Vec3(0));
            break;
        case BallType:
            for (int d=0; d < 3; ++d)
                m_H_FM[UIndex(mc.ux+d)] = SpatialVec(unitVec(d), Vec3(0));
            break;
        case FreeType:
            for (int d=0; d < 3; ++d) {
                m_H_FM[UIndex(mc.ux+d)]   = SpatialVec(unitVec(d), Vec3(0));
                m_H_FM[UIndex(mc.ux+3+d)] = SpatialVec(Vec3(0), unitVec(d));
            }
            break;
        }
    }

    void calcAcrossJointVelocityJacobianDot(MobilizedBodyIndex mbx) {
        const MobodCache& mc = m_cache[mbx];
        for (int d=0; d < mc.nu; ++d)
            m_HDot_FM[UIndex(mc.ux+d)] = SpatialVec(Vec3(0), Vec3(0));
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

    // H (== H_PB_G) maps u to the cross-body relative spatial velocity of B in
    // P, expressed in Ground and referred to Bo.
    void calcParentToChildVelocityJacobianInGround(MobilizedBodyIndex mbx) {
        const MobodCache& mc = m_cache[mbx];
        const Rotation R_GF = m_cache[mc.parent].X_GB.R() * mc.X_PF.R();
        const Vec3 r_MB_F = mc.X_FM.R() * mc.X_MB.p();
        for (int d=0; d < mc.nu; ++d) {
            const UIndex ux(mc.ux+d);
            const SpatialVec H_MB_F(Vec3(0), -(r_MB_F % m_H_FM[ux][0]));
            m_H[ux] = reexpress(R_GF, m_H_FM[ux] + H_MB_F);
        }
    }

    void calcParentToChildVelocityJacobianInGroundDot(MobilizedBodyIndex mbx) {
        const MobodCache& mc = m_cache[mbx];
        const Rotation R_GF = m_cache[mc.parent].X_GB.R() * mc.X_PF.R();
        const Vec3& w_GF = m_cache[mc.parent].V_GB[0];
        const Vec3 r_MB_F = mc.X_FM.R() * mc.X_MB.p();
        const Vec3& w_FM = mc.V_FM[0];
        for (int d=0; d < mc.nu; ++d) {
            const UIndex ux(mc.ux+d);
            const SpatialVec HDot_MB_F(Vec3(0),
                -(r_MB_F % m_HDot_FM[ux][0])
                - (w_FM % r_MB_F) % m_H_FM[ux][0]);
            m_HDot[ux] = reexpress(R_GF, m_HDot_FM[ux] + HDot_MB_F)
                       + SpatialVec(w_GF % m_H[ux][0], w_GF % m_H[ux][1]);
        }
    }

    // Phi, and the spatial mass properties about Bo expressed in Ground.
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
            SpatialVec(mc.VD_PB_G[0], mc.VD_PB_G[1] + w_GP % (v_GB - v_GP));
        mc.totalCoriolisAcceleration = PhiT * pc.totalCoriolisAcceleration
                                     + mc.mobilizerCoriolisAcceleration;
        mc.totalCentrifugalForces = mc.Mk_G * mc.totalCoriolisAcceleration
                                  + mc.gyroscopicForce;
    }

    // A_GB = ~Phi*A_GP + H*udot + a. Base to tip.
    void calcBodyAccelerationsFromUdotOutward(MobilizedBodyIndex mbx,
                                              const Vector& knownUdot) {
        MobodCache& mc = m_cache[mbx];
        SpatialVec A = ~mc.Phi * m_cache[mc.parent].A_GB;
        for (int d=0; d < mc.nu; ++d)
            A += m_H[UIndex(mc.ux+d)] * knownUdot[UIndex(mc.ux+d)];
        mc.A_GB = A + mc.mobilizerCoriolisAcceleration;
    }

    // F = Mk_G*A_GB + b - F_applied + sum_children Phi*F_child, then
    // tau = ~H*F - f_applied. Tip to base.
    void calcInverseDynamicsPass2Inward(MobilizedBodyIndex         mbx,
                                        const Vector&              mobilityForces,
                                        const Vector_<SpatialVec>& bodyForces,
                                        Vector&                    tau) {
        MobodCache& mc = m_cache[mbx];
        mc.F += mc.Mk_G * mc.A_GB + mc.gyroscopicForce - bodyForces[mbx];
        for (int d=0; d < mc.nu; ++d) {
            const UIndex ux(mc.ux+d);
            tau[ux] = dot(m_H[ux], mc.F) - mobilityForces[ux];
        }
        m_cache[mc.parent].F += mc.Phi * mc.F;
    }

    const SimbodyMatterSubsystem&          m_matter;
    Array_<MobodCache, MobilizedBodyIndex> m_cache;
    Array_<SpatialVec, UIndex>             m_H_FM, m_H, m_HDot_FM, m_HDot;
};






//==============================================================================
//                               OPERATOR SENSITIVITIES
//==============================================================================


// One column of the Eq. 18.28 sensitivities: the coordinate (i,d) is fixed and
// each entry is indexed by the body k it belongs to.
struct Derivatives {
    Vector_<SpatialVec> dVw_dqdot;          // 18.28a
    Vector_<SpatialVec> dVwParent_dqdot;    // 18.28b

    Derivatives(int nb)
    :   dVw_dqdot(nb, SpatialVec(Vec3(0), Vec3(0))),
        dVwParent_dqdot(nb, SpatialVec(Vec3(0), Vec3(0))) {}
};


// V^w is exactly linear in u, so the step size is uncritical and the central
// difference is exact to roundoff.
Derivatives calcFiniteDifferences(const SimbodyMatterSubsystem& matter,
                                  const State& state,
                                  MobilizedBodyIndex i, int d) {

    const MobilizedBody& mobod_i = matter.getMobilizedBody(i);
    SimTK_ERRCHK2_ALWAYS(d >= 0 && d < mobod_i.getNumU(state),
        "calcFiniteDifferences()",
        "Mobility %d is out of range for mobilized body %d.", d, (int)i);

    const Real h = 1e-5;
    const UIndex ux(mobod_i.getFirstUIndex(state) + d);

    // (i,d) is fixed, so perturb once rather than once per body.
    State pertPlus = state, pertMinus = state;
    pertPlus.updU()[ux]  += h;
    pertMinus.updU()[ux] -= h;
    matter.getSystem().realize(pertPlus,  Stage::Velocity);
    matter.getSystem().realize(pertMinus, Stage::Velocity);

    const int nb = matter.getNumBodies();
    Derivatives derivatives(nb);
    for (int k = 1; k < nb; ++k) {
        const MobilizedBody& mobod_k =
            matter.getMobilizedBody(MobilizedBodyIndex(k));
        const MobilizedBody& parent_k = mobod_k.getParentMobilizedBody();

        // dV^w(k) / dqdot_{i,d} (18.28a)
        derivatives.dVw_dqdot[k] = SpatialVec(
            (mobod_k.getBodyVelocity(pertPlus)[0]
           - mobod_k.getBodyVelocity(pertMinus)[0]) / (2*h), Vec3(0));

        // dV^w(p(k)) / dqdot_{i,d} (18.28b)
        derivatives.dVwParent_dqdot[k] = SpatialVec(
            (parent_k.getBodyVelocity(pertPlus)[0]
           - parent_k.getBodyVelocity(pertMinus)[0]) / (2*h), Vec3(0));
    }

    return derivatives;
}

Derivatives calcSensitivities(const SimbodyMatterSubsystem& matter,
                              const State& state,
                              MobilizedBodyIndex i, int d) {

    // Jain's 1_[i >= k]: joint i lies on the path from body k to
    // Ground. Returns false for Ground, which is what makes 1_[i > k] fall out
    // of asking this about p(k).
    auto isAncestorOf = [&](MobilizedBodyIndex iAnc, MobilizedBodyIndex kBody) {
        for (MobilizedBodyIndex b = kBody; b != 0;
                b = matter.getMobilizedBody(b).getParentMobilizedBody()
                        .getMobilizedBodyIndex()) {
            if (b == iAnc) return true;
        }
        return false;
    };

    const MobilizedBody& mobod_i = matter.getMobilizedBody(i);
    SimTK_ERRCHK2_ALWAYS(d >= 0 && d < mobod_i.getNumU(state),
        "calcSensitivities()",
        "Mobility %d is out of range for mobilized body %d.", d, (int)i);

    const SpatialVec& H_i = mobod_i.getHCol(state, MobilizerUIndex(d));

    const int nb = matter.getNumBodies();
    Derivatives derivatives(nb);
    for (int k = 1; k < nb; ++k) {
        const MobilizedBodyIndex p_k = matter.getMobilizedBody(k)
            .getParentMobilizedBody().getMobilizedBodyIndex();

        // dV^w(k) / dqdot_{i,d} = H*_w(i) 1_[i >= k]            (18.28a)
        if (isAncestorOf(i, k)) {
            derivatives.dVw_dqdot[k] = SpatialVec(H_i[0], Vec3(0));
        }

        // dV^w(p(k)) / dqdot_{i,d} = H*_w(i) 1_[i > k]          (18.28b)
        if (isAncestorOf(i, pk)) {
            derivatives.dVwParent_dqdot[k] = SpatialVec(H_i[0], Vec3(0));
        }
    }

    return derivatives;
}

int main() {
    PendulumSystem pendulum;
    pendulum.setArbitraryState();

    const SimbodyMatterSubsystem& matter = pendulum.getMatter();
    const State& state = pendulum.getState();
    const int nb = pendulum.getNumBodies();

    const Vector knownUdot = pendulum.calcArbitraryUDot();
    Vector              mobilityForces;
    Vector_<SpatialVec> bodyForces;
    pendulum.calcAppliedForces(state, mobilityForces, bodyForces);

    InverseDynamics id(pendulum);
    id.realizePosition(state.getQ());
    id.realizeVelocity(state.getU());

    Vector tau;
    id.calcInverseDynamics(knownUdot, mobilityForces, bodyForces, tau);

    Vector tauRef;
    matter.calcResidualForceIgnoringConstraints(state, mobilityForces,
                                                bodyForces, knownUdot, tauRef);

}
