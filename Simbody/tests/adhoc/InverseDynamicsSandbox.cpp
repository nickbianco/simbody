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

#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>

using namespace SimTK;
using std::cout;
using std::endl;


static const Vec3 GravityVec(0, -9.81, 0);

// Jain's Htilde*_w applied to a single spatial vector: the angular-only
// cross-product operator diag(w~, w~).
static SpatialVec angularCross(const Vec3& w, const SpatialVec& X) {
    return SpatialVec(w % X[0], w % X[1]);
}


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
            mc.p_BoMo_B         = mobod.getOutboardFrame(state).p();
            mc.qx = mobod.getFirstQIndex(state);
            mc.ux = mobod.getFirstUIndex(state);
            mc.nu = mobod.getNumU(state);
        }

        const int nb = pendulum.getNumBodies();
        m_subtree.resize(nb);
        m_path.resize(nb);
        for (MobilizedBodyIndex k(1); k < nb; ++k)
            for (MobilizedBodyIndex b = k; b != 0; b = m_cache[b].parent)
                m_subtree[b].push_back(k);
        for (MobilizedBodyIndex i(1); i < nb; ++i)
            for (MobilizedBodyIndex b = m_cache[i].parent; b != 0;
                 b = m_cache[b].parent)
                m_path[i].push_back(b);

        const SpatialVec zeroV(Vec3(0), Vec3(0));
        m_dV.resize(nb); m_dA.resize(nb); m_dF.resize(nb);
        m_eV.resize(nb); m_eA.resize(nb); m_eF.resize(nb);
        for (MobilizedBodyIndex k(0); k < nb; ++k) {
            m_dV[k] = m_dA[k] = m_dF[k] = zeroV;
            m_eV[k] = m_eA[k] = m_eF[k] = zeroV;
        }
        m_dH.resize(nu);
        for (UIndex ux(0); ux < nu; ++ux) m_dH[ux] = zeroV;
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
        m_u = u;
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

    // Derivatives of tau with respect to q and u. Requires realizePosition(),
    // realizeVelocity() and calcInverseDynamics() to have run: this reuses
    // their cached Phi, H, Mk_G, V_GB, A_GB and the composite transmitted
    // force F. Column (i,d) of each Jacobian is the derivative along that
    // coordinate; row ux is the affected mobility.
    //
    // Two things make this cheaper than calling a per-direction routine nu
    // times. First, everything depending only on (q,u,udot) is computed once by
    // the passes above instead of being rebuilt for each column. Second, every
    // primitive sensitivity carries an i >= k condition, so for a perturbation
    // at joint i they all vanish outside the subtree rooted at i, and dF is
    // nonzero only on that subtree plus i's path to Ground. The sweeps visit
    // exactly those bodies rather than all nb.
    void calcInverseDynamicsDerivatives(const Vector& knownUdot,
                                        Matrix& dtau_dq, Matrix& dtau_du) {
        const int nu = (int)m_H.size();
        dtau_dq.resize(nu, nu); dtau_dq.setToZero();
        dtau_du.resize(nu, nu); dtau_du.setToZero();
        for (MobilizedBodyIndex i(1); i < m_cache.size(); ++i)
            for (int d = 0; d < m_cache[i].nu; ++d)
                calcDerivativeColumn(i, d, knownUdot, dtau_dq, dtau_du);
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
        Vec3                p_BoMo_B;
        QIndex              qx;
        UIndex              ux;
        int                 nu{0};

        // Position.
        Transform           X_FM, X_PB, X_GB;
        PhiMatrix           Phi;
        Vec3                COM_G;
        SpatialInertia      Mk_G;
        Vec3                rho_G;

        // Velocity.
        SpatialVec          V_FM, V_PB_G, VD_PB_G, V_GB;
        SpatialVec          gyroscopicForce, mobilizerCoriolisAcceleration,
                            totalCoriolisAcceleration, totalCentrifugalForces;
        SpatialVec          Mw;       // Mk_G * (w_GB, 0), reused by every column

        // Acceleration.
        SpatialVec          A_GB, F;
        SpatialVec          MA;       // Mk_G * A_GB, reused by every column
    };

    // One column of dtau/dq and dtau/du, for the d-th free mode of joint i.
    //
    // Every primitive sensitivity in this file is a commutator with the
    // angular cross operator W(w_i) = diag(w_i~, w_i~). Singh's restructuring
    // observation is that such commutators never need to be formed, only
    // applied: for any operator X and spatial vector y,
    //     [W X - X W] y = W(X y) - X(W y)
    // which replaces a 6x6 construction plus two 6x6 products with two cross
    // products and one structured application of X. No dense 6x6 appears
    // below. This is a re-association of the verified per-column expressions,
    // not a different algorithm; see the note in main() on why Singh's Eq. 27
    // itself cannot be transcribed under Simbody's free-mode convention.
    void calcDerivativeColumn(MobilizedBodyIndex i, int d,
                              const Vector& knownUdot,
                              Matrix& dtau_dq, Matrix& dtau_du) {
        const SpatialVec zeroV(Vec3(0), Vec3(0));
        const UIndex col(m_cache[i].ux + d);
        const SpatialVec& H_i = m_H[col];
        const Vec3&       w_i = H_i[0];    // the mode's angular direction
        const Vec3&       v_i = H_i[1];    // and its linear part

        //---- Scratch reset --------------------------------------------------
        // Only the subtree of i, i's path to Ground, and Ground itself are ever
        // written, so only those are cleared; a short reset is the whole point
        // of restricting the sweeps.
        for (MobilizedBodyIndex k : m_subtree[i]) {
            m_dV[k] = m_dA[k] = m_dF[k] = zeroV;
            m_eV[k] = m_eA[k] = m_eF[k] = zeroV;
            for (int e = 0; e < m_cache[k].nu; ++e)
                m_dH[UIndex(m_cache[k].ux+e)] = zeroV;
        }
        for (MobilizedBodyIndex k : m_path[i]) {
            m_dV[k] = m_dA[k] = m_dF[k] = zeroV;
            m_eV[k] = m_eA[k] = m_eF[k] = zeroV;
        }
        m_dF[MobilizedBodyIndex(0)] = m_eF[MobilizedBodyIndex(0)] = zeroV;

        //---- Outward sweep, base to tip -------------------------------------
        for (MobilizedBodyIndex k : m_subtree[i]) {
            MobodCache& mc = m_cache[k];
            const MobodCache& pc = m_cache[mc.parent];
            const bool self = (k == i);   // within subtree(i), k != i => i > k

            // Hinge-matrix sensitivity.                        [Jain 18.29b]
            // dH = W(w_i) H for i > k; for k == i the modes are fixed in the
            // parent frame F while their line of action passes through Mo, so
            //     dH = (0, -H_w x (w_i x rho_k)).
            for (int e = 0; e < mc.nu; ++e) {
                const UIndex ux(mc.ux+e);
                const SpatialVec& H_ke = m_H[ux];
                m_dH[ux] = self
                    ? SpatialVec(Vec3(0), -(H_ke[0] % (w_i % mc.rho_G)))
                    : angularCross(w_i, H_ke);
            }

            // Shift-operator sensitivity, applied not formed.  [Jain 18.29a]
            // For i > k, dPhi = W Phi - Phi W, so
            //     (~dPhi) y = W(~Phi y) - ~Phi(W y).
            // For k == i only the joint's own translation enters, dPhi =
            // [[0, v_i~],[0,0]], whose transpose sends y to (0, -v_i x y_w).
            auto dPhiT = [&](const SpatialVec& y) -> SpatialVec {
                return self ? SpatialVec(Vec3(0), -(v_i % y[0]))
                            : angularCross(w_i, (~mc.Phi) * y)
                              - (~mc.Phi) * angularCross(w_i, y);
            };

            // Velocity sensitivity, q and u sides.
            //     dV = (~dPhi) V_p + ~Phi dV_p + sum_e dH u
            //     eV = ~Phi eV_p + H_i 1[i == k]
            SpatialVec dVr = zeroV;
            for (int e = 0; e < mc.nu; ++e)
                dVr += m_dH[UIndex(mc.ux+e)] * m_u[UIndex(mc.ux+e)];
            m_dV[k] = dPhiT(pc.V_GB) + (~mc.Phi) * m_dV[mc.parent] + dVr;

            const SpatialVec eVr = self ? H_i : zeroV;
            m_eV[k] = (~mc.Phi) * m_eV[mc.parent] + eVr;

            // Coriolis-acceleration sensitivity. a_c has the closed form
            //     a_c = (0, wp x (v_k - v_p)) + W(wp) Vr + (0, -wr x (wr x rho))
            // so differentiating it needs only dV, dVr and d(rho), the last
            // being w_i x rho since rho is fixed in body k.   [as Jain 18.29c]
            const SpatialVec& Vr = mc.V_PB_G;
            const Vec3& wp  = pc.V_GB[0];
            const Vec3& wr  = Vr[0];
            const Vec3& dwp = m_dV[mc.parent][0];
            const Vec3& ewp = m_eV[mc.parent][0];
            const Vec3  dwr = dVr[0], ewr = eVr[0];
            const Vec3  drho = w_i % mc.rho_G;

            const SpatialVec da_c =
                  SpatialVec(Vec3(0), dwp % (mc.V_GB[1] - pc.V_GB[1])
                                    + wp  % (m_dV[k][1] - m_dV[mc.parent][1]))
                + angularCross(dwp, Vr) + angularCross(wp, dVr)
                + SpatialVec(Vec3(0), -(dwr % (wr % mc.rho_G))
                                      -(wr  % (dwr % mc.rho_G))
                                      -(wr  % (wr  % drho)));

            // The u side drops the d(rho) term: rho is configuration-only.
            const SpatialVec ea_c =
                  SpatialVec(Vec3(0), ewp % (mc.V_GB[1] - pc.V_GB[1])
                                    + wp  % (m_eV[k][1] - m_eV[mc.parent][1]))
                + angularCross(ewp, Vr) + angularCross(wp, eVr)
                + SpatialVec(Vec3(0), -(ewr % (wr % mc.rho_G))
                                      -(wr  % (ewr % mc.rho_G)));

            // Acceleration sensitivity.
            //     dA = (~dPhi) A_p + ~Phi dA_p + sum_e dH udot + da_c
            //     eA = ~Phi eA_p + ea_c          (H is q-only, so no eH term)
            SpatialVec dA = dPhiT(pc.A_GB) + (~mc.Phi) * m_dA[mc.parent] + da_c;
            for (int e = 0; e < mc.nu; ++e)
                dA += m_dH[UIndex(mc.ux+e)] * knownUdot[UIndex(mc.ux+e)];
            m_dA[k] = dA;
            m_eA[k] = (~mc.Phi) * m_eA[mc.parent] + ea_c;
        }

        //---- Inward sweep, tip to base --------------------------------------
        for (int n = (int)m_subtree[i].size()-1; n >= 0; --n) {
            const MobilizedBodyIndex k = m_subtree[i][n];
            MobodCache& mc = m_cache[k];
            const bool self = (k == i);
            const Vec3& w  = mc.V_GB[0];
            const Vec3& dw = m_dV[k][0];
            const Vec3& ew = m_eV[k][0];

            // Inertia sensitivity, applied not formed.         [Jain 18.29c]
            // dM = W M - M W, hence dM y = W(M y) - M(W y). Both uses below
            // are of that form, with M y precomputed per body (MA and Mw).
            const SpatialVec dMA = angularCross(w_i, mc.MA)
                                 - mc.Mk_G * angularCross(w_i, mc.A_GB);
            const SpatialVec dMw = angularCross(w_i, mc.Mw)
                                 - mc.Mk_G * SpatialVec(w_i % w, Vec3(0));

            // Gyroscopic-force sensitivity. b = W(w)[M (w,0)], so
            //     db = W(dw)[M(w,0)] + W(w)[dM(w,0) + M(dw,0)]
            // and the u side drops the dM term.
            const SpatialVec db =
                  angularCross(dw, mc.Mw)
                + angularCross(w, dMw + mc.Mk_G * SpatialVec(dw, Vec3(0)));
            const SpatialVec eb =
                  angularCross(ew, mc.Mw)
                + angularCross(w, mc.Mk_G * SpatialVec(ew, Vec3(0)));

            // Transmitted-force sensitivity.
            //     dF += dM A + M dA + db ;  eF += M eA + eb
            m_dF[k] += dMA + mc.Mk_G * m_dA[k] + db;
            m_eF[k] += mc.Mk_G * m_eA[k] + eb;

            // Generalized-force sensitivity.                   [Singh J3]
            //     dtau = [dH] . F + H . dF ;  etau = H . eF
            // The first term is J3's residue and is nonzero only for multi-DoF
            // joints; there is no [eH] . F counterpart since H is q-only.
            for (int e = 0; e < mc.nu; ++e) {
                const UIndex ux(mc.ux+e);
                dtau_dq(ux, col) = dot(m_dH[ux], mc.F) + dot(m_H[ux], m_dF[k]);
                dtau_du(ux, col) = dot(m_H[ux], m_eF[k]);
            }

            // Push to the parent. The dPhi action on a force runs the other
            // way: dPhi F = W(Phi F) - Phi(W F), or (v_i x f, 0) when k == i.
            const SpatialVec dPhiF = self
                ? SpatialVec(v_i % mc.F[1], Vec3(0))
                : angularCross(w_i, mc.Phi * mc.F)
                  - mc.Phi * angularCross(w_i, mc.F);
            m_dF[mc.parent] += dPhiF + mc.Phi * m_dF[k];
            m_eF[mc.parent] += mc.Phi * m_eF[k];
        }

        //---- Continuation above i -------------------------------------------
        // Outside subtree(i) every body term vanishes and so does dPhi, since
        // dPhi(k,c) needs i >= c; only the transmitted force keeps propagating,
        // and dH is zero so dtau loses its first term.
        for (MobilizedBodyIndex k : m_path[i]) {
            const MobodCache& mc = m_cache[k];
            for (int e = 0; e < mc.nu; ++e) {
                const UIndex ux(mc.ux+e);
                dtau_dq(ux, col) = dot(m_H[ux], m_dF[k]);
                dtau_du(ux, col) = dot(m_H[ux], m_eF[k]);
            }
            m_dF[mc.parent] += mc.Phi * m_dF[k];
            m_eF[mc.parent] += mc.Phi * m_eF[k];
        }
    }

    static SpatialVec reexpress(const Rotation& R, const SpatialVec& H)
    {   return SpatialVec(R*H[0], R*H[1]); }

    static Vec3 unitVec(int i) {Vec3 e(0); e[i] = 1; return e;}

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

    void calcBodyTransforms(MobilizedBodyIndex mbx) {
        MobodCache& mc = m_cache[mbx];
        const Transform X_FB = mc.X_FM * mc.X_MB;
        mc.X_PB = mc.X_PF * X_FB;
        mc.X_GB = m_cache[mc.parent].X_GB * mc.X_PB;
    }

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

    void calcJointIndependentKinematicsPos(MobilizedBodyIndex mbx) {
        MobodCache& mc = m_cache[mbx];
        const Vec3 p_PB_G = m_cache[mc.parent].X_GB.R() * mc.X_PB.p();
        mc.Phi = PhiMatrix(p_PB_G);

        const Rotation& R_GB = mc.X_GB.R();
        const UnitInertia G_Bo_G = mc.unitInertia_Bo_B.reexpress(~R_GB);
        const Vec3 p_BBc_G = R_GB * mc.com_B;
        mc.COM_G = mc.X_GB.p() + p_BBc_G;
        mc.Mk_G   = SpatialInertia(mc.mass, p_BBc_G, G_Bo_G);
        mc.rho_G  = R_GB * mc.p_BoMo_B;
    }

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
        mc.Mw = mc.Mk_G * SpatialVec(w_GB, Vec3(0));
    }

    void calcBodyAccelerationsFromUdotOutward(MobilizedBodyIndex mbx,
                                              const Vector& knownUdot) {
        MobodCache& mc = m_cache[mbx];
        SpatialVec A = ~mc.Phi * m_cache[mc.parent].A_GB;
        for (int d=0; d < mc.nu; ++d)
            A += m_H[UIndex(mc.ux+d)] * knownUdot[UIndex(mc.ux+d)];
        mc.A_GB = A + mc.mobilizerCoriolisAcceleration;
    }

    void calcInverseDynamicsPass2Inward(MobilizedBodyIndex         mbx,
                                        const Vector&              mobilityForces,
                                        const Vector_<SpatialVec>& bodyForces,
                                        Vector&                    tau) {
        MobodCache& mc = m_cache[mbx];
        mc.MA = mc.Mk_G * mc.A_GB;
        mc.F += mc.MA + mc.gyroscopicForce - bodyForces[mbx];
        for (int d=0; d < mc.nu; ++d) {
            const UIndex ux(mc.ux+d);
            tau[ux] = dot(m_H[ux], mc.F) - mobilityForces[ux];
        }
        m_cache[mc.parent].F += mc.Phi * mc.F;
    }

    const SimbodyMatterSubsystem&          m_matter;
    Array_<MobodCache, MobilizedBodyIndex> m_cache;
    Array_<SpatialVec, UIndex>             m_H_FM, m_H, m_HDot_FM, m_HDot;
    Vector                                 m_u;

    std::vector<std::vector<MobilizedBodyIndex>> m_subtree, m_path;

    Array_<SpatialVec, UIndex>             m_dH;
    Array_<SpatialVec, MobilizedBodyIndex> m_dV, m_dA, m_dF;
    Array_<SpatialVec, MobilizedBodyIndex> m_eV, m_eA, m_eF;
};






//==============================================================================
//                            OPERATOR SENSITIVITIES
//==============================================================================


struct Derivatives {
    Vector_<SpatialVec> dVw_dqdot;          // 18.28a
    Vector_<SpatialVec> dVwParent_dqdot;    // 18.28b
    Vector_<SpatialVec> dDeltaVw_dqdot;     // 18.28c
    Vector_<SpatialVec> dDeltaVv_dqdot;     // 18.28d
    Vector_<SpatialVec> dDeltaV_dqdot;      // 18.28e

    Vector_<SpatialMat> dPhi_dq;            // 18.29a
    Vector_<SpatialVec> dH_dq;              // 18.29b
    Vector_<SpatialMat> dM_dq;              // 18.29c

    Vector_<SpatialVec> dV_dq;
    Vector_<SpatialVec> dA_dq;
    Vector              dtau_dq;

    Vector_<SpatialVec> dV_du;
    Vector_<SpatialVec> dA_du;
    Vector              dtau_du;

    Derivatives(int nb, int nu)
    :   dVw_dqdot(nb, SpatialVec(Vec3(0), Vec3(0))),
        dVwParent_dqdot(nb, SpatialVec(Vec3(0), Vec3(0))),
        dDeltaVw_dqdot(nb, SpatialVec(Vec3(0), Vec3(0))),
        dDeltaVv_dqdot(nb, SpatialVec(Vec3(0), Vec3(0))),
        dDeltaV_dqdot(nb, SpatialVec(Vec3(0), Vec3(0))),
        dPhi_dq(nb, SpatialMat(Mat33(0), Mat33(0), Mat33(0), Mat33(0))),
        dH_dq(nu, SpatialVec(Vec3(0), Vec3(0))),
        dM_dq(nb, SpatialMat(Mat33(0), Mat33(0), Mat33(0), Mat33(0))),
        dV_dq(nb, SpatialVec(Vec3(0), Vec3(0))),
        dA_dq(nb, SpatialVec(Vec3(0), Vec3(0))),
        dtau_dq(nu, Real(0)),
        dV_du(nb, SpatialVec(Vec3(0), Vec3(0))),
        dA_du(nb, SpatialVec(Vec3(0), Vec3(0))),
        dtau_du(nu, Real(0)) {}

    static const int NumIdentities = 14;
    static const int NumJainIdentities = 8;

    Vec<NumIdentities> compare(const Derivatives& other) const {
        Vec<NumIdentities> e;
        e[0]  = maxAbsDiff(dVw_dqdot,       other.dVw_dqdot);
        e[1]  = maxAbsDiff(dVwParent_dqdot, other.dVwParent_dqdot);
        e[2]  = maxAbsDiff(dDeltaVw_dqdot,  other.dDeltaVw_dqdot);
        e[3]  = maxAbsDiff(dDeltaVv_dqdot,  other.dDeltaVv_dqdot);
        e[4]  = maxAbsDiff(dDeltaV_dqdot,   other.dDeltaV_dqdot);
        e[5]  = maxAbsDiff(dPhi_dq,         other.dPhi_dq);
        e[6]  = maxAbsDiff(dH_dq,           other.dH_dq);
        e[7]  = maxAbsDiff(dM_dq,           other.dM_dq);
        e[8]  = maxAbsDiff(dV_dq,           other.dV_dq);
        e[9]  = maxAbsDiff(dA_dq,           other.dA_dq);
        e[10] = maxAbsDiff(dtau_dq,         other.dtau_dq);
        e[11] = maxAbsDiff(dV_du,           other.dV_du);
        e[12] = maxAbsDiff(dA_du,           other.dA_du);
        e[13] = maxAbsDiff(dtau_du,         other.dtau_du);
        return e;
    }

    Vec<NumIdentities> magnitude() const {
        const Derivatives zero(dVw_dqdot.size(), dH_dq.size());
        return zero.compare(*this);
    }

    static Real maxAbsDiff(const Vector_<SpatialVec>& a,
                           const Vector_<SpatialVec>& b) {
        Real e = 0;
        for (int k = 0; k < a.size(); ++k)
            for (int r = 0; r < 2; ++r)
                for (int c = 0; c < 3; ++c)
                    e = std::max(e, std::abs(a[k][r][c] - b[k][r][c]));
        return e;
    }

    static Real maxAbsDiff(const Vector& a, const Vector& b) {
        Real e = 0;
        for (int k = 0; k < a.size(); ++k)
            e = std::max(e, std::abs(a[k] - b[k]));
        return e;
    }

    static Real maxAbsDiff(const Vector_<SpatialMat>& a,
                           const Vector_<SpatialMat>& b) {
        Real e = 0;
        for (int k = 0; k < a.size(); ++k)
            for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
                for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
                    e = std::max(e, std::abs(a[k](i,j)(r,c)
                                           - b[k](i,j)(r,c)));
        return e;
    }
};



Derivatives calcFiniteDifferences(const SimbodyMatterSubsystem& matter,
                                  const State& state,
                                  const Vector& knownUdot,
                                  const Vector& mobilityForces,
                                  const Vector_<SpatialVec>& bodyForces,
                                  MobilizedBodyIndex i, int d) {

    const MobilizedBody& mobod_i = matter.getMobilizedBody(i);
    SimTK_ERRCHK2_ALWAYS(d >= 0 && d < mobod_i.getNumU(state),
        "calcFiniteDifferences()",
        "Mobility %d is out of range for mobilized body %d.", d, (int)i);

    const Real h = 1e-5;
    const UIndex ux(mobod_i.getFirstUIndex(state) + d);
    Vector e(matter.getNumMobilities(), Real(0));
    e[ux] = 1;

    // Generalized speed (u) perturbation.
    State state_up = state;
    State state_um = state;
    state_up.updU()[ux] += h;
    state_um.updU()[ux] -= h;
    matter.getSystem().realize(state_up, Stage::Velocity);
    matter.getSystem().realize(state_um, Stage::Velocity);

    // Generalized coordinate (q) perturbation.
    Vector Ne;
    matter.multiplyByN(state, false, e, Ne);
    State state_qp = state;
    State state_qm = state;
    state_qp.updQ() += h*Ne;
    state_qm.updQ() -= h*Ne;
    matter.getSystem().realize(state_qp, Stage::Velocity);
    matter.getSystem().realize(state_qm, Stage::Velocity);

    // Body accelerations at the perturbed configurations, holding udot FIXED.
    // That is the right semantics for inverse dynamics: udot is a given input,
    // not something recomputed from the dynamics at each perturbed q. Only
    // Stage::Velocity is required.
    Vector_<SpatialVec> A_qp, A_qm;
    matter.calcBodyAccelerationFromUDot(state_qp, knownUdot, A_qp);
    matter.calcBodyAccelerationFromUDot(state_qm, knownUdot, A_qm);

    // The same at the u-perturbed states, for the dtau/du half. udot is held
    // fixed here too: tau is a function of (q, u, udot) and we want the partial
    // in u alone.
    Vector_<SpatialVec> A_up, A_um;
    matter.calcBodyAccelerationFromUDot(state_up, knownUdot, A_up);
    matter.calcBodyAccelerationFromUDot(state_um, knownUdot, A_um);

    // Compute finite differences.
    const int nb = matter.getNumBodies();
    Derivatives derivatives(nb, matter.getNumMobilities());
    for (int k = 1; k < nb; ++k) {
        const MobilizedBody& mobod_k =
            matter.getMobilizedBody(MobilizedBodyIndex(k));
        const MobilizedBody& parent_k = mobod_k.getParentMobilizedBody();

        // Body velocities.
        const SpatialVec& Vk_p = mobod_k.getBodyVelocity(state_up);
        const SpatialVec& Vk_m = mobod_k.getBodyVelocity(state_um);
        const SpatialVec& Vpk_p = parent_k.getBodyVelocity(state_up);
        const SpatialVec& Vpk_m = parent_k.getBodyVelocity(state_um);

        // V_GB = ~Phi V_GP + V_PB_G
        // --> V_PB_G = V_GB - ~Phi V_GP
        // --> DeltaV = Vk - ~Phi Vpk.
        const Vec3 p_PB_G = mobod_k.getBodyOriginLocation(state)
                          - parent_k.getBodyOriginLocation(state);
        const PhiMatrix Phi(p_PB_G);
        const PhiMatrixTranspose PhiT = ~Phi;
        const SpatialVec DeltaV_p = Vk_p - PhiT*Vpk_p;
        const SpatialVec DeltaV_m = Vk_m - PhiT*Vpk_m;
        const SpatialVec dDeltaV = (DeltaV_p - DeltaV_m) / (2*h);

        // dV^w(k) / dqdot_{i,d} (18.28a)
        derivatives.dVw_dqdot[k] =
            SpatialVec((Vk_p[0] - Vk_m[0]) / (2*h), Vec3(0));

        // dV^w(p(k)) / dqdot_{i,d} (18.28b)
        derivatives.dVwParent_dqdot[k] =
            SpatialVec((Vpk_p[0] - Vpk_m[0])/ (2*h), Vec3(0));

        // dDeltaV^w(k) / dqdot_{i,d} (18.28c)
        derivatives.dDeltaVw_dqdot[k] = SpatialVec(dDeltaV[0], Vec3(0));

        // dDeltaV^v(k) / dqdot_{i,d} (18.28d)
        derivatives.dDeltaVv_dqdot[k] = SpatialVec(Vec3(0), dDeltaV[1]);

        // dDeltaV(k) / dqdot_{i,d} (18.28e)
        derivatives.dDeltaV_dqdot[k] = dDeltaV;

        // dPhi(p(k),k) / dq_{i,d} (18.29a)
        const Vec3 lPlus  = mobod_k.getBodyOriginLocation(state_qp)
                          - parent_k.getBodyOriginLocation(state_qp);
        const Vec3 lMinus = mobod_k.getBodyOriginLocation(state_qm)
                          - parent_k.getBodyOriginLocation(state_qm);
        derivatives.dPhi_dq[k] = (PhiMatrix(lPlus).toSpatialMat()
                                - PhiMatrix(lMinus).toSpatialMat()) / (2*h);

        // dH*(k) / dq_{i,d} (18.29b)
        for (int e = 0; e < mobod_k.getNumU(state); ++e) {
            const MobilizerUIndex me(e);
            const UIndex uxk(mobod_k.getFirstUIndex(state) + e);
            derivatives.dH_dq[uxk] = (mobod_k.getHCol(state_qp, me)
                                    - mobod_k.getHCol(state_qm, me)) / (2*h);
        }

        // dM(k) / dq_{i,d} (18.29c)
        derivatives.dM_dq[k] =
            (mobod_k.getBodySpatialInertiaInGround(state_qp).toSpatialMat()
           - mobod_k.getBodySpatialInertiaInGround(state_qm).toSpatialMat())
            / (2*h);

        // dV(k) / dq_{i,d}
        derivatives.dV_dq[k] = (mobod_k.getBodyVelocity(state_qp)
                              - mobod_k.getBodyVelocity(state_qm)) / (2*h);

        // dA(k) / dq_{i,d}
        derivatives.dA_dq[k] = (A_qp[k] - A_qm[k]) / (2*h);

        // dV(k) / du_{i,d} and dA(k) / du_{i,d}
        derivatives.dV_du[k] = (mobod_k.getBodyVelocity(state_up)
                              - mobod_k.getBodyVelocity(state_um)) / (2*h);
        derivatives.dA_du[k] = (A_up[k] - A_um[k]) / (2*h);
    }

    // dtau/dq_{i,d}
    Vector tau_qp, tau_qm;
    matter.calcResidualForceIgnoringConstraints(state_qp, mobilityForces,
                                                bodyForces, knownUdot, tau_qp);
    matter.calcResidualForceIgnoringConstraints(state_qm, mobilityForces,
                                                bodyForces, knownUdot, tau_qm);
    for (int ux = 0; ux < matter.getNumMobilities(); ++ux)
        derivatives.dtau_dq[UIndex(ux)] = (tau_qp[ux] - tau_qm[ux]) / (2*h);

    // dtau/du_{i,d}
    Vector tau_up, tau_um;
    matter.calcResidualForceIgnoringConstraints(state_up, mobilityForces,
                                                bodyForces, knownUdot, tau_up);
    matter.calcResidualForceIgnoringConstraints(state_um, mobilityForces,
                                                bodyForces, knownUdot, tau_um);
    for (int ux = 0; ux < matter.getNumMobilities(); ++ux)
        derivatives.dtau_du[UIndex(ux)] = (tau_up[ux] - tau_um[ux]) / (2*h);

    return derivatives;
}

Derivatives calcSensitivities(const SimbodyMatterSubsystem& matter,
                              const State& state,
                              const Vector& knownUdot,
                              const Vector_<SpatialVec>& bodyForces,
                              MobilizedBodyIndex i, int d) {

    auto isAncestorOf = [&](MobilizedBodyIndex i, MobilizedBodyIndex k) {
        for (MobilizedBodyIndex b = k; b != 0;
                b = matter.getMobilizedBody(b).getParentMobilizedBody()
                        .getMobilizedBodyIndex()) {
            if (b == i) return true;
        }
        return false;
    };

    const MobilizedBody& mobod_i = matter.getMobilizedBody(i);
    SimTK_ERRCHK2_ALWAYS(d >= 0 && d < mobod_i.getNumU(state),
        "calcSensitivities()",
        "Mobility %d is out of range for mobilized body %d.", d, (int)i);

    const SpatialVec& H_i = mobod_i.getHCol(state, MobilizerUIndex(d));

    Vector_<SpatialVec> A_GB;
    matter.calcBodyAccelerationFromUDot(state, knownUdot, A_GB);

    const int nb = matter.getNumBodies();
    Derivatives derivatives(nb, matter.getNumMobilities());
    for (MobilizedBodyIndex k(1); k < nb; ++k) {
        const MobilizedBody& mobod_k = matter.getMobilizedBody(k);
        const MobilizedBody& parent_k = mobod_k.getParentMobilizedBody();
        const MobilizedBodyIndex p_k = parent_k.getMobilizedBodyIndex();
        const Transform X_PB = mobod_k.getInboardFrame(state)
                             * mobod_k.getMobilizerTransform(state)
                             * ~mobod_k.getOutboardFrame(state);
        const Vec3 p_PB_G = parent_k.getBodyRotation(state) * X_PB.p();
        const PhiMatrix Phi(p_PB_G);

        // dV^w(k) / dqdot_{i,d} = H*_w(i) 1_[i >= k]            (18.28a)
        if (isAncestorOf(i, k)) {
            derivatives.dVw_dqdot[k] = SpatialVec(H_i[0], Vec3(0));
        }

        // dV^w(p(k)) / dqdot_{i,d} = H*_w(i) 1_[i > k]          (18.28b)
        if (isAncestorOf(i, p_k)) {
            derivatives.dVwParent_dqdot[k] = SpatialVec(H_i[0], Vec3(0));
        }

        // dDeltaV^w(k) / dqdot_{i,d} = H*_w(i) 1_[i == k]       (18.28c)
        if (i == k) {
            derivatives.dDeltaVw_dqdot[k] = SpatialVec(H_i[0], Vec3(0));
        }

        // dDeltaV^v(k) / dqdot_{i,d} = H*_v(i) 1_[i == k]       (18.28d)
        if (i == k) {
            derivatives.dDeltaVv_dqdot[k] = SpatialVec(Vec3(0), H_i[1]);
        }

        // dDeltaV(k) / dqdot_{i,d} = H*(i) 1_[i == k]           (18.28e)
        if (i == k) {
            derivatives.dDeltaV_dqdot[k] = H_i;
        }

        // Htilde*_w(i)
        const SpatialMat tilde_Hw(crossMat(H_i[0]), Mat33(0),
                                  Mat33(0),         crossMat(H_i[0]));

        // dPhi(p(k), k) / dq_{i,d}                              (18.29a)
        if (i == k) {
            derivatives.dPhi_dq[k] =
                SpatialMat(Mat33(0), crossMat(H_i[1]),
                           Mat33(0), Mat33(0));

        } else if (isAncestorOf(i, p_k)) {
            derivatives.dPhi_dq[k] = tilde_Hw*Phi - Phi*tilde_Hw;
        }

        // dH*_B(k) / dq_{i,d}                                   (18.29b)
        const Vec3 p_BoMo_G = mobod_k.getBodyRotation(state) *
                              mobod_k.getOutboardFrame(state).p();
        for (int e = 0; e < mobod_k.getNumU(state); ++e) {
            const UIndex uxk(mobod_k.getFirstUIndex(state) + e);
            const SpatialVec& H_k = mobod_k.getHCol(state, MobilizerUIndex(e));

            if (isAncestorOf(i, p_k)) {
                derivatives.dH_dq[uxk] = SpatialVec(H_i[0] % H_k[0],
                                                    H_i[0] % H_k[1]);
            } else if (i == k) {
                derivatives.dH_dq[uxk] =
                    SpatialVec(Vec3(0), -(H_k[0] % (H_i[0] % p_BoMo_G)));
            }
        }

        // dM(k) / dq_{i,d}                                      (18.29c)
        if (isAncestorOf(i, k)) {
            const SpatialMat Mk =
                mobod_k.getBodySpatialInertiaInGround(state).toSpatialMat();
            derivatives.dM_dq[k] = tilde_Hw*Mk - Mk*tilde_Hw;
        }

        // dV(k) / dq_{i,d}
        const SpatialVec& V_GP = parent_k.getBodyVelocity(state);
        SpatialVec dV = (~derivatives.dPhi_dq[k]) * V_GP
                      + (~Phi) * derivatives.dV_dq[p_k];
        for (int e = 0; e < mobod_k.getNumU(state); ++e) {
            const UIndex uxk(mobod_k.getFirstUIndex(state) + e);
            dV += derivatives.dH_dq[uxk] * state.getU()[uxk];
        }
        derivatives.dV_dq[k] = dV;

        // dA(k) / dq_{i,d} at fixed udot
        //
        // Differentiate Simbody's own acceleration recursion
        //     A(k) = ~Phi(p,k) A(p) + sum_e H(k,e) udot(k,e) + a_c(k)
        //
        // Simbody's a_c is exactly
        //     a_c = (0, wp x (v_k - v_p)) + (wp x Vr[0], wp x Vr[1])
        //                                 + (0, -wr x (wr x r))
        const SpatialVec& Vp = V_GP;
        const SpatialVec& dVp = derivatives.dV_dq[p_k];
        const SpatialVec& Vk = mobod_k.getBodyVelocity(state);
        const SpatialVec& dVk = derivatives.dV_dq[k];

        SpatialVec Vr(Vec3(0), Vec3(0)), dVr(Vec3(0), Vec3(0));
        for (int e = 0; e < mobod_k.getNumU(state); ++e) {
            const UIndex uxk(mobod_k.getFirstUIndex(state) + e);
            Vr  += mobod_k.getHCol(state, MobilizerUIndex(e))
                   * state.getU()[uxk];
            dVr += derivatives.dH_dq[uxk] * state.getU()[uxk];
        }

        const Vec3 r  = mobod_k.getBodyRotation(state)
                        * mobod_k.getOutboardFrame(state).p();
        const Vec3 dr = isAncestorOf(i, k) ? Vec3(H_i[0] % r) : Vec3(0);

        const Vec3& wp = Vp[0];  const Vec3& dwp = dVp[0];
        const Vec3& wr = Vr[0];  const Vec3& dwr = dVr[0];

        const SpatialVec da_c =
              SpatialVec(Vec3(0), dwp % (Vk[1] - Vp[1])
                                + wp  % (dVk[1] - dVp[1]))
            + SpatialVec(dwp % Vr[0] + wp % dVr[0],
                         dwp % Vr[1] + wp % dVr[1])
            + SpatialVec(Vec3(0), -(dwr % (wr % r))
                                - (wr % (dwr % r))
                                - (wr % (wr % dr)));

        SpatialVec dA = (~derivatives.dPhi_dq[k]) * A_GB[p_k]
                      + (~Phi) * derivatives.dA_dq[p_k]
                      + da_c;
        for (int e = 0; e < mobod_k.getNumU(state); ++e) {
            const UIndex uxk(mobod_k.getFirstUIndex(state) + e);
            dA += derivatives.dH_dq[uxk] * knownUdot[uxk];
        }
        derivatives.dA_dq[k] = dA;

        // dV(k)/du_{i,d} and dA(k)/du_{i,d}
        //
        // The u-side is structurally the same sweep but with fewer terms: M,
        // Phi, H and r depend on q alone, so none of them contributes, and
        // udot is a fixed input. Differentiating the same two recursions,
        //     dV(k)/du = ~Phi dV(p)/du + H(k,d) 1[i == k]
        //     dA(k)/du = ~Phi dA(p)/du + da_c(k)/du
        // and da_c/du drops the dr term of \eqref{da_c/dq} because r is
        // configuration-only.
        const SpatialVec& eVp = derivatives.dV_du[p_k];
        SpatialVec eVk = (~Phi) * eVp;
        if (i == k) eVk += H_i;
        derivatives.dV_du[k] = eVk;

        SpatialVec eVr(Vec3(0), Vec3(0));
        if (i == k) eVr = H_i;   // dVr/du_{i,d} = H(k,d) when i == k

        const Vec3& ewp = eVp[0];
        const Vec3& ewr = eVr[0];

        const SpatialVec da_c_du =
              SpatialVec(Vec3(0), ewp % (Vk[1] - Vp[1])
                                + wp  % (eVk[1] - eVp[1]))
            + SpatialVec(ewp % Vr[0] + wp % eVr[0],
                         ewp % Vr[1] + wp % eVr[1])
            + SpatialVec(Vec3(0), -(ewr % (wr % r))
                                - (wr % (ewr % r)));

        derivatives.dA_du[k] = (~Phi) * derivatives.dA_du[p_k] + da_c_du;
    }

    // Inward pass: the composite transmitted force and dtau/dq.
    //
    // Differentiating Simbody's second inverse-dynamics pass,
    //     F(k)     = M(k) A(k) + b(k) - f_app(k) + sum_c Phi(k,c) F(c)
    //     tau(k,e) = H(k,e) . F(k) - tau_app(k,e)
    // term by term gives
    //     dF(k)     = dM(k) A(k) + M(k) dA(k) + db(k)
    //                 + sum_c [ dPhi(k,c) F(c) + Phi(k,c) dF(c) ]
    //     dtau(k,e) = dH(k,e) . F(k) + H(k,e) . dF(k)
    Vector_<SpatialVec> F(nb, SpatialVec(Vec3(0), Vec3(0)));
    Vector_<SpatialVec> dF(nb, SpatialVec(Vec3(0), Vec3(0)));
    Vector_<SpatialVec> eF(nb, SpatialVec(Vec3(0), Vec3(0)));  // d/du
    for (int kk = nb-1; kk >= 1; --kk) {
        const MobilizedBodyIndex k(kk);
        const MobilizedBody& mb_k = matter.getMobilizedBody(k);
        const MobilizedBodyIndex pk =
            mb_k.getParentMobilizedBody().getMobilizedBodyIndex();

        const SpatialMat M =
            mb_k.getBodySpatialInertiaInGround(state).toSpatialMat();
        const SpatialMat& dM = derivatives.dM_dq[k];
        const Vec3& w  = mb_k.getBodyVelocity(state)[0];
        const Vec3& dw = derivatives.dV_dq[k][0];

        // Gyroscopic force. Simbody's b(k) is exactly W(w) (M (w,0)), verified
        // to 3e-17 against getGyroscopicForce, which is what lets db(k) be
        // built from dV and 18.29c with no further primitives.
        const SpatialVec P  = M * SpatialVec(w, Vec3(0));
        const SpatialVec dP = dM * SpatialVec(w, Vec3(0))
                            + M  * SpatialVec(dw, Vec3(0));
        const SpatialVec b  = angularCross(w, P);
        const SpatialVec db = angularCross(dw, P) + angularCross(w, dP);

        // Same force, differentiated in u. M carries no u-dependence, so db/du
        // keeps only the two terms in which w itself varies.
        const Vec3& ew = derivatives.dV_du[k][0];
        const SpatialVec eP = M * SpatialVec(ew, Vec3(0));
        const SpatialVec eb = angularCross(ew, P) + angularCross(w, eP);

        F[k]  += M * A_GB[k] + b - bodyForces[k];
        dF[k] += dM * A_GB[k] + M * derivatives.dA_dq[k] + db;
        eF[k] += M * derivatives.dA_du[k] + eb;

        for (int e = 0; e < mb_k.getNumU(state); ++e) {
            const UIndex uxk(mb_k.getFirstUIndex(state) + e);
            const SpatialVec H_ke = mb_k.getHCol(state, MobilizerUIndex(e));
            derivatives.dtau_dq[uxk] =
                  dot(derivatives.dH_dq[uxk], F[k])
                + dot(H_ke, dF[k]);
            // No [dH/du].F term: H depends on q alone.
            derivatives.dtau_du[uxk] = dot(H_ke, eF[k]);
        }

        const Transform X_PB_k = mb_k.getInboardFrame(state)
                               * mb_k.getMobilizerTransform(state)
                               * ~mb_k.getOutboardFrame(state);
        const PhiMatrix PhiK(matter.getMobilizedBody(pk).getBodyRotation(state)
                             * X_PB_k.p());
        F[pk]  += PhiK * F[k];
        dF[pk] += derivatives.dPhi_dq[k] * F[k] + PhiK * dF[k];
        eF[pk] += PhiK * eF[k];   // Phi is configuration-only
    }

    return derivatives;
}

int main() {
    PendulumSystem pendulum;
    pendulum.setArbitraryState();

    const SimbodyMatterSubsystem& matter = pendulum.getMatter();
    const State& state = pendulum.getState();

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

    // The optimized in-class Jacobians, checked against the per-direction
    // reference implementation. They evaluate the same expressions, differing
    // only in what is hoisted out of the column loop and in skipping bodies
    // whose sensitivities are structurally zero, so they should agree to a few
    // ulp -- not bit-exact, because the products are associated differently and
    // the force accumulation sums the same terms in a different order. Anything
    // larger than ~1e-14 would mean the subtree restriction dropped a term that
    // is not actually zero.
    {
        Matrix dtau_dq_opt, dtau_du_opt;
        id.calcInverseDynamicsDerivatives(knownUdot, dtau_dq_opt, dtau_du_opt);

        Real worstQ = 0, worstU = 0;
        for (MobilizedBodyIndex i(1); i < matter.getNumBodies(); ++i) {
            const MobilizedBody& mb = matter.getMobilizedBody(i);
            for (int d = 0; d < mb.getNumU(state); ++d) {
                const int col = mb.getFirstUIndex(state) + d;
                const Derivatives ref =
                    calcSensitivities(matter, state, knownUdot, bodyForces,
                                      i, d);
                for (int ux = 0; ux < matter.getNumMobilities(); ++ux) {
                    worstQ = std::max(worstQ,
                        std::abs(dtau_dq_opt(ux,col)-ref.dtau_dq[UIndex(ux)]));
                    worstU = std::max(worstU,
                        std::abs(dtau_du_opt(ux,col)-ref.dtau_du[UIndex(ux)]));
                }
            }
        }
        printf("\n  optimized vs per-direction reference:"
               "  dtau/dq %9.2e   dtau/du %9.2e\n", worstQ, worstU);
    }

    // Timing. The reference path is what the optimized method replaces:
    // one full sweep per perturbation direction, each rebuilding the
    // acceleration and force passes from scratch. Note the reference also
    // allocates a Derivatives per call, so the ratio below conflates allocation
    // with arithmetic and overstates the arithmetic win on a model this small.
    {
        using Clock = std::chrono::steady_clock;
        const int reps = 2000;

        auto t0 = Clock::now();
        volatile Real sink = 0;
        for (int r = 0; r < reps; ++r)
            for (MobilizedBodyIndex i(1); i < matter.getNumBodies(); ++i) {
                const MobilizedBody& mb = matter.getMobilizedBody(i);
                for (int d = 0; d < mb.getNumU(state); ++d) {
                    const Derivatives ref = calcSensitivities(
                        matter, state, knownUdot, bodyForces, i, d);
                    sink = sink + ref.dtau_dq[UIndex(0)];
                }
            }
        auto t1 = Clock::now();

        Matrix dq, du;
        for (int r = 0; r < reps; ++r) {
            id.calcInverseDynamics(knownUdot, mobilityForces, bodyForces, tau);
            id.calcInverseDynamicsDerivatives(knownUdot, dq, du);
            sink = sink + dq(0,0);
        }
        auto t2 = Clock::now();

        const Real usRef = std::chrono::duration<Real, std::micro>(t1-t0).count()
                           / reps;
        const Real usOpt = std::chrono::duration<Real, std::micro>(t2-t1).count()
                           / reps;
        printf("  full Jacobian per call:  reference %7.2f us   optimized"
               " %7.2f us   (%.1fx)\n", usRef, usOpt, usRef/usOpt);
    }

    // The new SimbodyMatterSubsystem operator, checked against this file's
    // verified implementation. The operator differentiates the same recursion
    // using Simbody's own caches, so it should agree to a few ulp.
    {
        Matrix dq_api, du_api, dq_opt, du_opt;
        matter.calcResidualForceDerivativesIgnoringConstraints(
            state, bodyForces, knownUdot, dq_api, du_api);
        id.calcInverseDynamicsDerivatives(knownUdot, dq_opt, du_opt);

        Real worstQ = 0, worstU = 0;
        for (int r = 0; r < matter.getNumMobilities(); ++r)
            for (int c = 0; c < matter.getNumMobilities(); ++c) {
                worstQ = std::max(worstQ, std::abs(dq_api(r,c)-dq_opt(r,c)));
                worstU = std::max(worstU, std::abs(du_api(r,c)-du_opt(r,c)));
            }
        printf("\n  SimbodyMatterSubsystem operator vs sandbox:"
               "  dtau/dq %9.2e   dtau/du %9.2e\n", worstQ, worstU);
    }

    static const char* names[Derivatives::NumIdentities] =
        {"18.28a", "18.28b", "18.28c", "18.28d", "18.28e", "18.29a", "18.29b",
         "18.29c", "dV/dq", "dA/dq", "dtau/dq",
         "dV/du", "dA/du", "dtau/du"};

    Array_<Vec<Derivatives::NumIdentities>> errs, mags;
    Array_<std::pair<int,int>> coords;
    Vec<Derivatives::NumIdentities> worst(0), scale(0);
    for (MobilizedBodyIndex i(1); i < matter.getNumBodies(); ++i) {
        const MobilizedBody& mobod = matter.getMobilizedBody(i);
        for (int d = 0; d < mobod.getNumU(state); ++d) {
            const Derivatives analytical =
                calcSensitivities(matter, state, knownUdot, bodyForces, i, d);
            const Derivatives fd =
                calcFiniteDifferences(matter, state, knownUdot, mobilityForces,
                                      bodyForces, i, d);
            const Vec<Derivatives::NumIdentities> e = analytical.compare(fd);
            const Vec<Derivatives::NumIdentities> m = analytical.magnitude();
            errs.push_back(e); mags.push_back(m);
            coords.push_back(std::make_pair((int)i, d));
            for (int t = 0; t < Derivatives::NumIdentities; ++t) {
                worst[t] = std::max(worst[t], e[t]);
                scale[t] = std::max(scale[t], m[t]);
            }
        }
    }

    const int groups[3] = {0, Derivatives::NumJainIdentities,
                           Derivatives::NumIdentities};
    const char* titles[2] = {"Jain operator sensitivities",
                             "Assembled quantities (Simbody convention)"};
    for (int g = 0; g < 2; ++g) {
        const int first = groups[g], last = groups[g+1];
        printf("\n  %s\n  max|analytic - fd| over all bodies, per coordinate"
               "\n\n    i  d", titles[g]);
        for (int t = first; t < last; ++t) printf("  %10s", names[t]);
        printf("\n");
        for (unsigned r = 0; r < errs.size(); ++r) {
            printf("   %2d %2d", coords[r].first, coords[r].second);
            for (int t = first; t < last; ++t) printf("  %10.2e", errs[r][t]);
            printf("\n");
        }
        printf("   -----");
        for (int t = first; t < last; ++t) printf("  ----------");
        printf("\n    max ");
        for (int t = first; t < last; ++t) printf("  %10.2e", worst[t]);
        printf("\n  scale");
        for (int t = first; t < last; ++t) printf("  %10.2e", scale[t]);
        printf("\n");
    }
    printf("\n  'scale' is max|analytic| over all i,d -- an identity whose\n"
           "  error and scale are both tiny is not actually being tested.\n\n");

}
