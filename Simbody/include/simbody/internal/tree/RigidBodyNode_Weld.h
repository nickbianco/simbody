#ifndef SimTK_SIMBODY_RIGID_BODY_NODE_WELD_H_
#define SimTK_SIMBODY_RIGID_BODY_NODE_WELD_H_

/* -------------------------------------------------------------------------- *
 *                               Simbody(tm)                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the SimTK biosimulation toolkit originating from           *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org/home/simbody.  *
 *                                                                            *
 * Portions copyright (c) 2005-12 Stanford University and the Authors.        *
 * Authors: Michael Sherman                                                   *
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

/**@file
 * This file contains the RigidBodyNodes which have no degrees of freedom --
 * Ground and Weld mobilizers. These cannot be derived from the usual
 * RigidBodyNodeSpec<dof> class because dof==0 is problematic there. Also,
 * these can have very efficient implementations here since they know they
 * have no dofs. They are templatized on the scalar type T.
 */

#include "RigidBodyNode.h"

/**
 * This still-abstract class is the common base for any MobilizedBody which
 * has no mobilities. Currently that is only the unique Ground body and
 * MobilizedBody::Weld, but it is conceivable that others could crop up.
 *
 * The base class overrides all the virtual methods which deal in q's and u's
 * so have common, trivial implementations if there are no q's and no u's.
 */
template <class T>
class ImmobileRigidBodyNode_ : public RigidBodyNode_<T> {
public:
    SimTK_RBNODE_SCALAR_TYPEDEFS(T);
    typedef RigidBodyNode_<T> Base;
    SimTK_RBNODE_BASE_MEMBERS;

    ImmobileRigidBodyNode_(const MassPropertiesP& mProps_B, const UIndex& uIx,
                           const USquaredIndex& usqIx, const QIndex& qIx)
    :   RigidBodyNode_<T>(mProps_B, Base::QDotIsAlwaysTheSameAsU,
                          Base::QuaternionIsNeverUsed)
    {
        uIndex   = uIx;
        uSqIndex = usqIx;
        qIndex   = qIx;
    }
    int getPosPoolSize(const SBStateDigest&) const {return 0;}
    int getVelPoolSize(const SBStateDigest&) const {return 0;}

    ~ImmobileRigidBodyNode_() override = default;

    int  getDOF()   const override {return 0;}
    int  getMaxNQ() const override {return 0;}
    int  getNQInUse(const SBModelVars&) const override {return 0;}
    int  getNUInUse(const SBModelVars&) const override {return 0;}
    bool isUsingQuaternion(const SBStateDigest&, 
                           MobilizerQIndex& ix) const {   ix.invalidate(); return false; }


    int calcQPoolSize(const SBModelVars&) const override {return 0;}

    void performQPrecalculations(const SBStateDigest_<T>& sbs,
                                 const RealP* q, int nq,
                                 RealP* qCache,  int nQCache,
                                 RealP* qErr,    int nQErr) const override
    {   assert(nq==0 && nQCache==0 && nQErr==0); }


    // An immobile mobilizer holds the mobilized body's M frame coincident 
    // with the parent body's F frame forever.
    void calcX_FM(const SBStateDigest_<T>& sbs,
                  const RealP* q,      int nq,
                  const RealP* qCache, int nQCache,
                  TransformP&  X_FM) const override
    {   assert(nq==0 && nQCache==0);
        X_FM.setToZero(); }

    void setQToFitTransformImpl(const SBStateDigest& sbs, const Transform& X_FM, 
                                Vector& q) const {}
    void setQToFitRotationImpl(const SBStateDigest& sbs, const Rotation& R_FM, 
                               Vector& q) const {}
    void setQToFitTranslationImpl(const SBStateDigest& sbs, const Vec3& p_FM, 
                                  Vector& q) const {}

    void setUToFitVelocityImpl
       (const SBStateDigest& sbs, const Vector& q, const SpatialVec& V_FM, 
        Vector& u) const {}
    void setUToFitAngularVelocityImpl
       (const SBStateDigest& sbs, const Vector& q, const Vec3& w_FM, 
        Vector& u) const {}
    void setUToFitLinearVelocityImpl
       (const SBStateDigest& sbs, const Vector& q, const Vec3& v_FM, 
        Vector& u) const {}

    
    void multiplyByN(const SBStateDigest_<T>&, bool matrixOnRight, 
                     const RealP* in, RealP* out) const override {}
    void multiplyByNInv(const SBStateDigest_<T>&, bool matrixOnRight,
                        const RealP* in, RealP* out) const override {}
    void multiplyByNDot(const SBStateDigest_<T>&, bool matrixOnRight,
                        const RealP* in, RealP* out) const override {}

    void calcQDot(const SBStateDigest_<T>&, const RealP* udot, 
                  RealP* qdotdot) const override {}
    void calcQDotDot(const SBStateDigest_<T>&, const RealP* udot, 
                     RealP* qdotdot) const override {}

    bool enforceQuaternionConstraints(
        const SBStateDigest& sbs,
        Vector&             q,
        Vector&             qErrest) const {return false;}

    void convertToEulerAngles(const Vector& inputQ, 
                              Vector&       outputQ) const {}
    void convertToQuaternions(const Vector& inputQ, 
                              Vector&       outputQ) const {}
};

/* This is the distinguished body representing the immobile ground frame. Other 
bodies may be fixed to this one, but only this is the actual Ground. */
template <class T>
class RBGroundBody_ : public ImmobileRigidBodyNode_<T> {
public:
    SimTK_RBNODE_SCALAR_TYPEDEFS(T);
    typedef RigidBodyNode_<T> Base;
    SimTK_RBNODE_BASE_MEMBERS;

    RBGroundBody_()
    :   ImmobileRigidBodyNode_<T>(groundMassProperties(),
                                  UIndex(0), USquaredIndex(0), QIndex(0)) {}

    // Ground's mass properties are never used in computations.
    static MassPropertiesP groundMassProperties() {
        if constexpr (std::is_same<T,Real>::value)
            return MassProperties(Infinity, Vec3(0), Inertia(Infinity));
        else
            return MassPropertiesP(RealP(Infinity), Vec3P(0),
                                   UnitInertiaP(RealP(Infinity)));
    }

    const char* type() const override { return "ground"; }

    // TODO: should ground set the various cache entries here?
    void realizeModel   (SBStateDigest&) const {}
    void realizeInstance(const SBStateDigest_<T>& sbs) const override {
        // Initialize cache entries that will never be changed at later stages.
        
        SBTreeVelocityCache_<T>& vc = sbs.updTreeVelocityCache();
        SBDynamicsCache_<T>& dc = sbs.updDynamicsCache();
        SBTreeAccelerationCache_<T>& ac = sbs.updTreeAccelerationCache();
        updY(dc) = SpatialMatP(Mat33P(0));
        updA_GB(ac) = 0;
    }
    void realizePosition(const SBStateDigest_<T>&) const override {}
    void realizeVelocity(const SBStateDigest_<T>&) const override {}
    void realizeDynamics(const SBStateDigest&) const {}
    // There is no realizeAcceleration().
    void realizeReport  (const SBStateDigest&) const {}

    // Ground's "composite" body inertia is still the infinite mass
    // and inertia it started with; no need to look at the children.
    // This overrides the base class default implementation.
    void calcCompositeBodyInertiasInward(
        const SBTreePositionCache_<T>&                  pc,
        Array_<SpatialInertiaP,MobilizedBodyIndex>&  R) const override
    {   R[GroundIndex] = SpatialInertiaP(Infinity, Vec3P(0), UnitInertiaP(1)); }

    // Ground's "articulated" body inertia is still the infinite mass and
    // inertia it started with; no need to look at the children.
    void realizeArticulatedBodyInertiasInward(
        const SBInstanceCache_<T>&,
        const SBTreePositionCache_<T>&,
        SBArticulatedBodyInertiaCache_<T>& abc) const override 
    {   ArticulatedInertiaP& P = updP(abc);
        P = ArticulatedInertiaP(SymMat33P(RealP(Infinity)), Mat33P(RealP(Infinity)), 
                                       SymMat33P(0)); 
        updPPlus(abc) = P;
    }

    void realizeYOutward(
        const SBInstanceCache&,
        const SBTreePositionCache&,
        const SBArticulatedBodyInertiaCache&,
        SBDynamicsCache&                        dc) const {
    }


    // Treat Ground as though welded to the universe at the ground
    // origin. The reaction there collects the effects of all the
    // base bodies and of any forces applied directly to Ground.
    void calcUDotPass1Inward(
        const SBInstanceCache_<T>&     ic,
        const SBTreePositionCache_<T>& pc,
        const SBArticulatedBodyInertiaCache_<T>&,
        const SBArticulatedBodyVelocityCache_<T>&,
        const RealP*                jointForces,
        const SpatialVecP*          bodyForces,
        const RealP*                allUDot,
        SpatialVecP*                allZ,
        SpatialVecP*                allZPlus,
        RealP*                      allEpsilon) const override
    {
        const SpatialVecP& F            = bodyForces[0];
        SpatialVecP&       z            = allZ[0];
        SpatialVecP&       zPlus        = allZPlus[0];

        z = -F;

        for (unsigned i=0; i<children.size(); ++i) {
            const PhiMatrixP&  phiChild   = children[i]->getPhi(pc);
            const SpatialVecP& zPlusChild = allZPlus[children[i]->getNodeNum()];
            z += phiChild * zPlusChild; // 18 flops
        }

        zPlus = z;
    } 

    void calcUDotPass2Outward(
        const SBInstanceCache_<T>&,
        const SBTreePositionCache_<T>&,
        const SBArticulatedBodyInertiaCache_<T>&,
        const SBTreeVelocityCache_<T>&,
        const SBDynamicsCache_<T>&,
        const RealP*                epsilonTmp,
        SpatialVecP*                allA_GB,
        RealP*                      allUDot,
        RealP*                      allTau) const override
    {
        allA_GB[0] = 0;
    }

    // Ground doesn't contribute to M^-1*f. Inward pass does nothing since
    // Ground can't be the child of any body.
    void multiplyByMInvPass1Inward(
        const SBInstanceCache_<T>&     ic,
        const SBTreePositionCache_<T>& pc,
        const SBArticulatedBodyInertiaCache_<T>&,
        const RealP*                f,
        SpatialVecP*                allZ,
        SpatialVecP*                allZPlus,
        RealP*                      allEpsilon) const override
    {
    } 

    // Outward pass must make sure A_GB[0] is zero so it can be propagated
    // outwards properly.
    void multiplyByMInvPass2Outward(
        const SBInstanceCache_<T>&,
        const SBTreePositionCache_<T>&,
        const SBArticulatedBodyInertiaCache_<T>&,
        const RealP*                 epsilonTmp,
        SpatialVecP*                 allA_GB,
        RealP*                       allUDot) const override
    {
        allA_GB[0] = 0;
    }

    // Also serves as pass 1 for inverse dynamics.
    void calcBodyAccelerationsFromUdotOutward(
        const SBTreePositionCache_<T>&  pc,
        const SBTreeVelocityCache_<T>&  vc,
        const RealP*                 allUDot,
        SpatialVecP*                 allA_GB) const override
    {
        allA_GB[0] = 0;
    }

    // Here Ground is the last body processed. Although it has no mobility forces
    // we can still collect up all the forces from the base bodies to Ground
    // in case anyone cares.
    void calcInverseDynamicsPass2Inward(
        const SBTreePositionCache_<T>&  pc,
        const SBTreeVelocityCache_<T>&  vc,
        const SpatialVecP*           allA_GB,
        const RealP*                 jointForces,
        const SpatialVecP*           bodyForces,
        SpatialVecP*                 allF,
        RealP*                       allTau) const override
    {
        allF[0] = -bodyForces[0];

        // Add in forces on base bodies, shifted to Ground.
        for (unsigned i=0; i<children.size(); ++i) {
            const PhiMatrixP&  phiChild  = children[i]->getPhi(pc);
            const SpatialVecP& FChild    = allF[children[i]->getNodeNum()];
            allF[0] += phiChild * FChild;
        }

        // no taus
    }

    void multiplyByMPass1Outward(
        const SBTreePositionCache_<T>&  pc,
        const RealP*                 allUDot,
        SpatialVecP*                 allA_GB) const override
    {
        allA_GB[0] = 0;
    }

    void multiplyByMPass2Inward(
        const SBTreePositionCache_<T>&  pc,
        const SpatialVecP*           allA_GB,
        SpatialVecP*                 allF,
        RealP*                       allTau) const override
    {
        allF[0] = 0;

        // Add in forces on base bodies, shifted to Ground.
        for (unsigned i=0; i<children.size(); ++i) {
            const PhiMatrixP&  phiChild  = children[i]->getPhi(pc);
            const SpatialVecP& FChild    = allF[children[i]->getNodeNum()];
            allF[0] += phiChild * FChild;
        }

        // no taus
    }


    void multiplyBySystemJacobian(
        const SBTreePositionCache_<T>&  pc,
        const RealP*                 v,
        SpatialVecP*                 Jv) const override    
    {
        Jv[0] = SpatialVecP(Vec3P(0));
    }

    void multiplyBySystemJacobianTranspose(
        const SBTreePositionCache_<T>&  pc, 
        SpatialVecP*                 zTmp,
        const SpatialVecP*           X, 
        RealP*                       JtX) const override 
    {
        zTmp[0] = X[0];
        for (unsigned i=0; i<children.size(); ++i) {
            const SpatialVecP& zChild   = zTmp[children[i]->getNodeNum()];
            const PhiMatrixP&  phiChild = children[i]->getPhi(pc);
            zTmp[0] += phiChild * zChild;
        }
        // No generalized speeds so no contribution to JtX.
    }

    void calcEquivalentJointForces(
        const SBTreePositionCache_<T>&  pc,
        const SBTreeVelocityCache_<T>&,
        const SpatialVecP*           bodyForces,
        SpatialVecP*                 allZ,
        RealP*                       jointForces) const override 
    { 
        allZ[0] = bodyForces[0];
        for (unsigned i=0; i<children.size(); ++i) {
            const SpatialVecP& zChild    = allZ[children[i]->getNodeNum()];
            const PhiMatrixP&  phiChild  = children[i]->getPhi(pc);
            allZ[0] += phiChild * zChild; 
        }
    }
};

    // WELD //

// This is a "joint" with no degrees of freedom, that simply forces
// the two reference frames to be identical. A Weld node always has a parent
// but has no q's and no u's.
template <class T>
class RBNodeWeld_ : public ImmobileRigidBodyNode_<T> {
public:
    SimTK_RBNODE_SCALAR_TYPEDEFS(T);
    typedef RigidBodyNode_<T> Base;
    SimTK_RBNODE_BASE_MEMBERS;

    RBNodeWeld_(const MassPropertiesP& mProps_B, const UIndex& uIx,
                const USquaredIndex& usqIx, const QIndex& qIx)
    :   ImmobileRigidBodyNode_<T>(mProps_B, uIx, usqIx, qIx) {}

    const char* type() const override { return "weld"; }

    void realizeModel(SBStateDigest& sbs) const {}
    void realizeInstance(const SBStateDigest_<T>& sbs) const override {
        // Initialize cache entries that will never be changed at later stages.
        
        SBTreeVelocityCache_<T>& vc = sbs.updTreeVelocityCache();
        SBTreeAccelerationCache_<T>& ac = sbs.updTreeAccelerationCache();
        updV_FM(vc) = 0;
        updV_PB_G(vc) = 0;
        updVD_PB_G(vc) = 0;
    }

    void realizePosition(const SBStateDigest_<T>& sbs) const override {
        const SBInstanceVars_<T>& iv = sbs.getInstanceVars();
        const SBInstanceCache_<T>& ic = sbs.getInstanceCache();
        SBTreePositionCache_<T>& pc = sbs.updTreePositionCache();

        const TransformP& X_MB = getX_MB(ic); // fixed
        const TransformP& X_PF = getX_PF(iv); // fixed
        const TransformP& X_GP = getX_GP(pc); // already calculated

        updX_FM(pc).setToZero();
        updX_PB(pc) = X_PF * X_MB;
        updX_GB(pc) = X_GP * getX_PB(pc);
        const Vec3P p_PB_G = getX_GP(pc).R() * getX_PB(pc).p();

        // The Phi matrix conveniently performs child-to-parent (inward) shifting
        // on spatial quantities (forces); its transpose does parent-to-child
        // (outward) shifting for velocities.
        updPhi(pc) = PhiMatrixP(p_PB_G);

        // Calculate spatial mass properties. That means we need to transform
        // the local mass moments into the Ground frame and reconstruct the
        // spatial inertia matrix Mk.

        const RotationP& R_GB = getX_GB(pc).R();
        const Vec3P&     p_GB = getX_GB(pc).p();

        // reexpress inertia in ground (57 flops)
        const UnitInertiaP G_Bo_G  = getUnitInertia_OB_B().reexpress(~R_GB);
        const Vec3P        p_BBc_G = R_GB*getCOM_B(); // 15 flops

        updCOM_G(pc) = p_GB + p_BBc_G; // 3 flops

        // Calc Mk: the spatial inertia matrix about the body origin.
        // Note: we need to calculate this now so that we'll be able to calculate
        // kinetic energy without going past the Velocity stage.
        updMk_G(pc) = SpatialInertiaP(getMass(), p_BBc_G, G_Bo_G);
    }
    
    void realizeVelocity(const SBStateDigest_<T>& sbs) const override {
        const SBTreePositionCache_<T>& pc = sbs.getTreePositionCache();
        SBTreeVelocityCache_<T>& vc = sbs.updTreeVelocityCache();
        calcJointIndependentKinematicsVel(pc,vc);
    }

    void realizeDynamics(const SBStateDigest&) const {
    }

    // There is no realizeAcceleration().

    void realizeReport(const SBStateDigest& sbs) const {}

    // Weld uses base class implementation of calcCompositeBodyInertiasInward() since
    // that is independent of mobilities.

    void realizeArticulatedBodyInertiasInward
       (const SBInstanceCache_<T>&          ic,
        const SBTreePositionCache_<T>&      pc, 
        SBArticulatedBodyInertiaCache_<T>&  abc) const override 
    {
        ArticulatedInertiaP& P = updP(abc);
        P = ArticulatedInertiaP(getMk_G(pc));
        for (unsigned i=0 ; i<children.size() ; i++) {
            const PhiMatrixP&          phiChild   = children[i]->getPhi(pc);
            const ArticulatedInertiaP& PPlusChild = children[i]->getPPlus(abc);

            P += PPlusChild.shift(phiChild.l());
        }
        updPPlus(abc) = P;
    }


    void realizeYOutward(
        const SBInstanceCache&,
        const SBTreePositionCache&              pc,
        const SBArticulatedBodyInertiaCache&    abc,
        SBDynamicsCache&                        dc) const {
        // This psi actually has the wrong sign, but it doesn't matter since we 
        // multiply by it twice.

        SpatialMat psi = getPhi(pc).toSpatialMat();
        updY(dc) = ~psi * parent->getY(dc) * psi;
    }

    
    void calcUDotPass1Inward(
        const SBInstanceCache_<T>&,
        const SBTreePositionCache_<T>&              pc,
        const SBArticulatedBodyInertiaCache_<T>&,
        const SBArticulatedBodyVelocityCache_<T>&   abvc,
        const RealP*,
        const SpatialVecP*                       bodyForces,
        const RealP*,
        SpatialVecP*                             allZ,
        SpatialVecP*                             allZPlus,
        RealP*) const override 
    {
        const SpatialVecP& myBodyForce  = bodyForces[nodeNum];
        SpatialVecP&       z            = allZ[nodeNum];
        SpatialVecP&       zPlus        = allZPlus[nodeNum];

        z = getArticulatedBodyCentrifugalForces(abvc) - myBodyForce;

        for (unsigned i=0; i<children.size(); ++i) {
            const PhiMatrixP&  phiChild   = children[i]->getPhi(pc);
            const SpatialVecP& zPlusChild = allZPlus[children[i]->getNodeNum()];

            z += phiChild * zPlusChild;                 // 18 flops
        }

        zPlus = z;
    }

    void calcUDotPass2Outward(
        const SBInstanceCache_<T>&,
        const SBTreePositionCache_<T>&  pc,
        const SBArticulatedBodyInertiaCache_<T>&,
        const SBTreeVelocityCache_<T>&  vc,
        const SBDynamicsCache_<T>&      dc,
        const RealP*                 allEpsilon,
        SpatialVecP*                 allA_GB,
        RealP*                       allUDot,
        RealP*                       allTau) const override
    {
        SpatialVecP& A_GB = allA_GB[nodeNum];

        const PhiMatrixP&    phi = getPhi(pc);
        const SpatialVecP&   a   = getMobilizerCoriolisAcceleration(vc);

        // Shift parent's acceleration outward (Ground==0). 12 flops
        const SpatialVecP& A_GP  = allA_GB[parent->getNodeNum()]; 
        const SpatialVecP  APlus = ~phi * A_GP;

        A_GB = APlus + a;  // no udot for weld
    }
    
    // A weld doesn't have udots but we still have to calculate z, zPlus,
    // for use by the parent of this body.
    void multiplyByMInvPass1Inward(
        const SBInstanceCache_<T>&      ic,
        const SBTreePositionCache_<T>&  pc,
        const SBArticulatedBodyInertiaCache_<T>&,
        const RealP*                 f,
        SpatialVecP*                 allZ,
        SpatialVecP*                 allZPlus,
        RealP*                       allEpsilon) const override
    {
        SpatialVecP& z       = allZ[nodeNum];
        SpatialVecP& zPlus   = allZPlus[nodeNum];

        z = 0;

        for (unsigned i=0; i<children.size(); i++) {
            const PhiMatrixP&  phiChild  = children[i]->getPhi(pc);
            const SpatialVecP& zPlusChild = allZPlus[children[i]->getNodeNum()];
            z += phiChild * zPlusChild; // 18 flops
        }

        zPlus = z;
    }

    // Must set A_GB properly for propagation to children.
    void multiplyByMInvPass2Outward(
        const SBInstanceCache_<T>&,
        const SBTreePositionCache_<T>&  pc,
        const SBArticulatedBodyInertiaCache_<T>&,
        const RealP*                 allEpsilon,
        SpatialVecP*                 allA_GB,
        RealP*                       allUDot) const override
    {
        SpatialVecP&      A_GB = allA_GB[nodeNum];
        const PhiMatrixP& phi  = getPhi(pc);

        // Shift parent's acceleration outward (Ground==0). 12 flops
        const SpatialVecP& A_GP  = allA_GB[parent->getNodeNum()]; 
        const SpatialVecP  APlus = ~phi * A_GP;

        A_GB = APlus;
    }

    // Also serves as pass 1 for inverse dynamics.
    void calcBodyAccelerationsFromUdotOutward(
        const SBTreePositionCache_<T>&  pc,
        const SBTreeVelocityCache_<T>&  vc,
        const RealP*                 allUDot,
        SpatialVecP*                 allA_GB) const override 
    {
        SpatialVecP& A_GB = allA_GB[nodeNum];

        // Shift parent's A_GB outward. (Ground A_GB is zero.)
        const SpatialVecP A_GP = ~getPhi(pc) * allA_GB[parent->getNodeNum()];

        A_GB = A_GP + getMobilizerCoriolisAcceleration(vc); // no udot for weld
    }

    void calcInverseDynamicsPass2Inward(
        const SBTreePositionCache_<T>&  pc,
        const SBTreeVelocityCache_<T>&  vc,
        const SpatialVecP*           allA_GB,
        const RealP*                 jointForces,
        const SpatialVecP*           bodyForces,
        SpatialVecP*                 allF,
        RealP*                       allTau) const override
    {
        const SpatialVecP& myBodyForce   = bodyForces[nodeNum];
        const SpatialVecP& A_GB          = allA_GB[nodeNum];
        SpatialVecP&       F             = allF[nodeNum];

        // Start with rigid body force from desired body acceleration and
        // gyroscopic forces due to angular velocity, minus external forces
        // applied directly to this body.
        F = getMk_G(pc)*A_GB + getGyroscopicForce(vc) - myBodyForce;

        // Add in forces on children, shifted to this body.
        for (unsigned i=0; i<children.size(); ++i) {
            const PhiMatrixP&  phiChild  = children[i]->getPhi(pc);
            const SpatialVecP& FChild    = allF[children[i]->getNodeNum()];
            F += phiChild * FChild;
        }

        // no taus.
    }

    void multiplyByMPass1Outward(
        const SBTreePositionCache_<T>&  pc,
        const RealP*                 allUDot,
        SpatialVecP*                 allA_GB) const override
    {
        SpatialVecP& A_GB = allA_GB[nodeNum];

        // Shift parent's A_GB outward. (Ground A_GB is zero.)
        const SpatialVecP A_GP = ~getPhi(pc) * allA_GB[parent->getNodeNum()];

        A_GB = A_GP;  
    }

    void multiplyByMPass2Inward(
        const SBTreePositionCache_<T>&  pc,
        const SpatialVecP*           allA_GB,
        SpatialVecP*                 allF,   // temp
        RealP*                       allTau) const override
    {
        const SpatialVecP& A_GB  = allA_GB[nodeNum];
        SpatialVecP&       F     = allF[nodeNum];

        F = getMk_G(pc)*A_GB;

        for (int i=0 ; i<(int)children.size() ; i++) {
            const PhiMatrixP&  phiChild  = children[i]->getPhi(pc);
            const SpatialVecP& FChild    = allF[children[i]->getNodeNum()];
            F += phiChild * FChild;
        }
    }

    void multiplyBySystemJacobian(
        const SBTreePositionCache_<T>&  pc,
        const RealP*                 v,
        SpatialVecP*                 Jv) const override    
    {
        SpatialVecP& out = Jv[nodeNum];

        // Shift parent's result outward (ground result is 0).
        const SpatialVecP outP = ~getPhi(pc) * Jv[parent->getNodeNum()];

        out = outP;  
    }

    void multiplyBySystemJacobianTranspose(
        const SBTreePositionCache_<T>&  pc, 
        SpatialVecP*                 zTmp,
        const SpatialVecP*           X, 
        RealP*                       JtX) const override
    {
        const SpatialVecP& in  = X[getNodeNum()];
        SpatialVecP&       z   = zTmp[getNodeNum()];

        z = in;

        for (unsigned i=0; i<children.size(); ++i) {
            const SpatialVecP& zChild   = zTmp[children[i]->getNodeNum()];
            const PhiMatrixP&  phiChild = children[i]->getPhi(pc);
            z += phiChild * zChild;
        }
        // No generalized speeds so no contribution to JtX.
    }

    void calcEquivalentJointForces(
        const SBTreePositionCache_<T>&  pc,
        const SBTreeVelocityCache_<T>&  vc,
        const SpatialVecP*           bodyForces,
        SpatialVecP*                 allZ,
        RealP*                       jointForces) const override 
    {
        const SpatialVecP& myBodyForce  = bodyForces[nodeNum];
        SpatialVecP&       z            = allZ[nodeNum];

        // Centrifugal forces are MA+b where M is body spatial inertia,
        // A is total coriolis acceleration, and b is gyroscopic force.
        z = myBodyForce - getTotalCentrifugalForces(vc);

        for (int i=0 ; i<(int)children.size() ; i++) {
            const SpatialVecP& zChild    = allZ[children[i]->getNodeNum()];
            const PhiMatrixP&  phiChild  = children[i]->getPhi(pc);
            z += phiChild * zChild; 
        }
    }
};


// The Ground node is special because it doesn't need a mobilizer.

typedef ImmobileRigidBodyNode_<Real> ImmobileRigidBodyNode;
typedef RBGroundBody_<Real>          RBGroundBody;
typedef RBNodeWeld_<Real>            RBNodeWeld;

#endif // SimTK_SIMBODY_RIGID_BODY_NODE_WELD_H_
