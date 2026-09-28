#ifndef SimTK_SIMBODY_RIGID_BODY_NODE_SPEC_H_
#define SimTK_SIMBODY_RIGID_BODY_NODE_SPEC_H_

/* -------------------------------------------------------------------------- *
 *                               Simbody(tm)                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the SimTK biosimulation toolkit originating from           *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org/home/simbody.  *
 *                                                                            *
 * Portions copyright (c) 2005-15 Stanford University and the Authors.        *
 * Authors: Michael Sherman                                                   *
 * Contributors:                                                              *
 *    Charles Schwieters (NIH): wrote the public domain IVM code from which   *
 *                              this was derived.                             *
 *    Peter Eastman: wrote the Euler Angle<->Quaternion conversion            *
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
 * This file contains just the templatized class RigidBodyNodeSpec_<T,dof> which
 * is used as the base class for most of the RigidBodyNodes (that is, the
 * implementations of Mobilizers). The only exceptions are nodes whose 
 * mobilizers provide no degrees of freedom -- Ground and Weld.
 *
 * This file contains all the multibody mechanics method declarations that 
 * involve a single body and its mobilizer (inboard joint), that is, one node 
 * in the multibody tree. These methods constitute the inner loops of the 
 * multibody calculations, and much suffering is undergone here to make them 
 * run fast. In particular most calculations are templatized by the number of
 * mobilities, so that compile-time sizes are known for everything. They are
 * also templatized by the scalar type T; RigidBodyNodeSpec<dof> is the
 * default Real precision version.
 *
 * Most methods here expect to be called in a particular order during traversal 
 * of the tree -- either base to tip or tip to base.
 */

#include "RigidBodyNode.h"

/**
 * This still-abstract class is a skeleton implementation of a built-in 
 * mobilizer, with the number of mobilities (dofs, u's) specified as a template 
 * parameter. That way all the code that is common except for the dimensionality
 * of the mobilizer can be written once, and the compiler generates specific 
 * implementations for each of the six possible dimensionalities (1-6 
 * mobilities). Each implementation works only on fixed size Vec<> and Mat<> 
 * types, so can use very high speed inline operators.
 */
template <class T, int dof, bool noR_FM=false>
class RigidBodyNodeSpec_ : public RigidBodyNode_<T> {
public:
SimTK_RBNODE_SCALAR_TYPEDEFS(T);
typedef RigidBodyNode_<T> Base;
using typename Base::QDotHandling;
using typename Base::QuaternionUse;
using Base::QDotIsAlwaysTheSameAsU;
using Base::QDotMayDifferFromU;
using Base::QuaternionIsNeverUsed;
using Base::QuaternionMayBeUsed;
SimTK_RBNODE_BASE_MEMBERS;


RigidBodyNodeSpec_(const MassProperties_<T>& mProps_B,
                  UIndex&               nextUSlot,
                  USquaredIndex&        nextUSqSlot,
                  QIndex&               nextQSlot,
                  QDotHandling          qdotHandling,
                  QuaternionUse         quaternionUse,
                  bool                  isReversed)
:   RigidBodyNode_<T>(mProps_B, qdotHandling, quaternionUse, isReversed)
{
    // don't call any virtual methods in here!
    uIndex   = nextUSlot;
    uSqIndex = nextUSqSlot;
    qIndex   = nextQSlot;
}

void updateSlots(UIndex& nextUSlot, USquaredIndex& nextUSqSlot, 
                 QIndex& nextQSlot) {
    // OK to call virtual method here.
    nextUSlot   += getDOF();
    nextUSqSlot += getDOF()*getDOF();
    nextQSlot   += getMaxNQ();
}

//------------------------------------------------------------------------------
// These override RigidBodyNode virtuals. Note that not all pure virtuals
// are overridden here; RigidBodyNodeSpec is still abstract.   

~RigidBodyNodeSpec_() override = default;

int getDOF() const override {return dof;}
int getMaxNQ() const override {
    assert(quaternionUse == QuaternionIsNeverUsed);
    return dof; // maxNQ can be larger than dof if there's a quaternion
}

// Default method must be overridden if there is a chance that NQ
// might not be the same as NU.
int getNQInUse(const SBModelVars&) const override {
    assert(quaternionUse == QuaternionIsNeverUsed); 
    return dof; // DOF <= NQ <= maxNQ
}

// Currently NU is always just the Mobilizer's DOFs (the template argument)
// Later we may want to offer modeling options to lock joints, or perhaps
// break them.
int getNUInUse(const SBModelVars&) const override {
    return dof;
}

// Available only for the default Real precision.
// Default method must be overridden if this mobilizer might use a
// quaternion under some conditions.
bool isUsingQuaternion(const SBStateDigest&, 
                       MobilizerQIndex& startOfQuaternion) const {
    assert(quaternionUse == QuaternionIsNeverUsed);
    startOfQuaternion.invalidate();
    return false;
}

// Concrete mobilizer must override calcQDot() and calcQDotDot() if qdot might 
// not be be the same as u under *any* circumstance.

// This method should calculate qdot=N*u, where N=N(q) is the kinematic
// coupling matrix. State digest should be at Stage::Position.
void calcQDot(const SBStateDigest_<T>&, 
              const RealP* u, RealP* qdot) const override 
{
    assert(qdotHandling == QDotIsAlwaysTheSameAsU);
    Vec<dof,T>::updAs(qdot) = Vec<dof,T>::getAs(u); // default says qdot=u
}

// This method should calculate qdotdot=N*udot + NDot*u, where N=N(q),
// NDot=NDot(q,u). State digest should be at Stage::Velocity.
void calcQDotDot(const SBStateDigest_<T>&, 
                         const RealP* udot, RealP* qdotdot) const override
{
    assert(qdotHandling == QDotIsAlwaysTheSameAsU);
    Vec<dof,T>::updAs(qdotdot) = Vec<dof,T>::getAs(udot); // default: qdotdot=udot
}

// State digest should be at Stage::Position for calculating N (the matrix that
// maps generalized speeds u to coordinate derivatives qdot, qdot=Nu). The 
// default implementation assumes nq==nu and the nqXnu block of N corresponding 
// to this mobilizer is identity. Then either operation (regardless of side) 
// just copies nu numbers from in to out.
//
// THIS MUST BE OVERRIDDEN by any mobilizer for which nq != nu, or qdot != u.
void multiplyByN(const SBStateDigest_<T>&, bool matrixOnRight,  
                 const RealP* in, RealP* out) const override
{
    assert(qdotHandling == QDotIsAlwaysTheSameAsU);
    Vec<dof,T>::updAs(out) = Vec<dof,T>::getAs(in);
}

void multiplyByNInv(const SBStateDigest_<T>&, bool matrixOnRight, 
                    const RealP* in, RealP* out) const override
{
    assert(qdotHandling == QDotIsAlwaysTheSameAsU);
    Vec<dof,T>::updAs(out) = Vec<dof,T>::getAs(in);
}

// State digest should be at Stage::Velocity for calculating NDot (the matrix 
// that is used in mapping generalized accelerations udot to coordinate 2nd 
// derivatives qdotdot, qdotdot=N udot + NDot u. The default implementation 
// assumes nq==nu and the nqXnu block of N corresponding to this mobilizer is 
// identity, so NDot is an nuXnu block of zeroes. Then either operation 
// (regardless of side) just copies nu zeros to out.
//
// THIS MUST BE OVERRIDDEN by any mobilizer for which nq != nu, or qdot != u.
void multiplyByNDot(const SBStateDigest_<T>&, bool matrixOnRight, 
                    const RealP* in, RealP* out) const override
{
    assert(qdotHandling == QDotIsAlwaysTheSameAsU);
    Vec<dof,T>::updAs(out) = 0;
}

// Available only for the default Real precision.
// Return true if any change is made to the q vector.
bool enforceQuaternionConstraints(
    const SBStateDigest&    sbs,
    Vector&                 q,
    Vector&                 qErrest) const
{
    assert(quaternionUse == QuaternionIsNeverUsed);
    return false;
}


void convertToEulerAngles(const Vector& inputQ, 
                          Vector& outputQ) const {
    // The default implementation just copies Q.  Subclasses may override this.
    assert(quaternionUse == QuaternionIsNeverUsed);
    toQ(outputQ) = fromQ(inputQ);
}

void convertToQuaternions(const Vector& inputQ, 
                          Vector& outputQ) const {
    // The default implementation just copies Q.  Subclasses may override this.
    assert(quaternionUse == QuaternionIsNeverUsed);
    toQ(outputQ) = fromQ(inputQ);
}

// These routines give each node a chance to set appropriate defaults in a piece
// of the state corresponding to a particular stage. Default implementations here
// assume non-ball joint; override if necessary.
//      setMobilizerDefault{Model,Instance,Time,Dynamics,Acceleration}Values
//          retain base class do-nothing default implementation.
void setMobilizerDefaultPositionValues(const SBModelVars& s, 
                                       Vector& q) const 
{   toQ(q) = 0; }

void setMobilizerDefaultVelocityValues(const SBModelVars&, 
                                       Vector& u) const 
{   toU(u) = 0; }

// Available only for the default Real precision.
void realizeModel(SBStateDigest& sbs) const 
{
}

void realizeInstance(const SBStateDigest_<T>& sbs) const override
{
}

// Set a new configuration and calculate the consequent kinematics.
// Must call base-to-tip.
void realizePosition(const SBStateDigest_<T>& sbs) const override 
{
    const SBModelVars&      mv   = sbs.getModelVars();
    const SBModelCache&     mc   = sbs.getModelCache();
    const SBInstanceVars_<T>&   iv   = sbs.getInstanceVars();
    const SBInstanceCache_<T>&  ic   = sbs.getInstanceCache();
    const auto&             allQ = sbs.getQ(); // Vector or const T*
    SBTreePositionCache_<T>&    pc   = sbs.updTreePositionCache();
    auto&&                  allQErr = sbs.updQErr(); // Vector& or T*

    // Mobilizer specific.
    
    const SBModelPerMobodInfo& mbInfo = getModelInfo(mc);

    // First perform precalculations on these new q's, such as stashing away
    // sines and cosines of angles. We'll put all the results into the State's
    // position cache for later use. If there are local constraints on the q's
    // (currently that means quaternion normalization constraints), then we
    // calculate those here and put the result in the appropriate slot of qerr.

    const int nq=mbInfo.nQInUse, nqpool=mbInfo.nQPoolInUse,
              nqerr=(mbInfo.hasQuaternionInUse ? 1 : 0);
    const RealP* q0    = nq     ? &allQ[mbInfo.firstQIndex]                : 0;
    RealP*       qpool0= nqpool ? &pc.mobilizerQCache[mbInfo.startInQPool] : 0;
    RealP*       qerr0 = nqerr  ? &allQErr[ic.firstQuaternionQErrSlot
                                          + mbInfo.quaternionPoolIndex]     
                               : 0;
    performQPrecalculations(sbs, q0, nq, qpool0, nqpool, qerr0, nqerr);

    // Now that we've done the necessary precalculations, calculate the cross-
    // mobilizer transform X_FM without recalculating anything. For reversed 
    // mobilizers we have to do our own reversal here because mobilizers don't
    // handle that themselves.

    if (isReversed()) {
        TransformP X_MF;
        calcX_FM(sbs, q0, nq, qpool0, nqpool, X_MF);
        updX_FM(pc) = ~X_MF;
    } else 
        calcX_FM(sbs, q0, nq, qpool0, nqpool, updX_FM(pc));

    // With X_FM in the cache, and X_GP for the parent already calculated (we're doing
    // an outward pass), we can calculate X_PB and X_GB now.
    calcBodyTransforms(iv, ic, pc, updX_PB(pc), updX_GB(pc));

    // Here we do allow the mobilizer to calculate the reversed H matrix, but the
    // default implementation of the reversed method just calls the forward method
    // and then reverses it. For some mobilizers that is unreasonably expensive.
    // REMINDER: our H matrix definition is transposed from Jain and Schwieters.
    if (isReversed()) calcReverseMobilizerH_FM       (sbs, updH_FM(pc));
    else              calcAcrossJointVelocityJacobian(sbs, updH_FM(pc));

    // Here we're using the cross-mobilizer hinge matrix H_FM that we just 
    // calculated to compute H(==H_PB_G), the equivalent hinge matrix between 
    // the parent's body frame P and child's body frame B, but expressed in
    // Ground. (F is fixed on P and M is fixed on B.)
    calcParentToChildVelocityJacobianInGround(mv, iv, ic, pc, updH(pc));

    // Mobilizer independent.
    calcJointIndependentKinematicsPos(pc);
}

// Set new velocities for the current configuration, and calculate
// all the velocity-dependent terms. Must call base-to-tip.
// This routine may assume that *all* position 
// kinematics (not just joint-specific) has been done for this node,
// that all velocity kinematics has been done for the parent, and
// that the velocity state variables (u) are available. The
// quantities that must be computed are:
//   V_FM   relative velocity of B's M frame in P's F frame, 
//             expressed in F (note: this is also V_PM_F since
//             F is fixed on P). (This is H_FM*u.)
//   V_PB_G  relative velocity of B in P (==H*u), expr. in G
//   HDot_FM time derivative of hinge matrix H_FM
//   HDot    time derivative of H (==H_PB_G) hinge matrix relating body frames
//   VD_PB_G acceleration remainder term HDot*u, expr. in G
// The code is the same for all joints, although parametrized by ndof.
void realizeVelocity(const SBStateDigest_<T>& sbs) const override
{
    const SBModelVars&          mv = sbs.getModelVars();
    const SBInstanceVars_<T>&       iv = sbs.getInstanceVars();
    const SBInstanceCache_<T>&      ic = sbs.getInstanceCache();
    const SBTreePositionCache_<T>&  pc = sbs.getTreePositionCache();
    SBTreeVelocityCache_<T>&        vc = sbs.updTreeVelocityCache();
    const auto&                 allU = sbs.getU(); // Vector or const T*
    const Vec<dof,T>&             u = fromU(allU);

    // Mobilizer specific.
    calcQDot(sbs, &allU[uIndex], &sbs.updQDot()[qIndex]);

    updV_FM(vc)    = getH_FM(pc) * u;   // 6*dof flops
    updV_PB_G(vc)  = getH(pc)    * u;   // 6*dof flops

    // REMINDER: our H matrix definition is transposed from Jain and Schwieters.
    if (isReversed()) calcReverseMobilizerHDot_FM       (sbs, updHDot_FM(vc));
    else              calcAcrossJointVelocityJacobianDot(sbs, updHDot_FM(vc));

    // Here we're using the cross-mobilizer hinge matrix derivative HDot_FM 
    // that we just calculated to compute HDot(==HDot_PB_G), the derivative
    // of H (==H_PB_G) between the parent's body frame P and child's body 
    // frame B. The derivative is taken in the Ground frame and the result
    // is expressed in Ground. (F is fixed on P and M is fixed on B.)
    calcParentToChildVelocityJacobianInGroundDot(mv, iv, ic, pc, vc,
                                                 updHDot(vc));
    updVD_PB_G(vc) = getHDot(vc) * u;   // 6*dof flops

    // Mobilizer independent.
    calcJointIndependentKinematicsVel(pc,vc);
}

// Available only for the default Real precision.
void realizeDynamics(const SBStateDigest&) const
{
    // nothing to do
}

// There is no realizeAcceleration().

void realizeReport(const SBStateDigest& sbs) const
{
}

// Use base class implementations of calcCompositeBodyInertiasInward()
// and realizeArticulatedBodyVelocityCache() since they are independent of 
// mobilities.

// This is a dynamics-stage calculation and must be called tip-to-base (inward).
void realizeArticulatedBodyInertiasInward(
    const SBInstanceCache_<T>&          ic,
    const SBTreePositionCache_<T>&      pc,
    SBArticulatedBodyInertiaCache_<T>&  abc) const override;

// Available only for the default Real precision.
// This dynamics-stage calculation is needed for handling constraints. It
// must be called base-to-tip (outward);
void realizeYOutward(
    const SBInstanceCache&                  ic,
    const SBTreePositionCache&              pc,
    const SBArticulatedBodyInertiaCache&    abc,
    SBDynamicsCache&                        dc) const;

void multiplyBySystemJacobian(
    const SBTreePositionCache_<T>&  pc,
    const RealP*                 v,
    SpatialVecP*                 Jv) const override;

void multiplyBySystemJacobianTranspose(
    const SBTreePositionCache_<T>&  pc, 
    SpatialVecP*                 zTmp,
    const SpatialVecP*           X, 
    RealP*                       JtX) const override;

void calcEquivalentJointForces(
    const SBTreePositionCache_<T>&  pc,
    const SBTreeVelocityCache_<T>&  vc,
    const SpatialVecP*           bodyForces,
    SpatialVecP*                 allZ,
    RealP*                       jointForces) const override;

void calcUDotPass1Inward(
    const SBInstanceCache_<T>&      ic,
    const SBTreePositionCache_<T>&  pc,
    const SBArticulatedBodyInertiaCache_<T>&,
    const SBArticulatedBodyVelocityCache_<T>&,
    const RealP*                 jointForces,
    const SpatialVecP*           bodyForces,
    const RealP*                 allUDot,
    SpatialVecP*                 allZ,
    SpatialVecP*                 allGepsilon,
    RealP*                       allEpsilon) const override;

void calcUDotPass2Outward(
    const SBInstanceCache_<T>&      ic,
    const SBTreePositionCache_<T>&  pc,
    const SBArticulatedBodyInertiaCache_<T>&,
    const SBTreeVelocityCache_<T>&  vc,
    const SBDynamicsCache_<T>&      dc,
    const RealP*                 epsilonTmp,
    SpatialVecP*                 allA_GB,
    RealP*                       allUDot,
    RealP*                       allTau) const override;

void multiplyByMInvPass1Inward(
    const SBInstanceCache_<T>&      ic,
    const SBTreePositionCache_<T>&  pc,
    const SBArticulatedBodyInertiaCache_<T>&,
    const RealP*                 f,
    SpatialVecP*                 allZ,
    SpatialVecP*                 allGepsilon,
    RealP*                       allEpsilon) const override;

void multiplyByMInvPass2Outward(
    const SBInstanceCache_<T>&      ic,
    const SBTreePositionCache_<T>&  pc,
    const SBArticulatedBodyInertiaCache_<T>&,
    const RealP*                 epsilonTmp,
    SpatialVecP*                 allA_GB,
    RealP*                       allUDot) const override;

// Also serves as pass 1 for inverse dynamics.
void calcBodyAccelerationsFromUdotOutward(
    const SBTreePositionCache_<T>&  pc,
    const SBTreeVelocityCache_<T>&  vc,
    const RealP*                 allUDot,
    SpatialVecP*                 allA_GB) const override;

void calcInverseDynamicsPass2Inward(
    const SBTreePositionCache_<T>&  pc,
    const SBTreeVelocityCache_<T>&  vc,
    const SpatialVecP*           allA_GB,
    const RealP*                 jointForces,
    const SpatialVecP*           bodyForces,
    SpatialVecP*                 allFTmp,
    RealP*                       allTau) const override; 

void multiplyByMPass1Outward(
    const SBTreePositionCache_<T>&  pc,
    const RealP*                 allUDot,
    SpatialVecP*                 allA_GB) const override;
void multiplyByMPass2Inward(
    const SBTreePositionCache_<T>&  pc,
    const SpatialVecP*           allA_GB,
    SpatialVecP*                 allFTmp,
    RealP*                       allTau) const override;

// Get a column of H_PB_G, which is what Jain calls H* and Schwieters calls H^T.
const SpatialVecP& 
getHCol(const SBTreePositionCache_<T>& pc, int j) const override {
    return getH(pc)(j);
}

// Get a column of H_FM the local cross-mobilizer hinge matrix expressed in the
// parent (inboard) mobilizer frame F.
const SpatialVecP& 
getH_FMCol(const SBTreePositionCache_<T>& pc, int j) const override {
    return getH_FM(pc)(j);
}

// Provide default implementations for setQToFitTransformImpl() and 
// setQToFitVelocityImpl() which are implemented using the rotational and 
// translational quantity routines. These assume that the rotational and 
// translational coordinates are independent, with rotation handled first and 
// then left alone. If a mobilizer type needs to deal with rotation and
// translation simultaneously, it should provide a specific implementation for 
// these two routines. *Each* mobilizer must implement 
// setQToFit{Rotation,Translation} and 
// setUToFit{AngularVelocity,LinearVelocity}; there are no defaults.

// Available only for the default Real precision.
void setQToFitTransformImpl(const SBStateDigest& sbs, const Transform& X_FM, 
                            Vector& q) const {
    this->setQToFitRotationImpl   (sbs,X_FM.R(),q);
    this->setQToFitTranslationImpl(sbs,X_FM.p(),q);
}

void setUToFitVelocityImpl(const SBStateDigest& sbs, const Vector& q, 
                           const SpatialVec& V_FM, Vector& u) const {
    this->setUToFitAngularVelocityImpl(sbs,q,V_FM[0],u);
    this->setUToFitLinearVelocityImpl (sbs,q,V_FM[1],u);
}

// End of RigidBodyNode overrides.
//------------------------------------------------------------------------------


// The following routines calculate joint-specific position kinematic
// quantities. They may assume that *all* position kinematics (not just
// joint-specific) has been done for the parent, and that the position
// state variables q are available. Each routine may also assume that the
// previous routines have been called, in the order below.
// The routines are structured as operators -- they use the State but
// do not change anything in it, including the cache. Instead they are
// passed arguments to write their results into. In practice, these
// arguments will typically be in the State cache (see below).

// This is the type of the joint transition matrix H, but our definition
// of H is transposed from Jain's or Schwieters'. That is, what we're 
// calling H here is what Jain calls H* and Schwieters calls H^T. So
// our H matrix is 6 x nu, or more usefully it is 2 rows of Vec3's.
// The first row define how u's contribute to angular velocities; the
// second defines how u's contribute to linear velocities.
typedef Mat<2,dof,Vec3P> HType;

// This mandatory routine calculates the joint transition matrix H_FM, giving
// the change of velocity induced by the generalized speeds u for this 
// mobilizer, expressed in the mobilizer's inboard "fixed" frame F (attached to
// this body's parent). 
// This method constitutes the *definition* of the generalized speeds for
// a particular joint.
// This routine can depend on X_FM having already
// been calculated and available in the PositionCache but must NOT depend
// on any quantities involving Ground or other bodies.
// Note: this calculates the H matrix *as defined*; we might reverse
// the frames in use. So we call this H_F0M0 while the possibly reversed
// version is H_FM. Be sure to use X_F0M0 in your calculation for H_F0M0
// if you need access to the cross-mobilizer transform.
// CAUTION: our definition of H is transposed from Jain and Schwieters.
virtual void calcAcrossJointVelocityJacobian(
    const SBStateDigest_<T>& sbs,
    HType&               H_F0M0) const = 0;

// This mandatory routine calculates the time derivative taken in F of
// the matrix H_FM (call it HDot_FM). This is zero if the generalized
// speeds are all defined in terms of the F frame, which is often the case.
// This routine can depend on X_FM and H_FM being available already in
// the state PositionCache, and V_FM being already in the state VelocityCache.
// However, it must NOT depend on any quantities involving Ground or other bodies.
// Note: this calculates the HDot matrix *as defined*; we might reverse
// the frames in use. So we call this HDot_F0M0 while the possibly reversed
// version is HDot_FM. Be sure to use X_F0M0 and V_F0M0 in your calculation
// of HDot_F0M0 if you need access to the cross-mobilizer transform and/or velocity.
// CAUTION: our definition of H is transposed from Jain and Schwieters.
virtual void calcAcrossJointVelocityJacobianDot(
    const SBStateDigest_<T>& sbs,
    HType&               HDot_F0M0) const = 0;

// We allow a mobilizer to be defined in reverse when that is more
// convenient. That is, the H matrix can be defined by giving H_F0M0=H_MF and 
// HDot_F0M0=HDot_MF instead of H_FM and HDot_FM. In that case the 
// following methods are called instead of the above; the default 
// implementation postprocesses the output from the above methods, but a 
// mobilizer can override it if it knows how to get the job done faster.
virtual void calcReverseMobilizerH_FM(
    const SBStateDigest_<T>& sbs,
    HType&               H_FM) const;

virtual void calcReverseMobilizerHDot_FM(
    const SBStateDigest_<T>& sbs,
    HType&               HDot_FM) const;

// This routine is NOT joint specific, but cannot be called until the across-joint
// transform X_FM has been calculated and is available in the State cache.
void calcBodyTransforms(
    const SBInstanceVars_<T>&       iv,
    const SBInstanceCache_<T>&      ic,
    const SBTreePositionCache_<T>&  pc,
    TransformP&                  X_PB,
    TransformP&                  X_GB) const
{
    const SBInstancePerMobodInfo& info = ic.getMobodInstanceInfo(this->getNodeNum());
    const bool noX_MB = info.noX_MB;
    const bool noR_PF = info.noR_PF;

    const TransformP& X_MB = getX_MB(ic); // fixed
    const TransformP& X_PF = getX_PF(iv); // fixed
    const TransformP& X_FM = getX_FM(pc); // just calculated
    const TransformP& X_GP = getX_GP(pc); // already calculated

    const TransformP X_FB = (noX_MB ? X_FM                   // cheap
                                   : X_FM*X_MB);            // 63 flops
    X_PB = (noR_PF ? TransformP(X_FB.R(), X_PF.p()+X_FB.p()) // cheap
                   : X_PF*X_FB);                            // 63 flops
    X_GB = X_GP * X_PB;                                     // 63 flops
}

// Same for all mobilizers. The return matrix here is precisely the 
// one used by Jain and Schwieters, but transposed.
void calcParentToChildVelocityJacobianInGround(
    const SBModelVars&          mv,
    const SBInstanceVars_<T>&       iv,
    const SBInstanceCache_<T>&      ic,
    const SBTreePositionCache_<T>&  pc, 
    HType&                      H_PB_G) const;

// Same for all mobilizers. This is the time derivative of 
// the matrix H_PB_G above, with the derivative taken in the 
// Ground frame.
void calcParentToChildVelocityJacobianInGroundDot(
    const SBModelVars&          mv,
    const SBInstanceVars_<T>&       iv,
    const SBInstanceCache_<T>&      ic,
    const SBTreePositionCache_<T>&  pc, 
    const SBTreeVelocityCache_<T>&  vc, 
    HType&                      HDot_PB_G) const;


// Access to body-oriented state and cache entries is the same for all nodes,
// and joint oriented access is almost the same but parametrized by dof. There 
// is a special case for quaternions because they use an extra state variable, 
// and although we don't have to we make special scalar routines available for
// 1-dof joints. Note that all State access routines are inline, not virtual, so
// the cost is just an indirection and an index.
//
// TODO: these inner-loop methods probably shouldn't be indexing a Vector, which
// requires several indirections. Instead, the top-level caller should find the 
// Real* data contained in the Vector and then pass that to the RigidBodyNode 
// routines which will call these ones.

// General joint-dependent select-my-goodies-from-the-pool routines.
const Vec<dof,T>&     fromQ  (const RealP* q)   const {return Vec<dof,T>::getAs(&q[qIndex]);}
Vec<dof,T>&           toQ    (      RealP* q)   const {return Vec<dof,T>::updAs(&q[qIndex]);}
const Vec<dof,T>&     fromU  (const RealP* u)   const {return Vec<dof,T>::getAs(&u[uIndex]);}
Vec<dof,T>&           toU    (      RealP* u)   const {return Vec<dof,T>::updAs(&u[uIndex]);}
const Mat<dof,dof,T>& fromUSq(const RealP* uSq) const {return Mat<dof,dof,T>::getAs(&uSq[uSqIndex]);}
Mat<dof,dof,T>&       toUSq  (      RealP* uSq) const {return Mat<dof,dof,T>::updAs(&uSq[uSqIndex]);}

// Same, but specialized for the common case where dof=1 so everything is scalar.
const RealP& from1Q  (const RealP* q)   const {return q[qIndex];}
RealP&       to1Q    (      RealP* q)   const {return q[qIndex];}
const RealP& from1U  (const RealP* u)   const {return u[uIndex];}
RealP&       to1U    (      RealP* u)   const {return u[uIndex];}
const RealP& from1USq(const RealP* uSq) const {return uSq[uSqIndex];}
RealP&       to1USq  (      RealP* uSq) const {return uSq[uSqIndex];}

// Same, specialized for quaternions. We're assuming that the quaternion is stored
// in the *first* four coordinates, if there are more than four altogether.
const Vec4P& fromQuat(const RealP* q) const {return Vec4P::getAs(&q[qIndex]);}
Vec4P&       toQuat  (      RealP* q) const {return Vec4P::updAs(&q[qIndex]);}

// Extract a Vec3 from a Q-like or U-like object, beginning at an offset from the qIndex or uIndex.
const Vec3P& fromQVec3(const RealP* q, int offs) const {return Vec3P::getAs(&q[qIndex+offs]);}
Vec3P&       toQVec3  (      RealP* q, int offs) const {return Vec3P::updAs(&q[qIndex+offs]);}
const Vec3P& fromUVec3(const RealP* u, int offs) const {return Vec3P::getAs(&u[uIndex+offs]);}
Vec3P&       toUVec3  (      RealP* u, int offs) const {return Vec3P::updAs(&u[uIndex+offs]);}

// These provide an identical interface for when q,u are given as Vectors rather 
// than Reals. Just convert to array of reals and then use an above method.
const Vec<dof,T>&     fromQ  (const VectorP& q)   const {return fromQ(&q[0]);}
Vec<dof,T>&           toQ    (      VectorP& q)   const {return toQ  (&q[0]);}
const Vec<dof,T>&     fromU  (const VectorP& u)   const {return fromU(&u[0]);}
Vec<dof,T>&           toU    (      VectorP& u)   const {return toU  (&u[0]);}
const Mat<dof,dof,T>& fromUSq(const VectorP& uSq) const {return fromUSq(&uSq[0]);}
Mat<dof,dof,T>&       toUSq  (      VectorP& uSq) const {return toUSq  (&uSq[0]);}

const RealP& from1Q  (const VectorP& q)   const {return from1Q  (&q[0]);}
RealP&       to1Q    (      VectorP& q)   const {return to1Q    (&q[0]);}
const RealP& from1U  (const VectorP& u)   const {return from1U  (&u[0]);}
RealP&       to1U    (      VectorP& u)   const {return to1U    (&u[0]);}
const RealP& from1USq(const VectorP& uSq) const {return from1USq(&uSq[0]);}
RealP&       to1USq  (      VectorP& uSq) const {return to1USq  (&uSq[0]);}

// Same, specialized for quaternions. We're assuming that the quaternion comes 
// first in the coordinates.
const Vec4P& fromQuat(const VectorP& q) const {return fromQuat(&q[0]);}
Vec4P&       toQuat  (      VectorP& q) const {return toQuat  (&q[0]);}

// Extract a Vec3 from a Q-like or U-like object, beginning at an offset from
// the qIndex or uIndex.
const Vec3P& fromQVec3(const VectorP& q, int offs) const {return fromQVec3(&q[0], offs);}
Vec3P&       toQVec3  (      VectorP& q, int offs) const {return toQVec3  (&q[0], offs);}
const Vec3P& fromUVec3(const VectorP& u, int offs) const {return fromUVec3(&u[0], offs);}
Vec3P&       toUVec3  (      VectorP& u, int offs) const {return toUVec3  (&u[0], offs);}

// Applications of the above extraction routines to particular interesting items
// in the State. Note that you can't use these for quaternions since they 
// extract "dof" items.

// Cache entries (cache is mutable in a const State)

    // Position


// CAUTION: our H definition is transposed from Jain and Schwieters.
const HType& getH_FM(const SBTreePositionCache_<T>& pc) const
{   return HType::getAs(&pc.storageForH_FM[2*uIndex]); }
HType&       updH_FM(SBTreePositionCache_<T>& pc) const
{   return HType::updAs(&pc.storageForH_FM[2*uIndex]); }

// "H" here should really be H_PB_G, that is, cross joint transition
// matrix relating parent and body frames, but expressed in Ground.
// CAUTION: our H definition is transposed from Jain and Schwieters.
const HType& getH(const SBTreePositionCache_<T>& pc) const
{   return HType::getAs(&pc.storageForH_PB_G[2*uIndex]); }
HType&       updH(SBTreePositionCache_<T>& pc) const
{   return HType::updAs(&pc.storageForH_PB_G[2*uIndex]); }

    // Velocity

// CAUTION: our H definition is transposed from Jain and Schwieters.
const HType& getHDot_FM(const SBTreeVelocityCache_<T>& vc) const
{   return HType::getAs(&vc.storageForHDot_FM[2*uIndex]); }
HType&       updHDot_FM(SBTreeVelocityCache_<T>& vc) const
{   return HType::updAs(&vc.storageForHDot_FM[2*uIndex]); }

// CAUTION: our H definition is transposed from Jain and Schwieters.
const HType& getHDot(const SBTreeVelocityCache_<T>& vc) const
{   return HType::getAs(&vc.storageForHDot[2*uIndex]); }
HType&       updHDot(SBTreeVelocityCache_<T>& vc) const
{   return HType::updAs(&vc.storageForHDot[2*uIndex]); }

    // Dynamics

// These are calculated with articulated body inertias.
const Mat<dof,dof,T>& getD(const SBArticulatedBodyInertiaCache_<T>& abc) const 
{   return fromUSq(abc.storageForD); }
Mat<dof,dof,T>&       updD(SBArticulatedBodyInertiaCache_<T>&       abc) const 
{   return toUSq  (abc.storageForD); }

const Mat<dof,dof,T>& getDI(const SBArticulatedBodyInertiaCache_<T>& abc) const 
{   return fromUSq(abc.storageForDI); }
Mat<dof,dof,T>&       updDI(SBArticulatedBodyInertiaCache_<T>&       abc) const 
{   return toUSq  (abc.storageForDI); }

const Mat<2,dof,Vec3P>& getG(const SBArticulatedBodyInertiaCache_<T>& abc) const
{   return Mat<2,dof,Vec3P>::getAs(&abc.storageForG[2*uIndex]); }
Mat<2,dof,Vec3P>&       updG(SBArticulatedBodyInertiaCache_<T>&       abc) const
{   return Mat<2,dof,Vec3P>::updAs(&abc.storageForG[2*uIndex]); }

const Vec<dof,T>& getTau(const SBInstanceCache_<T>& ic, const RealP* tau) const {
    const PresForcePoolIndex tauIx = 
        ic.getMobodInstanceInfo(nodeNum).firstPresForce;
    assert(tauIx.isValid());
    return Vec<dof,T>::getAs(&tau[tauIx]);
}
Vec<dof,T>& updTau(const SBInstanceCache_<T>& ic, RealP* tau) const {
    const PresForcePoolIndex tauIx = 
        ic.getMobodInstanceInfo(nodeNum).firstPresForce;
    assert(tauIx.isValid());
    return Vec<dof,T>::updAs(&tau[tauIx]);
}

    // Acceleration

const Vec<dof,T>& getEpsilon (const SBTreeAccelerationCache_<T>& ac) const
{   return fromU (ac.epsilon); }
Vec<dof,T>& updEpsilon (SBTreeAccelerationCache_<T>& ac) const
{   return toU   (ac.epsilon); }
const RealP& get1Epsilon(const SBTreeAccelerationCache_<T>& ac) const
{   return from1U(ac.epsilon); }
RealP& upd1Epsilon(SBTreeAccelerationCache_<T>& ac) const
{   return to1U  (ac.epsilon); }



};

// RigidBodyNodeSpec is the default Real precision version. The mobilizers
// that are not (yet) templatized on the scalar type derive from this.
template <int dof, bool noR_FM=false>
using RigidBodyNodeSpec = RigidBodyNodeSpec_<Real, dof, noR_FM>;

// The out-of-line methods are in RigidBodyNodeSpecDefs.h. They are
// instantiated for Real in RigidBodyNodeSpec.cpp.
extern template class RigidBodyNodeSpec_<Real, 1, false>;
extern template class RigidBodyNodeSpec_<Real, 2, false>;
extern template class RigidBodyNodeSpec_<Real, 3, false>;
extern template class RigidBodyNodeSpec_<Real, 4, false>;
extern template class RigidBodyNodeSpec_<Real, 5, false>;
extern template class RigidBodyNodeSpec_<Real, 6, false>;
extern template class RigidBodyNodeSpec_<Real, 1, true>;
extern template class RigidBodyNodeSpec_<Real, 3, true>;

#endif // SimTK_SIMBODY_RIGID_BODY_NODE_SPEC_H_
