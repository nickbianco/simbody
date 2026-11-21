#ifndef SimTK_SIMBODY_RIGID_BODY_NODE_SPEC_LINEORIENTATION_H_
#define SimTK_SIMBODY_RIGID_BODY_NODE_SPEC_LINEORIENTATION_H_

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
 * Contributors: Paul Mitiguy                                                 *
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
 * Define the RigidBodyNode that implements a LineOrientation mobilizer.
 */

#include "RigidBodyNode.h"
#include "RigidBodyNodeSpec.h"

namespace SimTK {


    // LINE ORIENTATION //

// LineOrientation joint. Like a Ball joint, this provides full rotational
// freedom, but for a degenerate body which is thin (inertialess) along its
// own z axis. These arise in molecular modeling for linear molecules formed
// by pairs of atoms, or by multiple atoms in a linear arrangement like
// carbon dioxide (CO2) whose structure is O=C=O in a straight line. We are
// assuming that there is no meaning to a rotation about the linear axis,
// so free orientation requires just *two* degrees of freedom, not *three*
// as is required for general rigid bodies. And in fact we can get away with
// just two generalized speeds. But so far, no one has been able to come up
// with a way to manage with only two generalized coordinates, so this joint
// has the same q's as a regular Orientation (Ball) joint: either a quaternion
// for unconditional stability, or a three-angle (body fixed 1-2-3)
// Euler sequence which will be dynamically singular when the middle (y) axis
// is 90 degrees. Use the Euler sequence only for small motions or for 
// kinematics problems (and note that only the first two are meaningful).
//
// To summarize, the generalized coordinates are:
//   * 4 quaternions or 3 1-2-3 body fixed Euler angles (that is, fixed in M)
// and generalized speeds are:
//   * the x,y components of the angular velocity w_FM_M, that is, the angular
//     velocity of M in F expressed in M (where we want wz=0).
//     NOTE: THAT IS A DIFFERENT FRAME THAN IS USED FOR BALL AND GIMBAL
// Thus the qdots have to be derived from the generalized speeds to
// be turned into either 4 quaternion derivatives or 3 Euler angle derivatives.
// This is templatized on the scalar type T.
template <class T>
class RBNodeLineOrientation_ : public RigidBodyNodeSpec_<T, 2, false> {
public:
SimTK_RBNODE_SCALAR_TYPEDEFS(T);
typedef RigidBodyNode_<T> Base;
typedef RigidBodyNodeSpec_<T, 2, false> Spec;

typedef typename Spec::HType HType;
const char* type() const override { return "lineOrientation"; }

RBNodeLineOrientation_(const MassProperties_<T>& mProps_B,
                      bool                  isReversed,
                      UIndex&               nextUSlot,
                      USquaredIndex&        nextUSqSlot,
                      QIndex&               nextQSlot)
  :   Spec(mProps_B,nextUSlot,nextUSqSlot,nextQSlot,
                         Base::QDotMayDifferFromU, Base::QuaternionMayBeUsed, isReversed)
{
    this->updateSlots(nextUSlot,nextUSqSlot,nextQSlot);
}

void setQToFitRotationImpl(const SBStateDigest& sbs, const Rotation& R_FM,
                          Vector& q) const 
{
    if (this->getUseEulerAngles(sbs.getModelVars()))
        this->toQVec3(q,0)    = R_FM.convertRotationToBodyFixedXYZ();
    else
        this->toQuat(q) = R_FM.convertRotationToQuaternion().asVec4();
}

void setQToFitTranslationImpl(const SBStateDigest& sbs, const Vec3& p_FM, 
                              Vector& q) const {
    // M and F frame origins are always coincident for this mobilizer so there is no
    // way to create a translation by rotating. So the only translation we can represent is 0.
}

void setUToFitAngularVelocityImpl(const SBStateDigest& sbs, 
                                  const Vector& q, const Vec3& w_FM,
                                  Vector& u) const
{
    Rotation R_FM;
    if (this->getUseEulerAngles(sbs.getModelVars()))
        R_FM.setRotationToBodyFixedXYZ( this->fromQVec3(q,0) );
    else {
        // TODO: should use qnorm pool
        R_FM.setRotationFromQuaternion( Quaternion(this->fromQuat(q)) ); // normalize
    }
    const Vec3 w_FM_M = ~R_FM*w_FM;
    this->toU(u) = Vec2(w_FM_M[0], w_FM_M[1]); // (x,y) of relative angular velocity always used as generalized speeds
}

void setUToFitLinearVelocityImpl
   (const SBStateDigest& sbs, const Vector&, const Vec3& v_FM, Vector& u) const
{
    // M and F frame origins are always coincident for this mobilizer so there is no
    // way to create a linear velocity by rotating. So the only linear velocity
    // we can represent is 0.
}

// When we're using Euler angles, we're going to want to cache cos and sin for
// each angle, and also 1/cos of the middle angle will be handy to have around.
// When we're using quaternions, we'll cache 1/|q| for convenient normalizing
// of quaternions.
enum {AnglePoolSize=7, QuatPoolSize=1};
// cos x,y,z sin x,y,z 1/cos(y)
enum {AngleCosQ=0, AngleSinQ=3, AngleOOCosQy=6};
enum {QuatOONorm=0};
int calcQPoolSize(const SBModelVars& mv) const
{   return this->getUseEulerAngles(mv) ? AnglePoolSize : QuatPoolSize; }

void performQPrecalculations(const SBStateDigest_<T>& sbs,
                             const RealP* q,      int nq,
                             RealP*       qCache, int nQCache,
                             RealP*       qErr,   int nQErr) const
{
    if (this->getUseEulerAngles(sbs.getModelVars())) {
        assert(q && nq==3 && qCache && nQCache==AnglePoolSize && nQErr==0);
        const RealP cy = cos(q[1]);
        Vec3P::updAs(&qCache[AngleCosQ]) =
            Vec3P(cos(q[0]), cy, cos(q[2]));
        Vec3P::updAs(&qCache[AngleSinQ]) =
            Vec3P(sin(q[0]), sin(q[1]), sin(q[2]));
        qCache[AngleOOCosQy] = 1/cy; // trouble at 90 degrees
    } else {
        assert(q && nq==4 && qCache && nQCache==QuatPoolSize && 
               qErr && nQErr==1);
        const RealP quatLen = Vec4P::getAs(q).norm();
        qErr[0] = quatLen - RealP(1);    // normalization error
        qCache[QuatOONorm] = 1/quatLen; // save for later
    }
}

void calcX_FM(const SBStateDigest_<T>& sbs,
              const RealP* q,      int nq,
              const RealP* qCache, int nQCache,
              TransformP&  X_F0M0) const
{
    X_F0M0.updP() = 0; // This joint can't translate.
    if (this->getUseEulerAngles(sbs.getModelVars())) {
        assert(q && nq==3 && qCache && nQCache==AnglePoolSize);
        X_F0M0.updR().setRotationToBodyFixedXYZ // 18 flops
           (Vec3P::getAs(&qCache[AngleCosQ]), Vec3P::getAs(&qCache[AngleSinQ]));
    } else {
        assert(q && nq==4 && qCache && nQCache==QuatPoolSize);
        // Must use a normalized quaternion to generate the rotation matrix.
        // Here we normalize with just 4 flops using precalculated 1/norm(q).
        const Quaternion_<T> quat(Vec4P::getAs(q)*qCache[QuatOONorm], true); 
        X_F0M0.updR().setRotationFromQuaternion(quat); // 29 flops
    }
}

// The generalized speeds for this 2-dof rotational joint are the x and y
// components of the angular velocity of M in the F frame, expressed in the *M*
// frame.
void calcAcrossJointVelocityJacobian(
    const SBStateDigest_<T>& sbs,
    HType&               H_FM) const
{
    const SBTreePositionCache_<T>& pc = sbs.updTreePositionCache(); // "upd" because we're realizing positions now
    const TransformP X_F0M0 = this->findX_F0M0(pc);

    // Dropping the 0's here.
    const RotationP& R_FM = X_F0M0.R();
    const Vec3P&     Mx_F = R_FM.x(); // M's x axis, expressed in F
    const Vec3P&     My_F = R_FM.y(); // M's y axis, expressed in F

    H_FM(0) = SpatialVecP( Mx_F, Vec3P(0) );
    H_FM(1) = SpatialVecP( My_F, Vec3P(0) );
}

// Since the Jacobian above is not constant in F,
// its time derivative is non zero. Here we use the fact that for
// a vector r_B_A fixed in a moving frame B but expressed in another frame A,
// its time derivative in A is the angular velocity of B in A crossed with
// the vector, i.e., d_A/dt r_B_A = w_AB % r_B_A.
void calcAcrossJointVelocityJacobianDot(
    const SBStateDigest_<T>& sbs,
    HType&               HDot_FM) const
{
    const SBTreePositionCache_<T>& pc = sbs.getTreePositionCache();
    const SBTreeVelocityCache_<T>& vc = sbs.updTreeVelocityCache(); // "upd" because we're realizing velocities now
    const TransformP  X_F0M0 = this->findX_F0M0(pc);

    // Dropping the 0's here.
    const RotationP& R_FM = X_F0M0.R();
    const Vec3P&     Mx_F = R_FM.x(); // M's x axis, expressed in F
    const Vec3P&     My_F = R_FM.y(); // M's y axis, expressed in F

    const Vec3P      w_FM = this->find_w_F0M0(pc,vc); // angular velocity of M in F

    HDot_FM(0) = SpatialVecP( w_FM % Mx_F, Vec3P(0) );
    HDot_FM(1) = SpatialVecP( w_FM % My_F, Vec3P(0) );
}

// CAUTION: we do not zero the unused 4th element of q for Euler angles; it
// is up to the caller to do that if it is necessary.
// Compute out_q = N * in_u
//   or    out_u = in_q * N
// The u quantity is a vector v_M in the child body frame (e.g. angular 
// velocity of child in parent, but expressed in child), while the q 
// quantity is the derivative of a quaternion or Euler sequence representing 
// R_FM, i.e. orientation of child in parent.
void multiplyByN(const SBStateDigest_<T>& sbs, bool matrixOnRight,
                 const RealP* in, RealP* out) const
{
    assert(sbs.getStage() >= Stage::Position);
    assert(in && out);

    const SBModelVars&          mv   = sbs.getModelVars();
    const SBTreePositionCache_<T>&  pc   = sbs.getTreePositionCache();
    const auto&                 allQ = sbs.getQ();

    if (this->getUseEulerAngles(mv)) {
        const Vec3P& q = this->fromQVec3(allQ,0);
        const Mat<3,2,T> N_FM = // N here mixes parent and body frames (convenient)
            RotationP::calcNForBodyXYZInBodyFrame(q)
                        .template getSubMat<3,2>(0,0); // drop 3rd column
        if (matrixOnRight) Row<2,T>::updAs(out) = Row<3,T>::getAs(in) * N_FM;
        else               Vec3P::updAs(out) = N_FM * Vec<2,T>::getAs(in);
    } else {
        // Quaternion: N block is only available expecting angular velocity 
        // in the parent frame F, but we have it in M for this joint so we
        // have to calculate N_FM = N_FF*R_FM.
        const RotationP  R_FM = this->findX_F0M0(pc).R();
        const Vec4P& q = this->fromQuat(allQ);
        const Mat<4,2,T> N_FM = (RotationP::calcUnnormalizedNForQuaternion(q)*R_FM)
                            .template getSubMat<4,2>(0,0); // drop 3rd column
        if (matrixOnRight) Row<2,T>::updAs(out) = Row<4,T>::getAs(in) * N_FM;
        else               Vec4P::updAs(out) = N_FM * Vec<2,T>::getAs(in);
    }
}

// Compute out_u = inv(N) * in_q
//   or    out_q = in_u * inv(N)
void multiplyByNInv(const SBStateDigest_<T>& sbs, bool matrixOnRight, 
                    const RealP* in, RealP* out) const
{
    assert(sbs.getStage() >= Stage::Position);
    assert(in && out);

    const SBModelVars&          mv   = sbs.getModelVars();
    const SBTreePositionCache_<T>&  pc   = sbs.getTreePositionCache();
    const auto&                 allQ = sbs.getQ();

    if (this->getUseEulerAngles(mv)) {
        const Vec3P& q = this->fromQVec3(allQ,0);
        const Mat<2,3,T> NInv_MF = RotationP::calcNInvForBodyXYZInBodyFrame(q)
                                .template getSubMat<2,3>(0,0);   // drop 3rd row
        if (matrixOnRight) Row<3,T>::updAs(out) = Row<2,T>::getAs(in) * NInv_MF;
        else               Vec<2,T>::updAs(out) = NInv_MF * Vec3P::getAs(in);
    } else {
        // Quaternion: NInv block is only available expecting angular 
        // velocity in the parent frame F, but we have it in M for 
        // this joint so we have to calculate NInv_MF = R_MF*NInv_FF.
        const RotationP  R_FM = this->findX_F0M0(pc).R();
        const Vec4P& q = this->fromQuat(allQ);
        const Mat<2,4,T> NInv_MF = (~R_FM*RotationP::calcUnnormalizedNInvForQuaternion(q))
                                .template getSubMat<2,4>(0,0);   // drop 3rd row
        if (matrixOnRight) Row<4,T>::updAs(out) = Row<2,T>::getAs(in) * NInv_MF;
        else               Vec<2,T>::updAs(out) = NInv_MF * Vec4P::getAs(in);
    }
}

// Compute out_q = NDot * in_u
//   or    out_u = in_q * NDot
void multiplyByNDot(const SBStateDigest_<T>& sbs, bool matrixOnRight,
                 const RealP* in, RealP* out) const
{
    assert(sbs.getStage() >= Stage::Velocity);
    assert(in && out);

    const SBModelVars&          mv      = sbs.getModelVars();
    const SBTreePositionCache_<T>&  pc      = sbs.getTreePositionCache();
    const auto&                 allQ    = sbs.getQ();
    const auto&                 allQDot = sbs.getQDot();

    if (this->getUseEulerAngles(mv)) {
        const Vec3P& q = this->fromQVec3(allQ,0);
        const Vec3P& qdot = this->fromQVec3(allQDot,0);
        const Mat<3,2,T> NDot_FM = // NDot here mixes parent and body frames (convenient)
            RotationP::calcNDotForBodyXYZInBodyFrame(q, qdot)
                        .template getSubMat<3,2>(0,0); // drop 3rd column
        if (matrixOnRight) Row<2,T>::updAs(out) = Row<3,T>::getAs(in) * NDot_FM;
        else               Vec3P::updAs(out) = NDot_FM * Vec<2,T>::getAs(in);
    } else {
        // Quaternion: NDot block is only available expecting angular velocity 
        // in the parent frame F, but we have it in M for this joint so we
        // have to differentiate N_FM = N_FF*R_FM, giving
        // NDot_FM = NDot_FF*R_FM + N_FF*RDot_FM with RDot_FM = w_FM x R_FM
        // and w_FM expressed in F.
        const RotationP  R_FM = this->findX_F0M0(pc).R();
        const Vec4P& q    = this->fromQuat(allQ);
        const Vec4P& qdot = this->fromQuat(allQDot);
        const Vec3P  w_FM_M = Vec3P(this->fromU(sbs.getU())[0],
                                  this->fromU(sbs.getU())[1], 0);
        const Mat33P RDot_FM = crossMat(R_FM*w_FM_M) * R_FM.asMat33();
        const Mat<4,2,T> NDot_FM = 
           (RotationP::calcUnnormalizedNDotForQuaternion(qdot)*R_FM
            + RotationP::calcUnnormalizedNForQuaternion(q)*RDot_FM)
                        .template getSubMat<4,2>(0,0); // drop 3rd column
        if (matrixOnRight) Row<2,T>::updAs(out) = Row<4,T>::getAs(in) * NDot_FM;
        else               Vec4P::updAs(out) = NDot_FM * Vec<2,T>::getAs(in);
    }
}

void calcQDot(
    const SBStateDigest_<T>&   sbs,
    const RealP*            u, 
    RealP*                  qdot) const 
{
    const SBModelVars& mv = sbs.getModelVars();
    const SBTreePositionCache_<T>& pc = sbs.getTreePositionCache();
    const Vec3P w_FM_M = Vec<2,T>::getAs(u).append1(RealP(0)); // angular velocity of M in F, exp in M (with wz=0) 
    if (this->getUseEulerAngles(mv)) {
        Vec3P::updAs(qdot) = RotationP::convertAngVelInBodyFrameToBodyXYZDot(this->fromQVec3(sbs.getQ(),0),
                                    w_FM_M); // need w in *body*, not parent
        qdot[3] = 0;
    } else {
        const RotationP  R_FM = this->findX_F0M0(pc).R();
        Vec4P::updAs(qdot) = RotationP::convertAngVelToQuaternionDot(this->fromQuat(sbs.getQ()),
                                    R_FM*w_FM_M); // need w in *parent* frame
    }
}

void calcQDotDot(
    const SBStateDigest_<T>&   sbs,
    const RealP*            udot, 
    RealP*                  qdotdot) const 
{
    const SBModelVars&          mv = sbs.getModelVars();
    const SBTreePositionCache_<T>&  pc = sbs.getTreePositionCache();
    const Vec3P w_FM_M     = this->fromU(sbs.getU()).append1(RealP(0)); // angular velocity of M in F, exp in M (with wz=0)
    const Vec3P w_FM_M_dot = Vec<2,T>::getAs(udot).append1(RealP(0));

    if (this->getUseEulerAngles(mv)) {
        Vec3P::updAs(qdotdot) = RotationP::convertAngVelDotInBodyFrameToBodyXYZDotDot
                                   (this->fromQVec3(sbs.getQ(),0), w_FM_M, w_FM_M_dot); // body frame
        qdotdot[3] = 0;
    } else {
        const RotationP  R_FM = this->findX_F0M0(pc).R();
        Vec4P::updAs(qdotdot) = RotationP::convertAngVelDotToQuaternionDotDot
                              (this->fromQuat(sbs.getQ()),R_FM*w_FM_M,R_FM*w_FM_M_dot); // parent frame
    }
}

int getMaxNQ()              const {return 4;}
int getNQInUse(const SBModelVars& mv) const {
    return this->getUseEulerAngles(mv) ? 3 : 4;
} 
bool isUsingQuaternion(const SBStateDigest_<T>& sbs, MobilizerQIndex& startOfQuaternion) const {
    if (this->getUseEulerAngles(sbs.getModelVars())) {startOfQuaternion.invalidate(); return false;}
    startOfQuaternion = MobilizerQIndex(0); // quaternion comes first
    return true;
}

void setMobilizerDefaultPositionValues(
    const SBModelVars& mv,
    Vector&            q) const 
{
    if (this->getUseEulerAngles(mv)) {
        //TODO: kludge
        this->toQuat(q) = Vec4(0); // clear unused element
        this->toQ(q) = 0;
    }
    else this->toQuat(q) = Vec4(1,0,0,0);
}

bool enforceQuaternionConstraints(
    const SBStateDigest& sbs,
    Vector&             q,
    Vector&             qErrest) const 
{
    if (this->getUseEulerAngles(sbs.getModelVars())) 
        return false;   // no change

    Vec4& quat = this->toQuat(q);
    quat = quat / quat.norm();

    if (qErrest.size()) {
        Vec4& qerr = this->toQuat(qErrest);
        qerr -= dot(qerr,quat) * quat;
    }

    return true;
}

void convertToEulerAngles(const Vector& inputQ, Vector& outputQ) const {
    this->toQVec3(outputQ, 4) = Vec3(0); // clear unused element
    this->toQVec3(outputQ, 2) = this->fromQVec3(inputQ, 3);
    this->toQVec3(outputQ, 0) = Rotation(Quaternion(this->fromQuat(inputQ))).convertRotationToBodyFixedXYZ();
}

void convertToQuaternions(const Vector& inputQ, Vector& outputQ) const {
    this->toQVec3(outputQ, 4) = this->fromQVec3(inputQ, 3);
    Rotation rot;
    rot.setRotationToBodyFixedXYZ(this->fromQVec3(inputQ, 0));
    this->toQuat(outputQ) = rot.convertRotationToQuaternion().asVec4();
}


};

// The default Real precision version.
typedef RBNodeLineOrientation_<Real> RBNodeLineOrientation;

} // namespace SimTK



#endif // SimTK_SIMBODY_RIGID_BODY_NODE_SPEC_LINEORIENTATION_H_

