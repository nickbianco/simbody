#ifndef SimTK_SIMBODY_RIGID_BODY_NODE_SPEC_BALL_H_
#define SimTK_SIMBODY_RIGID_BODY_NODE_SPEC_BALL_H_

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
 * Contributors: Derived from IVM code written by Charles Schwieters          *
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
 * Define the RigidBodyNode that implements a Ball mobilizer, also known as an
 * Orientation or Spherical joint.
 */

#include "RigidBodyNode.h"
#include "RigidBodyNodeSpec.h"

namespace SimTK {

// ORIENTATION (BALL) //

// Ball joint. This provides three degrees of rotational freedom,  i.e.,
// unrestricted orientation of the body's M frame in the parent's F frame.
// The generalized coordinates are:
//   * 4 quaternions or 3 X-Y-Z (1-2-3) body fixed Euler angles (that is, 
//     fixed in M)
// and generalized speeds are:
//   * angular velocity w_FM as a vector expressed in the F (parent) frame.
// Thus rotational qdots have to be derived from the generalized speeds to
// be turned into either 4 quaternion derivatives or 3 Euler angle derivatives.
//
// NOTE: An XYZ Euler angle sequence has a singularity when the middle angle
// is at 90 or 270 degrees; quaternions are never singular.
// This is templatized on the scalar type T.
template <class T>
class RBNodeBall_ : public RigidBodyNodeSpec_<T, 3, false> {
public:
SimTK_RBNODE_SCALAR_TYPEDEFS(T);
typedef RigidBodyNode_<T> Base;
typedef RigidBodyNodeSpec_<T, 3, false> Spec;

typedef typename Spec::HType HType;
const char* type() const override { return "ball"; }

RBNodeBall_(const MassProperties_<T>& mProps_B,
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
        this->toQ(q)    = R_FM.convertRotationToBodyFixedXYZ();
    else
        this->toQuat(q) = R_FM.convertRotationToQuaternion().asVec4();
}

void setQToFitTranslationImpl(const SBStateDigest& sbs, const Vec3& p_FM, Vector& q) const {
    // M and F frame origins are always coincident for this mobilizer so there is no
    // way to create a translation by rotating. So the only translation we can represent is 0.
}

void setUToFitAngularVelocityImpl(const SBStateDigest& sbs, const Vector&, const Vec3& w_FM,
                                 Vector& u) const
{
        this->toU(u) = w_FM; // relative angular velocity always used as generalized speeds
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
// These are indices into the pool: cos x,y,z sin x,y,z 1/cos(y)
enum {AngleCosQ=0, AngleSinQ=3, AngleOOCosQy=6};
enum {QuatOONorm=0};
int calcQPoolSize(const SBModelVars& mv) const
{   return this->getUseEulerAngles(mv) ? AnglePoolSize : QuatPoolSize; }

// This is expensive in Euler angle mode due to the three sin/cos computations
// required (about 150 flops). In quaternion mode it is much cheaper (about
// 25 flops).
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
        qCache[AngleOOCosQy] = 1/cy; // trouble at 90 or 270 degrees
    } else {
        assert(q && nq==4 && qCache && nQCache==QuatPoolSize && 
               qErr && nQErr==1);

        const RealP quatLen = Vec4P::getAs(q).norm();
        qErr[0] = quatLen - RealP(1);    // normalization error
        qCache[QuatOONorm] = 1/quatLen; // save for later
    }
}

// Because of the precalculations above we can calculate the cross-mobilizer
// transform in 18 flops (Euler angles) or 33 flops (quaternions).
void calcX_FM(const SBStateDigest_<T>& sbs,
              const RealP* q,      int nq,
              const RealP* qCache, int nQCache,
              TransformP&  X_F0M0) const
{
    // RotationP.
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

    // Translation.
    X_F0M0.updP() = 0; // This joint can't translate.
}

// Generalized speeds are the angular velocity expressed in F, so they
// cause rotations around F x,y,z axes respectively.
void calcAcrossJointVelocityJacobian(
    const SBStateDigest_<T>& sbs,
    HType&               H_FM) const
{
    H_FM(0) = SpatialVecP( Vec3P(1,0,0), Vec3P(0) );
    H_FM(1) = SpatialVecP( Vec3P(0,1,0), Vec3P(0) );
    H_FM(2) = SpatialVecP( Vec3P(0,0,1), Vec3P(0) );
}

// The derivative of constant matrix H_FM is not too interesting.
void calcAcrossJointVelocityJacobianDot(
    const SBStateDigest_<T>& sbs,
    HType&               HDot_FM) const
{
    HDot_FM(0) = SpatialVecP( Vec3P(0), Vec3P(0) );
    HDot_FM(1) = SpatialVecP( Vec3P(0), Vec3P(0) );
    HDot_FM(2) = SpatialVecP( Vec3P(0), Vec3P(0) );
}


// Calculate qdot=N(q)*u. Precalculations make this fast in Euler angle
// mode (10 flops); quaternion mode is 27 flops.
// TODO: we're expecting that there are 4 qdots even in Euler angle mode
// so we have to zero out the last one in that case.
void calcQDot(const SBStateDigest_<T>& sbs, const RealP* u, 
                             RealP* qdot) const {
    // We have to trust that the tree position cache is valid here.
    assert(u && qdot);

    const SBModelVars& mv = sbs.getModelVars();

    // Generalized speeds are the same in either mode -- angular velocity
    // of F in M (or B in P), expressed in F (fixed on parent) frame.
    const Vec3P& w_FM = Vec3P::getAs(u);

    if (this->getUseEulerAngles(mv)) {
        // Euler angle mode (10 flops)
        qdot[3] = 0; // TODO: kludge, clear unused element

        const SBModelCache&        mc   = sbs.getModelCache();
        const SBTreePositionCache_<T>& pc   = sbs.getTreePositionCache();
        const RealP*                pool = this->getQPool(mc, pc);

        Vec3P::updAs(qdot) = RotationP::convertAngVelInParentToBodyXYZDot(
             Vec<2,T>::getAs(&pool[AngleCosQ]), Vec<2,T>::getAs(&pool[AngleSinQ]), //x,y
             pool[AngleOOCosQy], w_FM);
    } else {
        // Quaternion_<T> mode (27 flops)
        const Vec4P& quat  = this->fromQuat(sbs.getQ()); // unnormalized
        Vec4P::updAs(qdot) = RotationP::convertAngVelToQuaternionDot(quat, w_FM);
    }
}

// CAUTION: we do not zero the unused 4th element of q for Euler angles; it
// is up to the caller to do that if it is necessary.
// Compute out_q = N * in_u
//   or    out_u = in_q * N
// The u quantity is a vector v_F in the parent frame (e.g. angular 
// velocity of child in parent), while the q quantity is the derivative 
// of a quaternion representing R_FM, i.e. orientation of child in parent.
// Cost: Euler angle mode - 10 flops, Quaternion_<T> mode - 27 flops.
void multiplyByN(const SBStateDigest_<T>& sbs, bool matrixOnRight, 
                 const RealP* in, RealP* out) const
{
    assert(sbs.getStage() >= Stage::Position);
    assert(in && out);

    const SBModelVars& mv = sbs.getModelVars();

    if (this->getUseEulerAngles(mv)) {
        // Euler angle mode (10 flops).

        const SBModelCache&        mc   = sbs.getModelCache();
        const SBTreePositionCache_<T>& pc   = sbs.getTreePositionCache();
        const RealP*                pool = this->getQPool(mc, pc);

        // Aliases.
        const Vec<2,T>& cosxy  = Vec<2,T>::getAs(&pool[AngleCosQ]); // only need x,y
        const Vec<2,T>& sinxy  = Vec<2,T>::getAs(&pool[AngleSinQ]);
        const RealP& oocosy = pool[AngleOOCosQy];
        const Vec3P& in_v   = Vec3P::getAs(in);
        Vec3P&       out_v  = Vec3P::updAs(out);

        if (matrixOnRight) { // out_u = in_q * N = ~N * in_q (9 flops)
            // Transpose on left is same as untransposed on right.
            out_v = RotationP::multiplyByBodyXYZ_NT_P(cosxy, sinxy, oocosy, in_v);
        } else {             // out_q = N*in_u (10 flops)
            out_v = RotationP::multiplyByBodyXYZ_N_P(cosxy, sinxy, oocosy, in_v);
        }
    } else {
        // Quaternion_<T> mode (27 flops).
        const Vec4P& quat  = this->fromQuat(sbs.getQ()); // unnormalized
        const Mat<4,3,T> N_F = // This method returns N for the parent frame
            RotationP::calcUnnormalizedNForQuaternion(quat); // 7 flops
        if (matrixOnRight) 
                Row<3,T>::updAs(out) = Row<4,T>::getAs(in) * N_F; // 20 flops
        else    Vec4P::updAs(out) = N_F * Vec3P::getAs(in);
    }
}

// Compute out_u = inv(N) * in_q
//   or    out_q = in_u * inv(N)
// Cost: Euler angle mode ~ 10 flops, Quaternion_<T> mode ~ 27 flops.
void multiplyByNInv(const SBStateDigest_<T>& sbs, bool matrixOnRight,
                    const RealP* in, RealP* out) const
{
    assert(sbs.getStage() >= Stage::Position);
    assert(in && out);

    const SBModelVars& mv = sbs.getModelVars();

    if (this->getUseEulerAngles(mv)) {
        // Euler angle mode (10 flops).
        const SBModelCache&        mc   = sbs.getModelCache();
        const SBTreePositionCache_<T>& pc   = sbs.getTreePositionCache();
        const RealP*                pool = this->getQPool(mc, pc);

        // Aliases.
        const Vec<2,T>& cosxy  = Vec<2,T>::getAs(&pool[AngleCosQ]); // only need x,y
        const Vec<2,T>& sinxy  = Vec<2,T>::getAs(&pool[AngleSinQ]);
        const Vec3P& in_v   = Vec3P::getAs(in);
        Vec3P&       out_v  = Vec3P::updAs(out);

        if (matrixOnRight) { // out_q = in_u * inv(N) = ~inv(N) * in_u  (10 flops)
            // transpose on left is same as untransposed on right
            out_v = RotationP::multiplyByBodyXYZ_NInvT_P(cosxy, sinxy, in_v);
        } else {             // out_u = inv(N) * in_q                   (9 flops)
            out_v = RotationP::multiplyByBodyXYZ_NInv_P(cosxy, sinxy, in_v);
        }
    } else {
        // Quaternion_<T> mode (27 flops).
        const Vec4P& quat  = this->fromQuat(sbs.getQ()); // unnormalized
        const Mat<3,4,T> NInv_F = // Returns NInv for parent frame.
            RotationP::calcUnnormalizedNInvForQuaternion(quat);  // 7 flops
        if (matrixOnRight) 
                Row<4,T>::updAs(out) = Row<3,T>::getAs(in) * NInv_F; // 20 flops
        else    Vec3P::updAs(out) = NInv_F * Vec4P::getAs(in); // 21 flops
    }
}

// Compute out_q = NDot * in_u
//   or    out_u = in_q * NDot
// Note that it is much cheaper to calculate
// qdotdot directly than to do it using N and NDot; see below.
// Cost: Euler angle mode - 36 flops, Quaternion_<T> mode - 27 flops.
void multiplyByNDot(const SBStateDigest_<T>& sbs, bool matrixOnRight,
                    const RealP* in, RealP* out) const
{
    assert(sbs.getStage() > Stage::Velocity);
    assert(in && out);

    const SBModelVars& mv = sbs.getModelVars();

    if (this->getUseEulerAngles(mv)) {
        // Euler angle mode (36 flops).
        const SBModelCache&        mc   = sbs.getModelCache();
        const SBTreePositionCache_<T>& pc   = sbs.getTreePositionCache();
        const RealP*                pool = this->getQPool(mc, pc);

        // Aliases.
        const Vec<2,T>& cosxy  = Vec<2,T>::getAs(&pool[AngleCosQ]); // only need x,y
        const Vec<2,T>& sinxy  = Vec<2,T>::getAs(&pool[AngleSinQ]);
        const RealP& oocosy = pool[AngleOOCosQy];
        const Vec3P& qdot   = this->fromQ(sbs.getQDot());

        // We don't have a nice multiply-by routine here so just get the NDot
        // matrix and use it (21 flops).
        const Mat33P NDot_F = 
            RotationP::calcNDotForBodyXYZInParentFrame(cosxy, sinxy, oocosy, qdot);

        if (matrixOnRight) {    // out_u = in_q * NDot (15 flops)
            Row<3,T>::updAs(out) = Row<3,T>::getAs(in) * NDot_F;
        } else {                // out_q = NDot * in_u (15 flops)
            Vec3P::updAs(out) = NDot_F * Vec3P::getAs(in);
        }
    } else {
        // Quaternion_<T> mode (27 flops).
        const Vec4P& qdot  = this->fromQuat(sbs.getQDot()); // unnormalized
        const Mat<4,3,T> NDot_F = // This method returns NDot for the parent frame
            RotationP::calcUnnormalizedNDotForQuaternion(qdot); // 7 flops
        if (matrixOnRight) 
                Row<3,T>::updAs(out) = Row<4,T>::getAs(in) * NDot_F;  // 21 flops
        else    Vec4P::updAs(out) = NDot_F * Vec3P::getAs(in);  // 20 flops
    }
}

// Calculate qdotdot from udot as qdotdot = N(q)*udot + NDot(q,qdot)*u, but
// faster. Here we assume that qdot=N*u has already been calculated, which
// is very helpful in Euler angle mode. In either mode it is a very bad
// idea to calculate this using explicit N and NDot matrices; we can save
// a lot of time by using more specialized methods.
// Cost: Euler angle mode - 22 flops, Quaternion_<T> mode - 41 flops.
void calcQDotDot(const SBStateDigest_<T>& sbs, const RealP* udot, 
                                   RealP* qdotdot) const {
    assert(sbs.getStage() > Stage::Velocity);
    assert(udot && qdotdot);

    const SBModelVars& mv = sbs.getModelVars();

    // Angular acceleration is the same in either mode.
    const Vec3P& b_FM = Vec3P::getAs(udot);   // a.k.a. w_FM_dot

    if (this->getUseEulerAngles(mv)) {
        // Euler angle mode (22 flops).
        Vec4P::updAs(qdotdot) = 0; // TODO: kludge, clear unused element

        const SBModelCache&        mc   = sbs.getModelCache();
        const SBTreePositionCache_<T>& pc   = sbs.getTreePositionCache();
        const RealP*                pool = this->getQPool(mc, pc);

        // Aliases.
        const Vec<2,T>& cosxy     = Vec<2,T>::getAs(&pool[AngleCosQ]); // only need x,y
        const Vec<2,T>& sinxy     = Vec<2,T>::getAs(&pool[AngleSinQ]);
        const RealP& oocosy    = pool[AngleOOCosQy];
        const Vec3P& qdot      = this->fromQ(sbs.getQDot());
        Vec3P&       qdotdot_v = Vec3P::updAs(qdotdot);

        // 22 flops.
        qdotdot_v = RotationP::convertAngAccInParentToBodyXYZDotDot
                                            (cosxy, sinxy, oocosy, qdot, b_FM);
    } else {
        // Quaternion_<T> mode (41 flops).
        const Vec4P& quat  = this->fromQuat(sbs.getQ()); // unnormalized
        const Vec3P& w_FM = this->fromU(sbs.getU());
        Vec4P::updAs(qdotdot) = 
            RotationP::convertAngVelDotToQuaternionDotDot(quat,w_FM,b_FM);
    }
}

int getMaxNQ()              const {return 4;}
int getNQInUse(const SBModelVars& mv) const {
    return this->getUseEulerAngles(mv) ? 3 : 4;
} 
bool isUsingQuaternion(const SBStateDigest_<T>& sbs, MobilizerQIndex& startOfQuaternion) const {
    if (this->getUseEulerAngles(sbs.getModelVars())) {
        startOfQuaternion.invalidate(); 
        return false;
    }
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
    this->toQuat(outputQ) = Vec4(0); // clear unused element
    this->toQ(outputQ) = Rotation(Quaternion(this->fromQuat(inputQ))).convertRotationToBodyFixedXYZ();
}

void convertToQuaternions(const Vector& inputQ, Vector& outputQ) const {
    Rotation rot;
    rot.setRotationToBodyFixedXYZ(this->fromQ(inputQ));
    this->toQuat(outputQ) = rot.convertRotationToQuaternion().asVec4();
}


};

// The default Real precision version.
typedef RBNodeBall_<Real> RBNodeBall;

} // namespace SimTK


#endif // SimTK_SIMBODY_RIGID_BODY_NODE_SPEC_BALL_H_

