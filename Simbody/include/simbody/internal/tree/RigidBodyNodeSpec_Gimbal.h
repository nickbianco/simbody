#ifndef SimTK_SIMBODY_RIGID_BODY_NODE_SPEC_GIMBAL_H_
#define SimTK_SIMBODY_RIGID_BODY_NODE_SPEC_GIMBAL_H_

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
 * Define the RigidBodyNode that implements a Gimbal mobilizer.
 */

#include "RigidBodyNode.h"
#include "RigidBodyNodeSpec.h"

namespace SimTK {


    // GIMBAL //

// Gimbal joint. This provides three degrees of rotational freedom,  i.e.,
// unrestricted orientation of the body's M frame in the parent's F frame.
// The generalized coordinates q are:
//   * 3 X-Y-Z (1-2-3) body fixed Euler angles (that is, fixed in M)
// and generalized speeds u are:
//   * u = qdot, that is, the 1-2-3 body fixed Euler angle derivatives
//
// NOTE: This joint has a singularity when the middle angle is near 90 degrees.
// In most cases you should use a Ball joint instead, which by default uses
// a quaternion as its generalized coordinates to avoid this singularity. A
// modeling option allows the Ball to be switched to use Euler angles when 
// convenient.

// This is templatized on the scalar type T.
template <class T>
class RBNodeGimbal_ : public RigidBodyNodeSpec_<T, 3, false> {
public:
SimTK_RBNODE_SCALAR_TYPEDEFS(T);
typedef RigidBodyNode_<T> Base;
typedef RigidBodyNodeSpec_<T, 3, false> Spec;

typedef typename Spec::HType HType;
const char* type() const override { return "gimbal"; }

RBNodeGimbal_(const MassProperties_<T>& mProps_B,
              bool                  isReversed,
              UIndex&               nextUSlot,
              USquaredIndex&        nextUSqSlot,
              QIndex&               nextQSlot)
:   Spec(mProps_B,nextUSlot,nextUSqSlot,nextQSlot,
        Base::QDotIsAlwaysTheSameAsU, 
        Base::QuaternionIsNeverUsed, 
        isReversed)
{
    this->updateSlots(nextUSlot,nextUSqSlot,nextQSlot);
}

void setQToFitRotationImpl(const SBStateDigest& sbs, const Rotation& R_FM,
                           Vector& q) const {
    this->toQ(q) = R_FM.convertRotationToBodyFixedXYZ();
}

void setQToFitTranslationImpl(const SBStateDigest& sbs, const Vec3& p_FM, 
                              Vector& q) const {
    // M and F frame origins are always coincident for this mobilizer so 
    // there is no way to create a translation by rotating. So the only 
    // translation we can represent is 0.
}

// Given the angular velocity of M in F, expressed in F, compute the Euler
// angle derivatives qdot that would produce that angular velocity, and 
// return u=qdot.
void setUToFitAngularVelocityImpl
   (const SBStateDigest& sbs, const Vector& q, const Vec3& w_FM,
    Vector& u) const 
{
    Real q0 = this->fromQ(q)[0];
    Real q1 = this->fromQ(q)[1];
    const Vec2 cosxy(std::cos(q0), std::cos(q1));
    const Vec2 sinxy(std::sin(q0), std::sin(q1));
    const Real oocosy = 1 / cosxy[1];
    const Vec3 qdot = 
        Rotation::convertAngVelInParentToBodyXYZDot(cosxy,sinxy,oocosy,w_FM);
    this->toU(u) = qdot;
}

void setUToFitLinearVelocityImpl
   (const SBStateDigest& sbs, const Vector&, const Vec3& v_FM, 
    Vector& u) const
{
    // M and F frame origins are always coincident for this mobilizer so 
    // there is no way to create a linear velocity by rotating. So the 
    // only linear velocity we can represent is 0.
}

// We want to cache cos and sin for each angle, and also 1/cos of the middle 
// angle will be handy to have around.
enum {PoolSize=7};
// cos x,y,z sin x,y,z 1/cos(y)
enum {CosQ=0, SinQ=3, OOCosQy=6};
int calcQPoolSize(const SBModelVars& mv) const
{   return PoolSize; }

// This is expensive since we have three sin/cos computations and a divide
// to do, approx. 150 flops. But we hope to re-use these calculations several
// times before we're done with a complete realization.
void performQPrecalculations(const SBStateDigest_<T>& sbs,
                             const RealP* q,      int nq,
                             RealP*       qCache, int nQCache,
                             RealP*       qErr,   int nQErr) const
{
    assert(q && nq==3 && qCache && nQCache==PoolSize && nQErr==0);

    const RealP cy = cos(q[1]);
    Vec3P::updAs(&qCache[CosQ]) =
        Vec3P(cos(q[0]), cy, cos(q[2]));
    Vec3P::updAs(&qCache[SinQ]) =
        Vec3P(sin(q[0]), sin(q[1]), sin(q[2]));
    qCache[OOCosQy] = 1/cy; // trouble at 90 or 270 (-90) degrees
}

// Because of the precalculations we can calculate the cross-mobilizer
// transform in only 18 flops.
void calcX_FM(const SBStateDigest_<T>& sbs,
              const RealP* q,      int nq,
              const RealP* qCache, int nQCache,
              TransformP&  X_F0M0) const
{
    assert(q && nq==3 && qCache && nQCache==PoolSize);

    X_F0M0.updR().setRotationToBodyFixedXYZ // 18 flops
        (Vec3P::getAs(&qCache[CosQ]), Vec3P::getAs(&qCache[SinQ]));
    X_F0M0.updP() = 0; // This joint can't translate.
}

// Generalized speeds are the Euler angle derivatives. The H_FM matrix maps
// those into the angular velocity of M in F, expressed in F, so
//  [w_FM]   [Hw_FM]
//  [    ] = [     ] * qdot      (where v_FM is always zero)
//  [v_FM]   [  0  ]
// Using the precalculations this requires only 3 flops, although there is
// a fair bit of poking around in memory required.
void calcAcrossJointVelocityJacobian(
    const SBStateDigest_<T>& sbs,
    HType&               H_FM) const
{
    const SBModelCache&        mc = sbs.getModelCache();
    // Use "upd" here because we're realizing positions now.
    const SBTreePositionCache_<T>& pc = sbs.updTreePositionCache();
    const RealP*                pool = this->getQPool(mc, pc);

    const RealP c0 = pool[CosQ], c1 = pool[CosQ+1];
    const RealP s0 = pool[SinQ], s1 = pool[SinQ+1];

    // Fill in columns of H_FM. See RotationP::calcNInvForBodyXYZInParentFrame().
    H_FM(0) = SpatialVecP( Vec3P(1,     0,    0),    Vec3P(0) );
    H_FM(1) = SpatialVecP( Vec3P(0,    c0,    s0),   Vec3P(0) );
    H_FM(2) = SpatialVecP( Vec3P(s1, -s0*c1, c0*c1), Vec3P(0) );
}

// Differentiate H_FM to get HDot_FM. Note that this depends on qdot:
//    d/dt cos(q0) = -sin(q0)*qdot0, etc.
void calcAcrossJointVelocityJacobianDot(
    const SBStateDigest_<T>& sbs,
    HType&               HDot_FM) const
{
    const SBModelCache&        mc = sbs.getModelCache();
    const SBTreePositionCache_<T>& pc = sbs.getTreePositionCache();

    const RealP* pool = this->getQPool(mc, pc);
    const RealP c0 = pool[CosQ], c1 = pool[CosQ+1];
    const RealP s0 = pool[SinQ], s1 = pool[SinQ+1];

    // Use "upd" here because we're realizing velocities now.
    const Vec3P& qdot = this->fromQ(sbs.updQDot());
    const RealP qd0 = qdot[0], qd1 = qdot[1];

    const RealP dc0 = -s0*qd0, dc1 = -s1*qd1; // derivatives of c0,c1,s0,s1
    const RealP ds0 =  c0*qd0, ds1 =  c1*qd1;

    // Compare with H_FM above.
    HDot_FM(0) = SpatialVecP(Vec3P(0,         0,              0),       Vec3P(0));
    HDot_FM(1) = SpatialVecP(Vec3P(0,        dc0,            ds0),      Vec3P(0));
    HDot_FM(2) = SpatialVecP(Vec3P(ds1, -ds0*c1-s0*dc1, dc0*c1+c0*dc1), Vec3P(0));
}

// Can use default for calcQDot, multiplyByN, etc., since qdot==u for Gimbal
// mobilizer.

};

// The default Real precision version.
typedef RBNodeGimbal_<Real> RBNodeGimbal;

} // namespace SimTK



#endif // SimTK_SIMBODY_RIGID_BODY_NODE_SPEC_GIMBAL_H_

