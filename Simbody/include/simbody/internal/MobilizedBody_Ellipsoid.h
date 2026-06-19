#ifndef SimTK_SIMBODY_MOBILIZED_BODY_ELLIPSOID_H_
#define SimTK_SIMBODY_MOBILIZED_BODY_ELLIPSOID_H_

/* -------------------------------------------------------------------------- *
 *                               Simbody(tm)                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the SimTK biosimulation toolkit originating from           *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org/home/simbody.  *
 *                                                                            *
 * Portions copyright (c) 2009-13 Stanford University and the Authors.        *
 * Authors: Ajay Seth, Michael Sherman                                        *
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

/** @file
Declares the MobilizedBody::Ellipsoid class. **/

#include "simbody/internal/MobilizedBody.h"

namespace SimTK {

/** Three mobilities -- coordinated rotation and translation along the
surface of an ellipsoid fixed to the parent (inboard) body.

The generalized coordinates q are the same as for a Ball (Orientation)
mobilizer, that is, a quaternion or an x-y-z body-fixed Euler sequence depending 
on the "use Euler angles" modeling option. The three generalized speeds u for 
this mobilizer are also the same as for a Ball mobilizer, that is
the three measure numbers of the angular velocity vector w_FM, the
relative angular velocity of the outboard M frame in the inboard F frame,
expressed in the F frame. That is unchanged by setting the "use Euler
angles" modeling option. Note that qdot != u for this mobilizer.

The semi-axis radii are an Instance-stage State variable; see setRadii() and
getRadii().
**/
class SimTK_SIMBODY_EXPORT MobilizedBody::Ellipsoid : public MobilizedBody {
public:
    /** Default constructor provides an empty handle that can be assigned to
    reference any %MobilizedBody::Ellipsoid. **/
    Ellipsoid() {}
      
    /** Create an %Ellipsoid mobilizer between an existing parent (inboard) body 
    P and a new child (outboard) body B created by copying the given \a bodyInfo 
    into a privately-owned Body within the constructed %MobilizedBody object. 
    Specify the mobilizer frames F fixed to parent P and M fixed to child B.
    The ellipsoid is placed on the mobilizer's inboard frame F, 
    with semi-axis dimensions given in \a radii along F's x,y,z respectively.
    @see MobilizedBody for a diagram and explanation of terminology. **/
    Ellipsoid(MobilizedBody& parent, const Transform& X_PF,
              const Body& bodyInfo,  const Transform& X_BM,
              const Vec3& radii, Direction=Forward);

    /** Abbreviated constructor you can use if the mobilizer frames are 
    coincident with the parent and child body frames. **/
    Ellipsoid(MobilizedBody& parent,  const Body& bodyInfo,
              const Vec3& radii, Direction=Forward);

    /** This constructor assumes you'll set the ellipsoid dimensions later;
    meanwhile it uses some default dimensions. **/
    Ellipsoid(MobilizedBody& parent, const Transform& X_PF,
              const Body& bodyInfo,  const Transform& X_BM,
              Direction=Forward);

    /** This constructor assumes you'll set the ellipsoid dimensions later;
    meanwhile it uses some default dimensions. The parent and child body frames
    are used as the mobilizer frames. **/
    Ellipsoid(MobilizedBody& parent, const Body& bodyInfo, Direction=Forward);

    /** Modify the default semi-axis dimensions of the ellipsoid, given in
    the F frame. These are usually set on construction. **/
    Ellipsoid& setDefaultRadii(const Vec3& radii);
    /** Get the default semi-axis dimensions of the ellipsoid as specified 
    during construction or via setDefaultRadii(). **/
    const Vec3& getDefaultRadii() const;

    /** Override the semi-axis dimensions of the ellipsoid for this State.
    This is an Instance-stage variable: setting it invalidates Stage::Instance
    and higher, so the State must be re-realized to Position before any
    kinematic query reflects the new radii. All three components must be
    strictly positive. The topology default (see setDefaultRadii()) is used
    as the initial value when the State is created.
    @see setDefaultRadii(), getRadii() **/
    void setRadii(State& state, const Vec3& radii) const;
    /** Get the semi-axis dimensions of the ellipsoid currently in effect for
    this State. The State must have been realized to Stage::Instance or higher.
    @see setRadii(), getDefaultRadii() **/
    const Vec3& getRadii(const State& state) const;

    /** Compute dp_GB = J_r(state) * dRadii, where J_r[i] =
    d(p_GB[i])/d(radii) for this Ellipsoid mobilizer. The local Jacobian column
    is R_GF * diag(R_FM.col(2)); the resulting body shift is applied to this
    mobilizer's body and propagated through R_GP * R_PB chains to every
    descendant body, with zero for every other body in the system.

    @param[in]  state   Realized through Stage::Position.
    @param[in]  dRadii  Perturbation of the three semi-axis radii.
    @param[out] dp_GB   Per-body shift in Ground, indexed by
                        MobilizedBodyIndex. Resized to the system's body count
                        and zeroed before accumulation. **/
    void multiplyByPositionJacobianWrtRadii(const State& state,
                                            const Vec3& dRadii,
                                            Vector_<Vec3>& dp_GB) const;

    /** Compute dRadii = ~J_r(state) * dp_GB. Sums dp_GB over this body and
    all descendants, then projects onto the local Jacobian column.

    @param[in]  state  Realized through Stage::Position.
    @param[in]  dp_GB  Per-body perturbation-like vector, indexed by
                       MobilizedBodyIndex.
    @return            ~(R_GF * diag(R_FM.col(2))) * (subtree-sum of dp_GB). **/
    Vec3 multiplyByPositionJacobianWrtRadiiTranspose(
            const State& state,
            const Vector_<Vec3>& dp_GB) const;


    /** Provide a default orientation for this mobilizer if you don't want to
    start with the identity rotation (that is, alignment of the F and M 
    frames). This is the orientation the mobilizer will have in the default
    state for the containing System. The supplied Rotation will be converted
    to a quaternion and used as the four generalized coordinates q. **/
    Ellipsoid& setDefaultRotation(const Rotation& R_FM) {
        return setDefaultQ(R_FM.convertRotationToQuaternion());
    }
    /** Get the value for the default orientation of this mobilizer; unless
    changed by setDefaultRotation() it will be the identity rotation. **/
    Rotation getDefaultRotation() const {return Rotation(getDefaultQ());}

    Ellipsoid& addBodyDecoration(const Transform& X_BD, const DecorativeGeometry& g) {
        (void)MobilizedBody::addBodyDecoration(X_BD,g); return *this;
    }
    Ellipsoid& addOutboardDecoration(const Transform& X_MD,  const DecorativeGeometry& g) {
        (void)MobilizedBody::addOutboardDecoration(X_MD,g); return *this;
    }
    Ellipsoid& addInboardDecoration (const Transform& X_FD, const DecorativeGeometry& g) {
        (void)MobilizedBody::addInboardDecoration(X_FD,g); return *this;
    }

    Ellipsoid& setDefaultInboardFrame(const Transform& X_PF) {
        (void)MobilizedBody::setDefaultInboardFrame(X_PF); return *this;
    }

    Ellipsoid& setDefaultOutboardFrame(const Transform& X_BM) {
        (void)MobilizedBody::setDefaultOutboardFrame(X_BM); return *this;
    }



    // Generic default state Topology methods.
    const Quaternion& getDefaultQ() const;
    Quaternion& updDefaultQ();
    Ellipsoid& setDefaultQ(const Quaternion& q) {updDefaultQ()=q; return *this;}

    const Vec4& getQ(const State&) const;
    const Vec4& getQDot(const State&) const;
    const Vec4& getQDotDot(const State&) const;
    const Vec3& getU(const State&) const;
    const Vec3& getUDot(const State&) const;

    void setQ(State&, const Vec4&) const;
    void setU(State&, const Vec3&) const;

    const Vec4& getMyPartQ(const State&, const Vector& qlike) const;
    const Vec3& getMyPartU(const State&, const Vector& ulike) const;
   
    Vec4& updMyPartQ(const State&, Vector& qlike) const;
    Vec3& updMyPartU(const State&, Vector& ulike) const;

    /** @cond **/ // hide from Doxygen
    SimTK_INSERT_DERIVED_HANDLE_DECLARATIONS(Ellipsoid, EllipsoidImpl, 
                                             MobilizedBody);
    /** @endcond **/
};

} // namespace SimTK

#endif // SimTK_SIMBODY_MOBILIZED_BODY_ELLIPSOID_H_



