#ifndef SimTK_SIMBODY_CONSTRAINT_EQUATIONS_H_
#define SimTK_SIMBODY_CONSTRAINT_EQUATIONS_H_

/* -------------------------------------------------------------------------- *
 *                               Simbody(tm)                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the SimTK biosimulation toolkit originating from           *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org/home/simbody.  *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 * Authors: Nick Bianco                                                       *
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
The constraint equations of the built-in Constraint types that support scalar
types other than Real, and descriptions of a SimbodyMatterSubsystem's enabled
constraints for evaluating them with those types.

The kernels (first) hold the equations and constraint forces of some of the
built-in Constraint types, as templates on the scalar type T. Each kernel
holds the constraint's (Real) parameters and computes with the poses X_AB,
spatial velocities V_AB, and spatial accelerations A_AB of its constrained
bodies, measured and expressed in the constraint's Ancestor frame A, which
are passed explicitly rather than taken from a State. The Real Constraint
implementations (in ConstraintImpl.h) call these with T=Real, and
MatterScalarRep<T> (in MultibodyScalarImpl.h) calls them with other scalar
types, so the equations are written once; see the derivations in
ConstraintImpl.h.

The descriptions (last) are used when a ScalarState<T> is realized through
Stage::Instance: SimbodyMatterSubsystem::calcConstraintDescriptions() asks
each enabled Constraint for its equations, with its parameters from the Real
State. The equations are plain data (a kernel or a Function, with the
constrained bodies or q's they apply to), so they can be created by the
compiled library and then evaluated with any scalar type: each equation type
has the same member templates, which MatterScalarRep<T> calls through
std::visit.

These are Simbody implementation details. **/

#include "simbody/internal/common.h"
#include "simbody/internal/tree/SimbodyTreeState.h"
#include "SimTKcommon/internal/FunctionEvaluator.h"

#include <type_traits>
#include <variant>
#include <vector>

namespace SimTK {

namespace ConstraintKernel {

// Station S on body B, given by p_BS (measured from Bo and expressed in B):
// its velocity in A (27 flops).
template <class T> inline Vec<3,T>
findStationVelocity(const Transform_<T>& X_AB, const SpatialVec_<T>& V_AB,
                    const Vec<3,T>& p_BS) {
    const Vec<3,T> p_BS_A = X_AB.R() * p_BS; // re-expressed in A
    return V_AB[1] + (V_AB[0] % p_BS_A);
}

// The acceleration of S in A, given p_BS re-expressed in A, and
// w_AB x (w_AB x p_BS_A) (15 flops).
template <class T> inline Vec<3,T>
findStationInAAcceleration(const SpatialVec_<T>& A_AB, const Vec<3,T>& p_BS_A,
                           const Vec<3,T>& wXwX_p_BS_A) {
    const Vec<3,T>& b_AB = A_AB[0]; const Vec<3,T>& a_AB = A_AB[1];
    return a_AB + (b_AB % p_BS_A) + wXwX_p_BS_A;
}

// Same, given only p_BS (48 flops).
template <class T> inline Vec<3,T>
findStationAcceleration(const Transform_<T>& X_AB, const SpatialVec_<T>& V_AB,
                        const SpatialVec_<T>& A_AB, const Vec<3,T>& p_BS) {
    const Vec<3,T>& w_AB = V_AB[0];
    const Vec<3,T> p_BS_A = X_AB.R() * p_BS;
    const Vec<3,T> wXwX_p_BS_A = w_AB % (w_AB % p_BS_A);
    return findStationInAAcceleration(A_AB, p_BS_A, wXwX_p_BS_A);
}

// Apply an A-frame force to station S of body B (30 flops).
template <class T> inline void
addInStationForce(const Transform_<T>& X_AB, const Vec<3,T>& p_BS,
                  const Vec<3,T>& forceInA, SpatialVec_<T>& bodyForceInA) {
    const Vec<3,T> p_BS_A = X_AB.R() * p_BS;
    bodyForceInA += SpatialVec_<T>(p_BS_A % forceInA, forceInA);
}

// Same, given p_BS re-expressed in A (15 flops).
template <class T> inline void
addInStationInAForce(const Vec<3,T>& p_BS_A, const Vec<3,T>& forceInA,
                     SpatialVec_<T>& bodyForceInA)
{   bodyForceInA += SpatialVec_<T>(p_BS_A % forceInA, forceInA); }

// Same, but subtract the force.
template <class T> inline void
subInStationInAForce(const Vec<3,T>& p_BS_A, const Vec<3,T>& negForceInA,
                     SpatialVec_<T>& bodyForceInA)
{   bodyForceInA -= SpatialVec_<T>(p_BS_A % negForceInA, negForceInA); }

template <class T> inline void
setErrors(const Vec<3,T>& e, T* err)
{   err[0] = e[0]; err[1] = e[1]; err[2] = e[2]; }

// The three "constant angle" errors and their derivatives that keep the axes
// of frames RB (on B) and RF (on F) aligned; see ConstantOrientationImpl.
template <class T> inline Vec<3,T>
calcOrientationErrors(const Rotation_<T>& RB, const Rotation_<T>& RF) {
    return Vec<3,T>(~RF.x()*RB.y(), ~RF.y()*RB.z(), ~RF.z()*RB.x());
}
template <class T> inline Vec<3,T>
calcOrientationDotErrors(const Rotation_<T>& RB, const Rotation_<T>& RF,
                         const Vec<3,T>& w_AB, const Vec<3,T>& w_AF) {
    const Vec<3,T> w_BF = w_AF-w_AB; // in A
    return Vec<3,T>(~w_BF * (RF.x() % RB.y()),
                    ~w_BF * (RF.y() % RB.z()),
                    ~w_BF * (RF.z() % RB.x()));
}
template <class T> inline Vec<3,T>
calcOrientationDotDotErrors(const Rotation_<T>& RB, const Rotation_<T>& RF,
                            const Vec<3,T>& w_AB, const Vec<3,T>& w_AF,
                            const Vec<3,T>& b_AB, const Vec<3,T>& b_AF) {
    const Vec<3,T> w_BF = w_AF-w_AB; // in A
    const Vec<3,T> b_BF = b_AF-b_AB; // in A
    return Vec<3,T>(
        dot( b_BF, RF.x() % RB.y() )
            + dot( w_BF, (w_AF%RF.x()) % RB.y() - (w_AB%RB.y()) % RF.x()),
        dot( b_BF, RF.y() % RB.z() )
            + dot( w_BF, (w_AF%RF.y()) % RB.z() - (w_AB%RB.z()) % RF.y()),
        dot( b_BF, RF.z() % RB.x() )
            + dot( w_BF, (w_AF%RF.z()) % RB.x() - (w_AB%RB.x()) % RF.z()));
}
template <class T> inline Vec<3,T>
calcOrientationTorque(const Rotation_<T>& RB, const Rotation_<T>& RF,
                      const T* lambda) {
    return   lambda[0] * (RF.x() % RB.y())
           + lambda[1] * (RF.y() % RB.z())
           + lambda[2] * (RF.z() % RB.x());
}

} // namespace ConstraintKernel

//==============================================================================
//                                   BALL
//==============================================================================
// Point P (point1) fixed on body B (B1) coincides with point S (point2) fixed
// on body F (B2). The force is applied at S and at the material point C of B
// coincident with S. 3 holonomic equations.
struct BallConstraintKernel {
    Vec3 point1, point2; // on B1 (in B1), on B2 (in B2)

    template <class T> void calcPositionErrors
       (const Transform_<T>& X_AB1, const Transform_<T>& X_AB2, T* perr) const {
        const Vec<3,T> p_AP = X_AB1 * sbCast<T>(point1);
        const Vec<3,T> p_AS = X_AB2 * sbCast<T>(point2);
        ConstraintKernel::setErrors<T>(p_AS - p_AP, perr);
    }
    template <class T> void calcPositionDotErrors
       (const Transform_<T>& X_AB1, const Transform_<T>& X_AB2,
        const SpatialVec_<T>& V_AB1, const SpatialVec_<T>& V_AB2,
        T* pverr) const {
        const Vec<3,T> p_FS = sbCast<T>(point2);
        const Vec<3,T> p_AS = X_AB2 * p_FS;
        const Vec<3,T> p_BC = ~X_AB1*p_AS; // C is a material point of body B
        const Vec<3,T> v_AS =
            ConstraintKernel::findStationVelocity(X_AB2, V_AB2, p_FS);
        const Vec<3,T> v_AC =
            ConstraintKernel::findStationVelocity(X_AB1, V_AB1, p_BC);
        ConstraintKernel::setErrors<T>(v_AS - v_AC, pverr);
    }
    template <class T> void calcPositionDotDotErrors
       (const Transform_<T>& X_AB1, const Transform_<T>& X_AB2,
        const SpatialVec_<T>& V_AB1, const SpatialVec_<T>& V_AB2,
        const SpatialVec_<T>& A_AB1, const SpatialVec_<T>& A_AB2,
        T* paerr) const {
        const Vec<3,T> p_FS = sbCast<T>(point2);
        const Vec<3,T> p_AS = X_AB2 * p_FS;
        const Vec<3,T> p_BC = ~X_AB1*p_AS; // C is a material point of body B
        const Vec<3,T> a_AS = ConstraintKernel::findStationAcceleration(
            X_AB2, V_AB2, A_AB2, p_FS);
        const Vec<3,T> a_AC = ConstraintKernel::findStationAcceleration(
            X_AB1, V_AB1, A_AB1, p_BC);
        ConstraintKernel::setErrors<T>(a_AS - a_AC, paerr);
    }
    template <class T> void addInPositionConstraintForces
       (const Transform_<T>& X_AB1, const Transform_<T>& X_AB2,
        const T* multipliers,
        SpatialVec_<T>& bodyForce1InA, SpatialVec_<T>& bodyForce2InA) const {
        const Vec<3,T> p_FS = sbCast<T>(point2);
        const Vec<3,T> p_AS = X_AB2 * p_FS;
        const Vec<3,T> p_BC = ~X_AB1 * p_AS; // shift to B origin, reexpress
        const Vec<3,T> force_A(multipliers[0], multipliers[1], multipliers[2]);
        ConstraintKernel::addInStationForce(X_AB2, p_FS,  force_A,
                                            bodyForce2InA);
        ConstraintKernel::addInStationForce(X_AB1, p_BC, Vec<3,T>(-force_A),
                                            bodyForce1InA);
    }
};

//==============================================================================
//                                   WELD
//==============================================================================
// Frame frameB fixed on body B coincides with frame frameF fixed on body F.
// 6 holonomic equations: 3 orientation then 3 position.
struct WeldConstraintKernel {
    Transform frameB, frameF; // on B (relative to B), on F (relative to F)

    template <class T> void calcPositionErrors
       (const Transform_<T>& X_AB, const Transform_<T>& X_AF, T* perr) const {
        const Rotation_<T> RB = X_AB.R() * sbCast<T>(frameB.R()); // in A
        const Rotation_<T> RF = X_AF.R() * sbCast<T>(frameF.R());
        ConstraintKernel::setErrors<T>(
            ConstraintKernel::calcOrientationErrors(RB, RF), perr);
        const Vec<3,T> p_AF1 = X_AB * sbCast<T>(frameB.p());
        const Vec<3,T> p_AF2 = X_AF * sbCast<T>(frameF.p());
        ConstraintKernel::setErrors<T>(p_AF2 - p_AF1, perr+3);
    }
    template <class T> void calcPositionDotErrors
       (const Transform_<T>& X_AB, const Transform_<T>& X_AF,
        const SpatialVec_<T>& V_AB, const SpatialVec_<T>& V_AF,
        T* pverr) const {
        const Rotation_<T> RB = X_AB.R() * sbCast<T>(frameB.R()); // in A
        const Rotation_<T> RF = X_AF.R() * sbCast<T>(frameF.R());
        ConstraintKernel::setErrors<T>(ConstraintKernel::
            calcOrientationDotErrors(RB, RF, V_AB[0], V_AF[0]), pverr);
        const Vec<3,T> p_FF2 = sbCast<T>(frameF.p());
        const Vec<3,T> p_AF2 = X_AF * p_FF2;
        const Vec<3,T> p_BC  = ~X_AB*p_AF2; // C is a material point of body B
        const Vec<3,T> v_AF2 =
            ConstraintKernel::findStationVelocity(X_AF, V_AF, p_FF2);
        const Vec<3,T> v_AC  =
            ConstraintKernel::findStationVelocity(X_AB, V_AB, p_BC);
        ConstraintKernel::setErrors<T>(v_AF2 - v_AC, pverr+3);
    }
    template <class T> void calcPositionDotDotErrors
       (const Transform_<T>& X_AB, const Transform_<T>& X_AF,
        const SpatialVec_<T>& V_AB, const SpatialVec_<T>& V_AF,
        const SpatialVec_<T>& A_AB, const SpatialVec_<T>& A_AF,
        T* paerr) const {
        const Rotation_<T> RB = X_AB.R() * sbCast<T>(frameB.R()); // in A
        const Rotation_<T> RF = X_AF.R() * sbCast<T>(frameF.R());
        ConstraintKernel::setErrors<T>(ConstraintKernel::
            calcOrientationDotDotErrors(RB, RF, V_AB[0], V_AF[0],
                                        A_AB[0], A_AF[0]), paerr);
        const Vec<3,T> p_FF2 = sbCast<T>(frameF.p());
        const Vec<3,T> p_AF2 = X_AF * p_FF2;
        const Vec<3,T> p_BC  = ~X_AB*p_AF2; // C is a material point of body B
        const Vec<3,T> a_AF2 = ConstraintKernel::findStationAcceleration(
            X_AF, V_AF, A_AF, p_FF2);
        const Vec<3,T> a_AC  = ConstraintKernel::findStationAcceleration(
            X_AB, V_AB, A_AB, p_BC);
        ConstraintKernel::setErrors<T>(a_AF2 - a_AC, paerr+3);
    }
    template <class T> void addInPositionConstraintForces
       (const Transform_<T>& X_AB, const Transform_<T>& X_AF,
        const T* multipliers,
        SpatialVec_<T>& bodyForceBInA, SpatialVec_<T>& bodyForceFInA) const {
        const Rotation_<T> RB = X_AB.R() * sbCast<T>(frameB.R()); // in A
        const Rotation_<T> RF = X_AF.R() * sbCast<T>(frameF.R());
        const Vec<3,T> torque_F_A =
            ConstraintKernel::calcOrientationTorque(RB, RF, multipliers);
        bodyForceFInA[0] += torque_F_A;
        bodyForceBInA[0] -= torque_F_A;

        const Vec<3,T> force_A(multipliers[3], multipliers[4], multipliers[5]);
        const Vec<3,T> p_FF2 = sbCast<T>(frameF.p());
        const Vec<3,T> p_AF2 = X_AF * p_FF2;
        const Vec<3,T> p_BC  = ~X_AB * p_AF2;
        ConstraintKernel::addInStationForce(X_AF, p_FF2, force_A,
                                            bodyForceFInA);
        ConstraintKernel::addInStationForce(X_AB, p_BC, Vec<3,T>(-force_A),
                                            bodyForceBInA);
    }
};

//==============================================================================
//                              POINT IN PLANE
//==============================================================================
// Follower point S fixed on body F stays in a plane (normal n, height h)
// fixed on body B. 1 holonomic equation.
struct PointInPlaneConstraintKernel {
    UnitVec3 planeNormal;   // n on B, exp. in B
    Real     planeHeight;   // h
    Vec3     followerPoint; // S on F, exp. in F

    template <class T> void calcPositionErrors
       (const Transform_<T>& X_AB, const Transform_<T>& X_AF, T* perr) const {
        const Vec<3,T> p_AS = X_AF * sbCast<T>(followerPoint);
        const Vec<3,T> p_BC = ~X_AB * p_AS; // C is material pt of B at S
        perr[0] = dot(p_BC, sbCast<T>(Vec3(planeNormal))) - T(planeHeight);
    }
    template <class T> void calcPositionDotErrors
       (const Transform_<T>& X_AB, const Transform_<T>& X_AF,
        const SpatialVec_<T>& V_AB, const SpatialVec_<T>& V_AF,
        T* pverr) const {
        const Vec<3,T> p_FS = sbCast<T>(followerPoint);
        const Vec<3,T> p_AS = X_AF * p_FS;
        const Vec<3,T> p_BC = ~X_AB * p_AS; // C is material pt of B at S
        const Vec<3,T> n_A  = X_AB.R() * sbCast<T>(Vec3(planeNormal));
        const Vec<3,T> v_AS =
            ConstraintKernel::findStationVelocity(X_AF, V_AF, p_FS);
        const Vec<3,T> v_AC =
            ConstraintKernel::findStationVelocity(X_AB, V_AB, p_BC);
        pverr[0] = dot( v_AS-v_AC, n_A );
    }
    template <class T> void calcPositionDotDotErrors
       (const Transform_<T>& X_AB, const Transform_<T>& X_AF,
        const SpatialVec_<T>& V_AB, const SpatialVec_<T>& V_AF,
        const SpatialVec_<T>& A_AB, const SpatialVec_<T>& A_AF,
        T* paerr) const {
        const Vec<3,T> p_FS = sbCast<T>(followerPoint);
        const Vec<3,T> p_AS = X_AF * p_FS;
        const Vec<3,T> p_BC = ~X_AB * p_AS; // C is material pt of B at S
        const Vec<3,T> n_A  = X_AB.R() * sbCast<T>(Vec3(planeNormal));
        const Vec<3,T>& w_AB = V_AB[0];
        const Vec<3,T> v_AS =
            ConstraintKernel::findStationVelocity(X_AF, V_AF, p_FS);
        const Vec<3,T> v_AC =
            ConstraintKernel::findStationVelocity(X_AB, V_AB, p_BC);
        const Vec<3,T> a_AS = ConstraintKernel::findStationAcceleration(
            X_AF, V_AF, A_AF, p_FS);
        const Vec<3,T> a_AC = ConstraintKernel::findStationAcceleration(
            X_AB, V_AB, A_AB, p_BC);
        paerr[0] = dot( (a_AS-a_AC) - T(2)*w_AB % (v_AS-v_AC), n_A );
    }
    template <class T> void addInPositionConstraintForces
       (const Transform_<T>& X_AB, const Transform_<T>& X_AF,
        const T* multipliers,
        SpatialVec_<T>& bodyForceBInA, SpatialVec_<T>& bodyForceFInA) const {
        const T& lambda = multipliers[0];
        const Vec<3,T> p_FS    = sbCast<T>(followerPoint);
        const Vec<3,T> p_AS    = X_AF * p_FS;
        const Vec<3,T> p_BC    = ~X_AB * p_AS; // measured & expressed in B
        const Vec<3,T> force_A =
            X_AB.R()*(lambda*sbCast<T>(Vec3(planeNormal)));
        ConstraintKernel::addInStationForce(X_AF, p_FS,  force_A,
                                            bodyForceFInA);
        ConstraintKernel::addInStationForce(X_AB, p_BC, Vec<3,T>(-force_A),
                                            bodyForceBInA);
    }
};

//==============================================================================
//                               POINT ON LINE
//==============================================================================
// Follower point S fixed on body F stays on a line (through point P, along z)
// fixed on body B, as two point-in-plane constraints with normals x and y
// perpendicular to the line. 2 holonomic equations.
struct PointOnLineConstraintKernel {
    UnitVec3 x, y;          // plane normals on B, exp. in B
    Vec3     pointOnLine;   // P on B, meas. & exp. in B
    Vec3     followerPoint; // S on F, meas. & exp. in F

    template <class T> void calcPositionErrors
       (const Transform_<T>& X_AB, const Transform_<T>& X_AF, T* perr) const {
        const Vec<3,T> p_AS = X_AF * sbCast<T>(followerPoint);
        const Vec<3,T> p_BC = ~X_AB * p_AS; // C is material pt of B at S
        const Vec<3,T> p_PC = p_BC - sbCast<T>(pointOnLine);
        perr[0] = ~p_PC * sbCast<T>(Vec3(x));
        perr[1] = ~p_PC * sbCast<T>(Vec3(y));
    }
    template <class T> void calcPositionDotErrors
       (const Transform_<T>& X_AB, const Transform_<T>& X_AF,
        const SpatialVec_<T>& V_AB, const SpatialVec_<T>& V_AF,
        T* pverr) const {
        const Vec<3,T> p_FS = sbCast<T>(followerPoint);
        const Vec<3,T> p_AS = X_AF * p_FS;
        const Vec<3,T> p_BC = ~X_AB * p_AS;
        const Vec<3,T> v_AS =
            ConstraintKernel::findStationVelocity(X_AF, V_AF, p_FS);
        const Vec<3,T> v_AC =
            ConstraintKernel::findStationVelocity(X_AB, V_AB, p_BC);
        const Vec<3,T> v_CS_B = ~X_AB.R()*(v_AS-v_AC); // reexpress in B
        pverr[0] = ~v_CS_B * sbCast<T>(Vec3(x));
        pverr[1] = ~v_CS_B * sbCast<T>(Vec3(y));
    }
    template <class T> void calcPositionDotDotErrors
       (const Transform_<T>& X_AB, const Transform_<T>& X_AF,
        const SpatialVec_<T>& V_AB, const SpatialVec_<T>& V_AF,
        const SpatialVec_<T>& A_AB, const SpatialVec_<T>& A_AF,
        T* paerr) const {
        const Vec<3,T> p_FS = sbCast<T>(followerPoint);
        const Vec<3,T> p_AS = X_AF * p_FS;
        const Vec<3,T> p_BC = ~X_AB * p_AS; // C is material pt of B at S
        const Vec<3,T>& w_AB = V_AB[0];
        const Vec<3,T> v_AS =
            ConstraintKernel::findStationVelocity(X_AF, V_AF, p_FS);
        const Vec<3,T> v_AC =
            ConstraintKernel::findStationVelocity(X_AB, V_AB, p_BC);
        const Vec<3,T> a_AS = ConstraintKernel::findStationAcceleration(
            X_AF, V_AF, A_AF, p_FS);
        const Vec<3,T> a_AC = ConstraintKernel::findStationAcceleration(
            X_AB, V_AB, A_AB, p_BC);
        const Vec<3,T> a_CS_B =
            ~X_AB.R()*(a_AS-a_AC - T(2) * w_AB % (v_AS-v_AC));
        paerr[0] = ~a_CS_B * sbCast<T>(Vec3(x));
        paerr[1] = ~a_CS_B * sbCast<T>(Vec3(y));
    }
    template <class T> void addInPositionConstraintForces
       (const Transform_<T>& X_AB, const Transform_<T>& X_AF,
        const T* multipliers,
        SpatialVec_<T>& bodyForceBInA, SpatialVec_<T>& bodyForceFInA) const {
        const Vec<3,T> p_FS    = sbCast<T>(followerPoint);
        const Vec<3,T> p_AS    = X_AF * p_FS;
        const Vec<3,T> p_BC    = ~X_AB * p_AS; // measured & expressed in B
        const Vec<3,T> force_B = multipliers[0] * sbCast<T>(Vec3(x))
                               + multipliers[1] * sbCast<T>(Vec3(y));
        const Vec<3,T> force_A = X_AB.R() * force_B;
        ConstraintKernel::addInStationForce(X_AF, p_FS,  force_A,
                                            bodyForceFInA);
        ConstraintKernel::addInStationForce(X_AB, p_BC, Vec<3,T>(-force_A),
                                            bodyForceBInA);
    }
};

//==============================================================================
//                              CONSTANT ANGLE
//==============================================================================
// The angle between axis b fixed on body B and axis f fixed on body F is
// constant. 1 holonomic equation.
struct ConstantAngleConstraintKernel {
    UnitVec3 axisB, axisF;     // on B (exp. in B), on F (exp. in F)
    Real     cosineOfAngle;

    template <class T> void calcPositionErrors
       (const Transform_<T>& X_AB, const Transform_<T>& X_AF, T* perr) const {
        const Vec<3,T> b_A = X_AB.R() * sbCast<T>(Vec3(axisB));
        const Vec<3,T> f_A = X_AF.R() * sbCast<T>(Vec3(axisF));
        perr[0] = dot(b_A, f_A) - T(cosineOfAngle);
    }
    template <class T> void calcPositionDotErrors
       (const Transform_<T>& X_AB, const Transform_<T>& X_AF,
        const SpatialVec_<T>& V_AB, const SpatialVec_<T>& V_AF,
        T* pverr) const {
        const Vec<3,T> b_A = X_AB.R() * sbCast<T>(Vec3(axisB));
        const Vec<3,T> f_A = X_AF.R() * sbCast<T>(Vec3(axisF));
        pverr[0] = dot( V_AF[0]-V_AB[0],  f_A % b_A );
    }
    template <class T> void calcPositionDotDotErrors
       (const Transform_<T>& X_AB, const Transform_<T>& X_AF,
        const SpatialVec_<T>& V_AB, const SpatialVec_<T>& V_AF,
        const SpatialVec_<T>& A_AB, const SpatialVec_<T>& A_AF,
        T* paerr) const {
        const Vec<3,T> b_A = X_AB.R() * sbCast<T>(Vec3(axisB));
        const Vec<3,T> f_A = X_AF.R() * sbCast<T>(Vec3(axisF));
        const Vec<3,T>& w_AB = V_AB[0]; const Vec<3,T>& w_AF = V_AF[0];
        const Vec<3,T>& b_AB = A_AB[0]; const Vec<3,T>& b_AF = A_AF[0];
        paerr[0] =    dot( b_AF-b_AB, f_A % b_A )
                    + dot( w_AF-w_AB, (w_AF%f_A) % b_A - (w_AB%b_A) % f_A);
    }
    template <class T> void addInPositionConstraintForces
       (const Transform_<T>& X_AB, const Transform_<T>& X_AF,
        const T* multipliers,
        SpatialVec_<T>& bodyForceBInA, SpatialVec_<T>& bodyForceFInA) const {
        const Vec<3,T> b_A = X_AB.R() * sbCast<T>(Vec3(axisB));
        const Vec<3,T> f_A = X_AF.R() * sbCast<T>(Vec3(axisF));
        const Vec<3,T> torque_F_A = multipliers[0] * (f_A % b_A); // on F, in A
        bodyForceFInA[0] += torque_F_A;
        bodyForceBInA[0] -= torque_F_A;
    }
};

//==============================================================================
//                                   ROD
//==============================================================================
// The distance between point Sf fixed on body F and point Sb fixed on body B
// is constant. 1 holonomic equation; see Constraint_RodImpl.h. The position
// and velocity quantities are computed once and reused (the Real Rod caches
// them in the State).
struct RodConstraintKernel {
    Vec3 p_FSf, p_BSb; // on F (in F), on B (in B)
    Real length;

    template <class T>
    struct PositionCache {
        Vec<3,T> p_FSf_A, p_BSb_A;  // stations expressed in A
        Vec<3,T> p_SfSb_A;          // vector from Sf to Sb, exp. in A
        T r, oor;                   // r=||p_SfSb_A||; oor=1/r (maybe Infinity)
        Vec<3,T> Cz_A;              // force direction, usually p_SfSb/r
        bool isSingular = false;    // if r is too small to be useful (Real)
    };
    template <class T>
    struct VelocityCache {
        Vec<3,T> wXwX_p_FSf_A;      // w_AF x (w_AF x p_FSf_A) [coriolis accel.]
        Vec<3,T> wXwX_p_BSb_A;      // w_AB x (w_AB x p_BSb_A) [coriolis accel.]
        Vec<3,T> pd_SfSb_A;         // v_ASb - vASf
        Vec<3,T> Czd_A;             // d/dt_A Cz_A -- depends on isSingular
    };

    // The distance between Sf and Sb. Other scalar types can't be compared,
    // so for them it's regularized, r = sqrt(||p||^2 + TinyReal^2), which is
    // never zero and differs from ||p|| by less than TinyReal. The position
    // error, its derivatives, and the force direction (r's gradient, p/r) are
    // all computed from it, so they're consistent, and finite and smooth
    // everywhere; when Sf and Sb coincide, the force direction is zero.
    template <class T> static T calcDistance(const Vec<3,T>& p_SfSb) {
        if constexpr (std::is_floating_point<T>::value)
            return p_SfSb.norm();
        else
            return sqrt(~p_SfSb*p_SfSb + T(TinyReal*TinyReal));
    }

    // About 72 flops. For Real, if r is tiny the direction is arbitrary (F's z
    // axis). For other scalar types, see calcDistance().
    template <class T> void calcPositionCache
       (const Transform_<T>& X_AF, const Transform_<T>& X_AB,
        PositionCache<T>& pc) const {
        pc.p_FSf_A = X_AF.R() * sbCast<T>(p_FSf);  // exp. in A, 15 flops
        const Vec<3,T> p_ASf = X_AF.p() + pc.p_FSf_A; // meas. from Ao
        pc.p_BSb_A = X_AB.R() * sbCast<T>(p_BSb);  // exp. in A, 15 flops
        const Vec<3,T> p_ASb = X_AB.p() + pc.p_BSb_A; // meas. from Ao
        pc.p_SfSb_A = p_ASb - p_ASf;  // vec from Sf to Sb, exp. in A
        pc.r = calcDistance(pc.p_SfSb_A); // ~20 flops
        pc.oor = T(1)/pc.r;           // ~10 flops (might be Infinity)
        pc.Cz_A = pc.p_SfSb_A * pc.oor; // 3 flops
        pc.isSingular = false;
        if constexpr (std::is_floating_point<T>::value) {
            if (pc.r < TinyReal) {
                pc.isSingular = true;
                pc.Cz_A = X_AF.z().asVec3(); // arbitrary
            }
        }
    }
    // About 57 flops.
    template <class T> void calcVelocityCache
       (const PositionCache<T>& pc, const SpatialVec_<T>& V_AF,
        const SpatialVec_<T>& V_AB, VelocityCache<T>& vc) const {
        const Vec<3,T>& Cz_A = pc.Cz_A;
        const Vec<3,T>& w_AF = V_AF[0]; const Vec<3,T>& v_AF = V_AF[1];
        const Vec<3,T>& w_AB = V_AB[0]; const Vec<3,T>& v_AB = V_AB[1];
        // These are d/dt_A p_FSf and d/dt_A p_BSb
        const Vec<3,T> wX_p_FSf_A = w_AF % pc.p_FSf_A;  // 9 flops
        const Vec<3,T> wX_p_BSb_A = w_AB % pc.p_BSb_A;  // 9 flops
        const Vec<3,T> v_ASf = v_AF + wX_p_FSf_A;       // 3 flops
        const Vec<3,T> v_ASb = v_AB + wX_p_BSb_A;       // 3 flops
        vc.pd_SfSb_A = v_ASb - v_ASf;                   // 3 flops
        // These are the Coriolis accelerations of Sf and Sb, needed later.
        vc.wXwX_p_FSf_A = w_AF % wX_p_FSf_A;            // 9 flops
        vc.wXwX_p_BSb_A = w_AB % wX_p_BSb_A;            // 9 flops
        // Calculate d/dt_A Cz. (For other scalar types, Cz = p/r is shorter
        // than a unit vector when p is tiny, and this is d/dt (p/r) with the
        // regularized r of calcDistance().)
        vc.Czd_A = pc.isSingular
            ? Vec<3,T>(w_AF % Cz_A) // rare
            : Vec<3,T>(pc.oor*(vc.pd_SfSb_A - (~vc.pd_SfSb_A*Cz_A)*Cz_A));
    }

    // perr = ||p_SfSb|| - d (~60 flops)
    template <class T> void calcPositionErrors
       (const Transform_<T>& X_AF, const Transform_<T>& X_AB, T* perr) const {
        const Vec<3,T> p_ASf = X_AF * sbCast<T>(p_FSf);
        const Vec<3,T> p_ASb = X_AB * sbCast<T>(p_BSb);
        const Vec<3,T> p_SfSb_A = p_ASb - p_ASf;    //  3 flops
        const T r = calcDistance(p_SfSb_A);         // ~20 flops
        perr[0] = r - T(length);                    //  1 flop
    }
    // verr = pd_SfSb_A . Cz (32 flops)
    template <class T> void calcPositionDotErrors
       (const PositionCache<T>& pc, const SpatialVec_<T>& V_AF,
        const SpatialVec_<T>& V_AB, T* pverr) const {
        const Vec<3,T>& w_AF = V_AF[0]; const Vec<3,T>& v_AF = V_AF[1];
        const Vec<3,T>& w_AB = V_AB[0]; const Vec<3,T>& v_AB = V_AB[1];
        const Vec<3,T> v_ASf = v_AF + w_AF % pc.p_FSf_A;    // 12 flops
        const Vec<3,T> v_ASb = v_AB + w_AB % pc.p_BSb_A;    // 12 flops
        const Vec<3,T> pd_SfSb_A = v_ASb - v_ASf;           //  3 flops
        pverr[0] = ~pd_SfSb_A * pc.Cz_A;                    //  5 flops
    }
    // aerr = pdd_SfSb . Cz + pd_SfSb . Czd (44 flops)
    template <class T> void calcPositionDotDotErrors
       (const PositionCache<T>& pc, const VelocityCache<T>& vc,
        const SpatialVec_<T>& A_AF, const SpatialVec_<T>& A_AB,
        T* paerr) const {
        const Vec<3,T> a_ASf = ConstraintKernel::findStationInAAcceleration(
            A_AF, pc.p_FSf_A, vc.wXwX_p_FSf_A);
        const Vec<3,T> a_ASb = ConstraintKernel::findStationInAAcceleration(
            A_AB, pc.p_BSb_A, vc.wXwX_p_BSb_A);
        const Vec<3,T> pdd_SfSb_A = a_ASb - a_ASf; // 2nd deriv in A
        paerr[0] = ~pdd_SfSb_A * pc.Cz_A + ~vc.pd_SfSb_A * vc.Czd_A;
    }
    template <class T> void addInPositionConstraintForces
       (const PositionCache<T>& pc, const T* multipliers,
        SpatialVec_<T>& bodyForceFInA, SpatialVec_<T>& bodyForceBInA) const {
        const Vec<3,T> force_A = multipliers[0] * pc.Cz_A;  // 3 flops
        ConstraintKernel::addInStationInAForce(pc.p_BSb_A, force_A,
                                               bodyForceBInA);
        ConstraintKernel::subInStationInAForce(pc.p_FSf_A, force_A,
                                               bodyForceFInA);
    }

    // The same equations with the signatures of the other two-body kernels,
    // computing the position and velocity quantities for each call (the
    // Real Rod caches them in the State instead).
    template <class T> void calcPositionDotErrors
       (const Transform_<T>& X_AF, const Transform_<T>& X_AB,
        const SpatialVec_<T>& V_AF, const SpatialVec_<T>& V_AB,
        T* pverr) const {
        PositionCache<T> pc;
        calcPositionCache(X_AF, X_AB, pc);
        calcPositionDotErrors(pc, V_AF, V_AB, pverr);
    }
    template <class T> void calcPositionDotDotErrors
       (const Transform_<T>& X_AF, const Transform_<T>& X_AB,
        const SpatialVec_<T>& V_AF, const SpatialVec_<T>& V_AB,
        const SpatialVec_<T>& A_AF, const SpatialVec_<T>& A_AB,
        T* paerr) const {
        PositionCache<T> pc;
        VelocityCache<T> vc;
        calcPositionCache(X_AF, X_AB, pc);
        calcVelocityCache(pc, V_AF, V_AB, vc);
        calcPositionDotDotErrors(pc, vc, A_AF, A_AB, paerr);
    }
    template <class T> void addInPositionConstraintForces
       (const Transform_<T>& X_AF, const Transform_<T>& X_AB,
        const T* multipliers, SpatialVec_<T>& bodyForceFInA,
        SpatialVec_<T>& bodyForceBInA) const {
        PositionCache<T> pc;
        calcPositionCache(X_AF, X_AB, pc);
        addInPositionConstraintForces(pc, multipliers, bodyForceFInA,
                                      bodyForceBInA);
    }
};

//==============================================================================
//                                 NO SLIP 1D
//==============================================================================
// The material points P0 of body B0 and P1 of body B1 that are coincident
// with contact point P fixed on case body C have the same velocity in C along
// direction n fixed on C. 1 nonholonomic equation. The case body may be one of
// the moving bodies.
struct NoSlip1DConstraintKernel {
    Vec3     contactPoint;     // P on C, exp. in C
    UnitVec3 noSlipDirection;  // n on C, exp. in C

    // P's stations on B0 and B1, and n in A.
    template <class T> void findStations
       (const Transform_<T>& X_AC, const Transform_<T>& X_AB0,
        const Transform_<T>& X_AB1,
        Vec<3,T>& p_P0, Vec<3,T>& p_P1, Vec<3,T>& n_A) const {
        const Vec<3,T> p_AP = X_AC * sbCast<T>(contactPoint); // P in A
        p_P0 = ~X_AB0 * p_AP;     // P0's station in B0
        p_P1 = ~X_AB1 * p_AP;     // P1's station in B1
        n_A  = X_AC.R() * sbCast<T>(Vec3(noSlipDirection));
    }

    // verr = ~(v_AP1 - v_AP0) * n_A
    template <class T> void calcVelocityErrors
       (const Transform_<T>& X_AC, const Transform_<T>& X_AB0,
        const Transform_<T>& X_AB1, const SpatialVec_<T>& V_AB0,
        const SpatialVec_<T>& V_AB1, T* verr) const {
        Vec<3,T> p_P0, p_P1, n_A;
        findStations(X_AC, X_AB0, X_AB1, p_P0, p_P1, n_A);
        const Vec<3,T> v_AP0 =
            ConstraintKernel::findStationVelocity(X_AB0, V_AB0, p_P0);
        const Vec<3,T> v_AP1 =
            ConstraintKernel::findStationVelocity(X_AB1, V_AB1, p_P1);
        verr[0] = ~(v_AP1-v_AP0) * n_A;
    }
    // aerr = ~(a_AP1 - a_AP0 - w_AC X (v_AP1 - v_AP0)) * n_A
    template <class T> void calcVelocityDotErrors
       (const Transform_<T>& X_AC, const Transform_<T>& X_AB0,
        const Transform_<T>& X_AB1, const SpatialVec_<T>& V_AC,
        const SpatialVec_<T>& V_AB0, const SpatialVec_<T>& V_AB1,
        const SpatialVec_<T>& A_AB0, const SpatialVec_<T>& A_AB1,
        T* vaerr) const {
        Vec<3,T> p_P0, p_P1, n_A;
        findStations(X_AC, X_AB0, X_AB1, p_P0, p_P1, n_A);
        const Vec<3,T> v_AP0 =
            ConstraintKernel::findStationVelocity(X_AB0, V_AB0, p_P0);
        const Vec<3,T> v_AP1 =
            ConstraintKernel::findStationVelocity(X_AB1, V_AB1, p_P1);
        const Vec<3,T>& w_AC = V_AC[0];
        const Vec<3,T> a_AP0 = ConstraintKernel::findStationAcceleration(
            X_AB0, V_AB0, A_AB0, p_P0);
        const Vec<3,T> a_AP1 = ConstraintKernel::findStationAcceleration(
            X_AB1, V_AB1, A_AB1, p_P1);
        vaerr[0] = ~(a_AP1-a_AP0 - w_AC % (v_AP1-v_AP0)) * n_A;
    }
    // Apply f=lambda*n to P1 of B1 and -f to P0 of B0.
    template <class T> void addInVelocityConstraintForces
       (const Transform_<T>& X_AC, const Transform_<T>& X_AB0,
        const Transform_<T>& X_AB1, const T* multipliers,
        SpatialVec_<T>& bodyForce0InA, SpatialVec_<T>& bodyForce1InA) const {
        const Vec<3,T> p_AP = X_AC * sbCast<T>(contactPoint); // P in A
        const Vec<3,T> p_P0 = ~X_AB0 * p_AP;   // P0's station in B0
        const Vec<3,T> p_P1 = ~X_AB1 * p_AP;   // P1's station in B1
        const Vec<3,T> force_A =
            X_AC.R()*(multipliers[0]*sbCast<T>(Vec3(noSlipDirection)));
        ConstraintKernel::addInStationForce(X_AB1, p_P1,  force_A,
                                            bodyForce1InA);
        ConstraintKernel::addInStationForce(X_AB0, p_P0, Vec<3,T>(-force_A),
                                            bodyForce0InA);
    }
};

//==============================================================================
//                         CONSTRAINT DESCRIPTIONS
//==============================================================================
/** The kinematics of a constraint's constrained bodies and mobilizers that
its equations use: the bodies' poses X_AB, velocities V_AB, and accelerations
A_AB in the constraint's Ancestor frame A (indexed by ConstrainedBodyIndex),
and the constrained q's and their time derivatives (indexed by
ConstrainedQIndex). Only the quantities available at the stage of each
computation are set; the others are null. **/
template <class T>
struct ConstraintKinematics_ {
    const Transform_<T>*  X_AB    = nullptr;
    const SpatialVec_<T>* V_AB    = nullptr;
    const SpatialVec_<T>* A_AB    = nullptr;
    const T*              q       = nullptr;
    const T*              qdot    = nullptr;
    const T*              qdotdot = nullptr;
};

/* Each equation type below has these member templates. Holonomic equations
have mp errors (Stage::Time: X_AB, q) and their first (Position: also V_AB,
qdot) and second (Velocity: also A_AB, qdotdot) time derivatives, and forces
for mp multipliers (Position: X_AB, q). Nonholonomic equations have mv errors
(Position: X_AB, V_AB) and their time derivatives (Velocity: also A_AB), and
forces for mv multipliers. Forces are on the constrained bodies (in A) and,
for holonomic equations, on the constrained q's. The methods for the kinds of
equations a type doesn't have do nothing. */

/** The holonomic equations of a kernel above that relates
two constrained bodies: Ball, Weld, PointInPlane, PointOnLine, ConstantAngle,
or Rod. **/
template <class Kernel>
struct TwoBodyConstraintEquations {
    ConstrainedBodyIndex body1, body2; // the kernel's two bodies, in order
    Kernel kernel;

    template <class T> using Kinematics = ConstraintKinematics_<T>;

    template <class T>
    void calcPositionErrors(const Kinematics<T>& k, T* perr) const
    {   kernel.calcPositionErrors(k.X_AB[body1], k.X_AB[body2], perr); }
    template <class T>
    void calcPositionDotErrors(const Kinematics<T>& k, T* pverr) const {
        kernel.calcPositionDotErrors(k.X_AB[body1], k.X_AB[body2],
                                     k.V_AB[body1], k.V_AB[body2], pverr);
    }
    template <class T>
    void calcPositionDotDotErrors(const Kinematics<T>& k, T* paerr) const {
        kernel.calcPositionDotDotErrors(k.X_AB[body1], k.X_AB[body2],
            k.V_AB[body1], k.V_AB[body2], k.A_AB[body1], k.A_AB[body2], paerr);
    }
    template <class T>
    void addInPositionConstraintForces(const Kinematics<T>& k,
        const T* multipliers, SpatialVec_<T>* bodyForces, T*) const {
        kernel.addInPositionConstraintForces(k.X_AB[body1], k.X_AB[body2],
            multipliers, bodyForces[body1], bodyForces[body2]);
    }
    template <class T>
    void calcVelocityErrors(const Kinematics<T>&, T*) const {}
    template <class T>
    void calcVelocityDotErrors(const Kinematics<T>&, T*) const {}
    template <class T>
    void addInVelocityConstraintForces(const Kinematics<T>&, const T*,
                                       SpatialVec_<T>*) const {}
};

/** The nonholonomic equation of a NoSlip1D constraint. **/
struct NoSlip1DConstraintEquations {
    ConstrainedBodyIndex caseBody, movingBody0, movingBody1;
    NoSlip1DConstraintKernel kernel;

    template <class T> using Kinematics = ConstraintKinematics_<T>;

    template <class T>
    void calcPositionErrors(const Kinematics<T>&, T*) const {}
    template <class T>
    void calcPositionDotErrors(const Kinematics<T>&, T*) const {}
    template <class T>
    void calcPositionDotDotErrors(const Kinematics<T>&, T*) const {}
    template <class T>
    void addInPositionConstraintForces(const Kinematics<T>&, const T*,
                                       SpatialVec_<T>*, T*) const {}
    template <class T>
    void calcVelocityErrors(const Kinematics<T>& k, T* verr) const {
        kernel.calcVelocityErrors(k.X_AB[caseBody], k.X_AB[movingBody0],
            k.X_AB[movingBody1], k.V_AB[movingBody0], k.V_AB[movingBody1],
            verr);
    }
    template <class T>
    void calcVelocityDotErrors(const Kinematics<T>& k, T* vaerr) const {
        kernel.calcVelocityDotErrors(k.X_AB[caseBody], k.X_AB[movingBody0],
            k.X_AB[movingBody1], k.V_AB[caseBody], k.V_AB[movingBody0],
            k.V_AB[movingBody1], k.A_AB[movingBody0], k.A_AB[movingBody1],
            vaerr);
    }
    template <class T>
    void addInVelocityConstraintForces(const Kinematics<T>& k,
        const T* multipliers, SpatialVec_<T>* bodyForces) const {
        kernel.addInVelocityConstraintForces(k.X_AB[caseBody],
            k.X_AB[movingBody0], k.X_AB[movingBody1], multipliers,
            bodyForces[movingBody0], bodyForces[movingBody1]);
    }
};

/** The holonomic equation of a Constraint::CoordinateCoupler, f(q) = 0 for a
(Real) Function f of some of the constrained q's, evaluated with
FunctionEvaluator<T> (so these methods can only be used for a scalar type T
that specializes it). See Constraint::CoordinateCouplerImpl in
ConstraintImpl.h; the equations are
    perr  = f(q)
    pverr = sum_i df/dq_i qdot_i
    paerr = sum_ij d2f/dq_i dq_j qdot_i qdot_j + sum_i df/dq_i qdotdot_i
and the multiplier's force on q_i is lambda df/dq_i. **/
struct CoordinateCouplerConstraintEquations {
    const Function* function = nullptr; // owned by the Constraint
    std::vector<ConstrainedQIndex> arguments;

    template <class T> using Kinematics = ConstraintKinematics_<T>;

    template <class T>
    void calcPositionErrors(const Kinematics<T>& k, T* perr) const
    {   perr[0] = value(*function, getArguments(k.q)); }
    template <class T>
    void calcPositionDotErrors(const Kinematics<T>& k, T* pverr) const {
        const std::vector<T> x = getArguments(k.q);
        T pverr0(0);
        for (int i=0; i < (int)arguments.size(); ++i)
            pverr0 += derivative(*function, {i}, x)
                      * k.qdot[arguments[i]];
        pverr[0] = pverr0;
    }
    template <class T>
    void calcPositionDotDotErrors(const Kinematics<T>& k, T* paerr) const {
        const std::vector<T> x = getArguments(k.q);
        const int n = (int)arguments.size();
        T paerr0(0);
        for (int i=0; i < n; ++i) {
            const T& qdoti = k.qdot[arguments[i]];
            for (int j=0; j < n; ++j)
                paerr0 += derivative(*function,
                              {i, j}, x) * qdoti * k.qdot[arguments[j]];
        }
        for (int i=0; i < n; ++i)
            paerr0 += derivative(*function, {i}, x)
                      * k.qdotdot[arguments[i]];
        paerr[0] = paerr0;
    }
    template <class T>
    void addInPositionConstraintForces(const Kinematics<T>& k,
        const T* multipliers, SpatialVec_<T>*, T* qForces) const {
        const std::vector<T> x = getArguments(k.q);
        for (int i=0; i < (int)arguments.size(); ++i)
            qForces[arguments[i]] += multipliers[0]
                * derivative(*function, {i}, x);
    }
    template <class T>
    void calcVelocityErrors(const Kinematics<T>&, T*) const {}
    template <class T>
    void calcVelocityDotErrors(const Kinematics<T>&, T*) const {}
    template <class T>
    void addInVelocityConstraintForces(const Kinematics<T>&, const T*,
                                       SpatialVec_<T>*) const {}

private:
    // These are instantiated for every scalar type (through std::visit), but
    // a CoordinateCoupler is only accepted for a scalar type with a
    // FunctionEvaluator (see MatterScalarRep<T>::realizeInstance()).
    template <class T>
    static T value(const Function& f, const std::vector<T>& x) {
        if constexpr (hasFunctionEvaluator<T>)
            return FunctionEvaluator<T>::calcValue(f, x);
        else return unavailable<T>();
    }
    template <class T>
    static T derivative(const Function& f, const std::vector<int>& components,
                        const std::vector<T>& x) {
        if constexpr (hasFunctionEvaluator<T>)
            return FunctionEvaluator<T>::calcDerivative(f, components, x);
        else return unavailable<T>();
    }
    template <class T> static T unavailable() {
        SimTK_ERRCHK_ALWAYS(false, "CoordinateCouplerConstraintEquations",
            "SimTK::FunctionEvaluator isn't specialized for this scalar type.");
        return T(0);
    }

    template <class T>
    std::vector<T> getArguments(const T* q) const {
        std::vector<T> x(arguments.size());
        for (int i=0; i < (int)arguments.size(); ++i) x[i] = q[arguments[i]];
        return x;
    }
};

/** The equations of an enabled Constraint of a supported type. **/
using ConstraintEquations = std::variant<
    TwoBodyConstraintEquations<BallConstraintKernel>,
    TwoBodyConstraintEquations<WeldConstraintKernel>,
    TwoBodyConstraintEquations<PointInPlaneConstraintKernel>,
    TwoBodyConstraintEquations<PointOnLineConstraintKernel>,
    TwoBodyConstraintEquations<ConstantAngleConstraintKernel>,
    TwoBodyConstraintEquations<RodConstraintKernel>,
    NoSlip1DConstraintEquations,
    CoordinateCouplerConstraintEquations>;

/** The q's and u's of one of a Constraint's constrained mobilizers, as
contiguous ranges of the Constraint's constrained q's and u's. **/
struct ConstrainedMobilizerNodeInfo {
    MobilizedBodyIndex mobod;
    int firstConstrainedQ = 0, nq = 0;
    int firstConstrainedU = 0, nu = 0;
};

/** One enabled Constraint, from the Real State: its equations, where they go,
and what they apply to. **/
struct ConstraintDescription {
    ConstraintIndex index;
    /// The numbers of holonomic, nonholonomic, and acceleration-only
    /// constraint equations.
    int mp = 0, mv = 0, ma = 0;
    /// The offsets of this Constraint's equations within the matter
    /// subsystem's holonomic, nonholonomic, and acceleration-only blocks
    /// (see SBInstancePerConstraintInfo).
    int holoOffset = 0, nonholoOffset = 0, accOnlyOffset = 0;
    /// The constrained bodies, indexed by ConstrainedBodyIndex.
    std::vector<MobilizedBodyIndex> constrainedBodies;
    /// The Ancestor of the constrained bodies, in whose frame A the
    /// constraint equations are written (GroundIndex if it is Ground).
    MobilizedBodyIndex ancestor;
    /// The constrained mobilizers, indexed by ConstrainedMobilizerIndex.
    std::vector<ConstrainedMobilizerNodeInfo> constrainedMobilizers;
    /// The subsystem's q and u indices of the constrained q's and u's,
    /// indexed by ConstrainedQIndex and ConstrainedUIndex.
    std::vector<QIndex> constrainedQ;
    std::vector<UIndex> constrainedU;
    /// The equations.
    ConstraintEquations equations;
};

} // namespace SimTK

#endif // SimTK_SIMBODY_CONSTRAINT_EQUATIONS_H_
