#ifndef SimTK_SIMBODY_MOBILIZER_KERNELS_H_
#define SimTK_SIMBODY_MOBILIZER_KERNELS_H_

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
Mobilizer kinematics that are shared by the Real implementation of a
mobilizer and the RigidBodyNode for other scalar types, as templates on the
scalar type T (as in ConstraintKernels.h). The callers evaluate the
mobilizer's inputs (and cache them, if they want to); the kernels turn them
into the cross-mobilizer transform X_FM, the hinge matrix H_FM, and its time
derivative HDot_FM (taken in F). **/

#include "simbody/internal/common.h"
#include "simbody/internal/tree/SimbodyTreeState.h"

#include <type_traits>

namespace SimTK {

//==============================================================================
//                              FUNCTION BASED
//==============================================================================
// A MobilizedBody::FunctionBased mobilizer with nu mobilities, whose qdots are
// its u's. Six spatial coordinates s, each a Function of some of the q's,
// give X_FM: three angles of a body-fixed rotation sequence about the unit
// axes a1, a2, a3 (the columns of Arot), in order, then three translations
// along the unit axes of Atrans, measured in F.
//
// The kernel's inputs are the spatial coordinates s(q), their partial
// derivatives Fq = ds/dq (6 x nu), and the time derivatives of the partials,
// FqDot = d/dt Fq (6 x nu, the second partials times u), all with every
// Function's value, including those of constant Functions (of no q's).
//
// With R_F1 = R(s0,a1) and R_F2 = R_F1*R(s1,a2), the angular velocity is
//   w_FM = a1 thetadot0 + R_F1 a2 thetadot1 + R_F2 a3 thetadot2
//        = W thetadot,  W = [a1, R_F1 a2, R_F2 a3],  thetadot = Fq_rot u,
// and the linear velocity is v_FM = Atrans Fq_trans u. So
//   H_rot = W Fq_rot,  H_trans = Atrans Fq_trans,
//   HDot_rot = WDot Fq_rot + W FqDot_rot,  HDot_trans = Atrans FqDot_trans,
// where W's columns rotate with the preceding rotations:
//   WDot = [0, (a1 thetadot0) x W1, (a1 thetadot0 + W1 thetadot1) x W2].
struct FunctionBasedMobilizerKernel {
    Mat33 Arot, Atrans; // columns are unit axes, expressed in F

    // The rotation by the given angle about the unit axis a.
    template <class T>
    static Rotation_<T> calcRotation(const T& angle, const Vec3& a) {
        if constexpr (std::is_same<T,Real>::value) {
            return Rotation(angle, UnitVec3(a, true));
        } else {
            // cos*I + (1-cos)*a*~a + sin*crossMat(a), which unlike
            // Rotation(angle, a) has no value-dependent branches.
            const T c = cos(angle), s = sin(angle), oneMinusC = 1 - c;
            Mat<3,3,T> R;
            for (int i=0; i < 3; ++i)
                for (int j=0; j < 3; ++j)
                    R(i,j) = oneMinusC*(a[i]*a[j]) + (i == j ? c : T(0));
            R(0,1) -= s*a[2]; R(1,0) += s*a[2];
            R(0,2) += s*a[1]; R(2,0) -= s*a[1];
            R(1,2) -= s*a[0]; R(2,1) += s*a[0];
            Rotation_<T> rotation;
            rotation.setRotationFromMat33TrustMe(R);
            return rotation;
        }
    }

    template <class T>
    Transform_<T> calcX_FM(const Vec<6,T>& s) const {
        Transform_<T> X_FM;
        X_FM.updR() = calcRotation(s[0], Arot.col(0))
                      * calcRotation(s[1], Arot.col(1))
                      * calcRotation(s[2], Arot.col(2));
        X_FM.updP() = s[3]*sbCast<T>(Atrans.col(0))
                      + s[4]*sbCast<T>(Atrans.col(1))
                      + s[5]*sbCast<T>(Atrans.col(2));
        return X_FM;
    }

    // W = [a1, R_F1 a2, R_F2 a3]; see above.
    template <class T>
    Mat<3,3,T> calcW(const Vec<6,T>& s) const {
        const Rotation_<T> R_F1 = calcRotation(s[0], Arot.col(0));
        const Rotation_<T> R_F2 = R_F1*calcRotation(s[1], Arot.col(1));
        Mat<3,3,T> W;
        W.col(0) = sbCast<T>(Arot.col(0));
        W.col(1) = R_F1*sbCast<T>(Arot.col(1));
        W.col(2) = R_F2*sbCast<T>(Arot.col(2));
        return W;
    }

    // H_FM; column i is the spatial velocity of M in F due to unit u_i.
    template <class T, int nu>
    void calcH(const Vec<6,T>& s, const Mat<6,nu,T>& Fq,
               Mat<2,nu,Vec<3,T>>& H_FM) const {
        const Mat<3,3,T> W = calcW(s);
        const Mat<3,3,T> A = sbCast<T>(Atrans);
        for (int i=0; i < nu; ++i) {
            H_FM(0,i) = W*Vec<3,T>(Fq(0,i), Fq(1,i), Fq(2,i));
            H_FM(1,i) = A*Vec<3,T>(Fq(3,i), Fq(4,i), Fq(5,i));
        }
    }

    // HDot_FM, the time derivative of H_FM in F.
    template <class T, int nu>
    void calcHDot(const Vec<6,T>& s, const Mat<6,nu,T>& Fq,
                  const Mat<6,nu,T>& FqDot, const Vec<nu,T>& u,
                  Mat<2,nu,Vec<3,T>>& HDot_FM) const {
        const Mat<3,3,T> W = calcW(s);
        const Mat<3,3,T> A = sbCast<T>(Atrans);
        // The rates of the three angles, thetadot = Fq_rot u.
        Vec<3,T> thetaDot(T(0));
        for (int k=0; k < 3; ++k)
            for (int j=0; j < nu; ++j)
                thetaDot[k] += Fq(k,j)*u[j];
        // The angular velocities of the frames after the first and second
        // rotations, which rotate the second and third axes.
        const Vec<3,T> w1 = W.col(0)*thetaDot[0];
        const Vec<3,T> w2 = w1 + W.col(1)*thetaDot[1];
        Mat<3,3,T> WDot;
        WDot.col(0) = Vec<3,T>(T(0));
        WDot.col(1) = w1 % W.col(1);
        WDot.col(2) = w2 % W.col(2);
        for (int i=0; i < nu; ++i) {
            const Vec<3,T> FqRot(Fq(0,i), Fq(1,i), Fq(2,i));
            const Vec<3,T> FqDotRot(FqDot(0,i), FqDot(1,i), FqDot(2,i));
            const Vec<3,T> FqDotTrans(FqDot(3,i), FqDot(4,i), FqDot(5,i));
            HDot_FM(0,i) = WDot*FqRot + W*FqDotRot;
            HDot_FM(1,i) = A*FqDotTrans;
        }
    }
};

} // namespace SimTK

#endif // SimTK_SIMBODY_MOBILIZER_KERNELS_H_
