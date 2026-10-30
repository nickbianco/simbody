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

/* This file contains implementations of the base class methods for the
templatized class RigidBodyNodeSpec<dof, noR_FM>, and instantiations of
the class for all possible values of the arguments. */

#include "SimbodyMatterSubsystemRep.h"
#include "RigidBodyNode.h"
#include "RigidBodyNodeSpec.h"

#include "RigidBodyNodeSpec_Pin.h"
#include "RigidBodyNodeSpec_Slider.h"
#include "RigidBodyNodeSpec_Ball.h"
#include "RigidBodyNodeSpec_Free.h"
#include "RigidBodyNodeSpec_Custom.h"

//==============================================================================
//                              CALC H_PB_G
//==============================================================================
// Same for all mobilizers.
// CAUTION: our H matrix definition is transposed from Jain and Schwieters.
// Cost: 60 + 45*dof flops
template<int dof, bool noR_FM> void
RigidBodyNodeSpec<dof, noR_FM>::
calcParentToChildVelocityJacobianInGround
   (const SBModelVars&          mv,
    const SBInstanceVars&       iv,
    const SBInstanceCache&      ic,
    const SBTreePositionCache&  pc,
    HType&                      H_PB_G) const
{
    const SBInstancePerMobodInfo& info =
        ic.getMobodInstanceInfo(this->getNodeNum());
    const bool noX_MB = info.noX_MB;
    const bool noR_PF = info.noR_PF;

    const HType& H_FM = getH_FM(pc);

    // We want R_GF so we can reexpress the cross-joint velocity V_FB (==V_PB)
    // in the ground frame, to get V_PB_G.

    const Rotation& R_PF = getX_PF(iv).R();      // fixed config of F in P

    // Calculated already since we're going base to tip.
    const Rotation& R_GP = getX_GP(pc).R(); // parent orientation in ground
    const Rotation  R_GF = (noR_PF ? R_GP : R_GP * R_PF);     // 45 flops

    if (noX_MB || noR_FM)
        H_PB_G = R_GF * H_FM;       // 3*dof flops
    else {
        // want r_MB_F, that is, the vector from Mo to Bo, expressed in F
        const Vec3&     r_MB   = getX_MB(ic).p();   // fixed
        const Rotation& R_FM   = getX_FM(pc).R();   // just calculated
        const Vec3      r_MB_F = (noR_FM ? r_MB : R_FM*r_MB);       // 15 flops
        HType H_MB_F;
        H_MB_F[0] =  Vec3(0); // fills top row with zero
        H_MB_F[1] = -r_MB_F % H_FM[0]; // 9*dof (negation not actually done)
        H_PB_G = R_GF * (H_FM + H_MB_F); // 36*dof flops
    }
}

//==============================================================================
//                            CALC H_PB_G_DOT
//==============================================================================
// Same for all mobilizers.
// CAUTION: our H matrix definition is transposed from Jain and Schwieters. 
// Cost is 69 + 65*dof flops
template<int dof, bool noR_FM> void
RigidBodyNodeSpec<dof, noR_FM>::
calcParentToChildVelocityJacobianInGroundDot(
    const SBModelVars&          mv,
    const SBInstanceVars&       iv,
    const SBInstanceCache&      ic,
    const SBTreePositionCache&  pc,
    const SBTreeVelocityCache&  vc,
    HType&                      HDot_PB_G) const
{
    const SBInstancePerMobodInfo& info =
        ic.getMobodInstanceInfo(this->getNodeNum());
    const bool noX_MB = info.noX_MB;
    const bool noR_PF = info.noR_PF;

    const HType& H_FM    = getH_FM(pc);
    const HType& HDot_FM = getHDot_FM(vc);

    // We want R_GF so we can reexpress the cross-joint velocity V_FB (==V_PB)
    // in the ground frame, to get V_PB_G.

    const Rotation& R_PF = getX_PF(iv).R();       // fixed config of F in P

    // Calculated already since we're going base to tip.
    const Rotation& R_GP = getX_GP(pc).R();     // parent orientation in ground
    const Rotation  R_GF = (noR_PF ? R_GP : R_GP * R_PF); // 45 flops (TODO: again??)

    const Vec3& w_GF = getV_GP(vc)[0]; // F and P have same angular velocity

    // Note: time derivative of R_GF is crossMat(w_GF)*R_GF.
    //      H = H_PB_G =  R_GF * (H_FM + H_MB_F) (see above method)
    const HType& H_PB_G = getH(pc);
    if (noX_MB || noR_FM)
        HDot_PB_G = R_GF * HDot_FM // 48*dof
                  + HType(w_GF % H_PB_G[0], 
                          w_GF % H_PB_G[1]);
    else {
        // want r_MB_F, that is, the vector from OM to OB, expressed in F 
        const Vec3&     r_MB   = getX_MB(ic).p();     // fixed
        const Rotation& R_FM   = getX_FM(pc).R();   // just calculated
        const Vec3      r_MB_F = (noR_FM ? r_MB : R_FM*r_MB); // 15 flops

        const Vec3& w_FM = getV_FM(vc)[0]; // local angular velocity

        HType HDot_MB_F;
        HDot_MB_F[0] = Vec3(0);
        HDot_MB_F[1] =          -r_MB_F  % HDot_FM[0] // 21*dof + 9 flops
                       - (w_FM % r_MB_F) % H_FM[0];


        HDot_PB_G =   R_GF * (HDot_FM + HDot_MB_F) // 54*dof
                    + HType(w_GF % H_PB_G[0], 
                            w_GF % H_PB_G[1]);
    }
}

//==============================================================================
//                       CALC REVERSE MOBILIZER H_FM
//==============================================================================
// This is the default implementation for turning H_MF into H_FM. 
// A mobilizer can override this to do it faster.
// From the Simbody theory manual,
//      H_FM_w = - R_FM*H_MF_w
//      H_FM_v = -(R_FM*H_MF_v + p_FM x H_FM_w)
//             
template<int dof, bool noR_FM> void
RigidBodyNodeSpec<dof, noR_FM>::calcReverseMobilizerH_FM(
    const SBStateDigest& sbs,
    HType&               H_FM) const
{
    // Must use "upd" here rather than "get" because this is
    // called during realize(Position).
    const SBTreePositionCache& pc = sbs.updTreePositionCache();

    HType H_MF;
    calcAcrossJointVelocityJacobian(sbs, H_MF);

    const Rotation& R_FM = getX_FM(pc).R();
    const Vec3&     p_FM = getX_FM(pc).p();

    // Make cross product matrix (3 flops). Saves a few flops (3*dof) to
    // transpose this and get the negative of the cross product matrix.
    const Mat33     np_FM_x  = ~crossMat(p_FM);

    if (noR_FM) {
        H_FM[0] = -H_MF[0];
        H_FM[1] = np_FM_x * H_FM[0] - H_MF[1]; // 15*dof flops
    }
    else {
        H_FM[0] = -(R_FM * H_MF[0]);                // 18*dof flops
        H_FM[1] = np_FM_x * H_FM[0] - R_FM*H_MF[1]; // 33*dof flops
    }
}

//==============================================================================
//                     CALC REVERSE MOBILIZER HDOT_FM
//==============================================================================
// This is the default implementation for turning HDot_MF into HDot_FM. 
// A mobilizer can override this to do it faster.
// We depend on H_FM having already been calculated.
//
// From the Simbody theory manual,
//      HDot_FM_w = -R_FM * HDot_MF_w + w_FM_x H_FM_w
//                  
//      HDot_FM_v = -R_FM * HDot_MF_v + w_FM_x H_FM_v 
//                  - (p_FM_x HDot_FM_w
//                     + (v_FM_x - w_FM_x p_FM_x)H_FM_w)
//
// where "a_x" indicates the cross product matrix of vector a.
//  
template<int dof, bool noR_FM> void
RigidBodyNodeSpec<dof, noR_FM>::calcReverseMobilizerHDot_FM(
    const SBStateDigest& sbs,
    HType&               HDot_FM) const
{
    const SBTreePositionCache& pc = sbs.getTreePositionCache();
    // Must use "upd" here rather than "get" because this is
    // called during realize(Velocity).
    const SBTreeVelocityCache& vc = sbs.updTreeVelocityCache();

    HType HDot_MF;
    calcAcrossJointVelocityJacobianDot(sbs, HDot_MF);

    const Rotation& R_FM    = getX_FM(pc).R();
    const Vec3&     p_FM    = getX_FM(pc).p();
    const HType&    H_FM    = getH_FM(pc);

    const Vec3&     w_FM    = getV_FM(vc)[0];
    const Vec3&     v_FM    = getV_FM(vc)[1];
    
    // Make cross product matrices.
    const Mat33     p_FM_x  = crossMat(p_FM);   // 3 flops
    const Mat33     w_FM_x  = crossMat(w_FM);   // 3 flops
    const Mat33     v_FM_x  = crossMat(v_FM);   // 3 flops
    const Mat33     vwp     = v_FM_x - w_FM_x*p_FM_x;   // 54 flops

    // Initialize both rows with the first two terms above.
    HDot_FM = w_FM_x*H_FM - (noR_FM ? HDot_MF : R_FM*HDot_MF); // 66*dof flops

    // Add in the additional terms in the second row.
    HDot_FM[1] -= p_FM_x * HDot_FM[0] + vwp * H_FM[0];  // 36*dof flops
}


//==============================================================================
//                     REALIZE ARTICULATED BODY INERTIAS
//==============================================================================
// Compute articulated body inertia and related quantities for this body B.
// This must be called tip-to-base (inward).
//
// Given only position-related quantities from the State 
//      Mk  (this body's spatial inertia matrix)
//      Phi (composite body child-to-parent shift matrix)
//      H   (joint transition matrix; sense is transposed from Jain & Schwieters)
// we calculate dynamic quantities 
//      P   (articulated body inertia)
//    PPlus (articulated body inertia as seen through the mobilizer)
// For a prescribed mobilizer, we have PPlus==P. Otherwise we also compute
//      D   (factored mass matrix LDL' diagonal part D=~H*P*H)
//      DI  (inverse of D)
//      G   (P * H * DI)
//    PPlus (P - P * H * DI * ~H * P = P - G * ~H * P)
// and put them in the state cache.
//
// This is Algorithm 6.1 on page 106 of Jain's 2011 book modified to accommodate
// prescribed motion as described on page 323. Note that although P+ is "as 
// felt" on the inboard side of the mobilizer it is still calculated about 
// Bo just as is P.
//
// Cost is 93 flops per child plus
//   n^3 + 23*n^2 + 115*n + 12
//   e.g. pin=143, ball=591 (197/dof), free=1746 (291/dof)
// Note that per-child cost is paid just once for each non-base body in
// the whole tree; that is, each body is touched just once.
template<int dof, bool noR_FM> void
RigidBodyNodeSpec<dof, noR_FM>::
realizeArticulatedBodyInertiasInward(
    const SBInstanceCache&          ic,
    const SBTreePositionCache&      pc,
    SBArticulatedBodyInertiaCache&  abc) const 
{
    ArticulatedInertia& P = updP(abc);

    // Start with the spatial inertia of the current body (in Ground frame).
    P = ArticulatedInertia(getMk_G(pc)); // 12 flops

    // For each child, we previously took its articulated body inertia P and 
    // removed the portion of that inertia that can't be felt from the parent
    // because of the joint mobilities. That is, we calculated 
    // P+ = P - P H DI ~H P, where the second term is the projection of P into
    // the mobility space of the child's inboard mobilizer. Note that if the
    // child's mobilizer is prescribed, then the entire inertia will be felt
    // by the parent so P+ = P in that case. Now we're going to shift P+
    // from child to parent: Pparent += Phi*P+*~Phi.
    // TODO: can this be optimized for the common case where the
    // child is a terminal body and hence its P is an ordinary
    // spatial inertia? (Spatial inertia shift is 37 flops vs. 72 here; not
    // really much help.)
    for (unsigned i=0; i<children.size(); ++i) {
        const PhiMatrix&          phiChild   = children[i]->getPhi(pc);
        const ArticulatedInertia& PPlusChild = children[i]->getPPlus(abc);

        // Apply the articulated body shift.
        // This takes 93 flops (72 for the shift and 21 to add it in).
        // (Note that PPlusChild==PChild if child's mobilizer is prescribed.)
        P += PPlusChild.shift(phiChild.l());
    }

    // Now compute PPlus. P+ = P for a prescribed mobilizer. Otherwise
    // it is P+ = P - P H DI ~H P = P - G*~PH. In the prescribed case
    // we leave G, D, DI untouched -- they should have been set to NaN at
    // Instance stage.
    ArticulatedInertia& PPlus = updPPlus(abc);

    if (isUDotKnown(ic)) {
        PPlus = P;  // prescribed
        return;
    }

    // This is a non-prescribed mobilizer. Compute D, DI, G then P+.

    const HType&  H  = getH(pc);
    HType&        G  = updG(abc);
    Mat<dof,dof>& D  = updD(abc);
    Mat<dof,dof>& DI = updDI(abc);

    const HType PH = P*H;   // 66*dof   flops
    D  = ~H * PH;           // 11*dof^2 flops (symmetric result)

    // this will throw an exception if the matrix is ill conditioned
    DI = D.invert();                        // ~dof^3 flops (symmetric)
    G  = PH * DI;                           // 12*dof^2-6*dof flops

    // Want P+ = P - G*~PH. We can do this in about 55*dof flops.
    // These require 9 dot products of length dof. The symmetric ones could
    // be done with 6 dot products instead for a small savings but this gives
    // us a chance to symmetrize and hopefully clean up some numerical errors.
    // The full price for all three is 54*dof-27 flops.
    Mat33 massMoment = G.row(0)*~PH.row(1); // (full)            9*(2*dof-1) flops
    Mat33 mass       = G.row(1)*~PH.row(1); // symmetric result  9*(2*dof-1) flops
    Mat33 inertia    = G.row(0)*~PH.row(0); // symmetric result  9*(2*dof-1) flops
    // These must be symmetrized due to numerical errors for 12 more flops. 
    SymMat33 symMass( mass(0,0), 
                     (mass(1,0)+mass(0,1))/2,  mass(1,1), 
                     (mass(2,0)+mass(0,2))/2, (mass(2,1)+mass(1,2))/2, mass(2,2));
    SymMat33 symInertia( 
         inertia(0,0), 
        (inertia(1,0)+inertia(0,1))/2,  inertia(1,1), 
        (inertia(2,0)+inertia(0,2))/2, (inertia(2,1)+inertia(1,2))/2, inertia(2,2));
    PPlus = P - ArticulatedInertia(symMass, massMoment, symInertia); // 21 flops
}



//==============================================================================
//                                   CALC UDOT
//==============================================================================
// To be called from tip to base.
// Temps do not need to be initialized.
//
// First pass
// ----------
// Given previously-calculated quantities from the State 
//      P       this body's articulated body inertia
//     Phi      composite body child-to-parent shift matrix
//      H       joint transition matrix; sense is transposed from Jain
//      G       P * H * DI
// and supplied arguments
//      F       body force applied to B
//      f       generalize forces applied to B's mobilities
//    udot_p    known udots (if mobilizer is prescribed)
// calculate 
//      z       articulated body residual force on B
//    zPlus     AB residual force as felt on inboard side of mobilizer
//     eps      f - ~H*z  gen force plus gen equiv of body forces
// For a prescribed mobilizer we have zPlus=z. Otherwise, zPlus = z + G*eps.
//
// This is the first (inward) loop of Algorithm 16.2 on page 323 of Jain's
// 2011 book, modified to include the applied body force and not including
// the auxiliary quantity "nu=DI*eps".
template<int dof, bool noR_FM> void
RigidBodyNodeSpec<dof, noR_FM>::calcUDotPass1Inward(
    const SBInstanceCache&                  ic,
    const SBTreePositionCache&              pc,
    const SBArticulatedBodyInertiaCache&    abc,
    const SBArticulatedBodyVelocityCache&   abvc,
    const Real*                             jointForces,
    const SpatialVec*                       bodyForces,
    const Real*                             allUDot,
    SpatialVec*                             allZ,
    SpatialVec*                             allZPlus,
    Real*                                   allEpsilon) const 
{
    const Vec<dof>&   f     = fromU(jointForces);
    const SpatialVec& F     = bodyForces[nodeNum];
    SpatialVec&       z     = allZ[nodeNum];
    SpatialVec&       zPlus = allZPlus[nodeNum];
    Vec<dof>&         eps   = toU(allEpsilon);

    const bool isPrescribed = isUDotKnown(ic);
    const HType&              H = getH(pc);
    const ArticulatedInertia& P = getP(abc);
    const HType&              G = getG(abc);

    // z = Pa+b - F
    z = getArticulatedBodyCentrifugalForces(abvc) - F; // 6 flops

    // z += P H udot_p if prescribed
    if (isPrescribed && !isUDotKnownToBeZero(ic)) {
        const Vec<dof>& udot = fromU(allUDot);
        z += P*(H*udot); // 66+12*dof flops
    }

    // z += sum(Phi(child) * zPlus(child)) for all children
    for (unsigned i=0; i<children.size(); ++i) {
        const PhiMatrix&  phiChild   = children[i]->getPhi(pc);
        const SpatialVec& zPlusChild = allZPlus[children[i]->getNodeNum()];
        z += phiChild * zPlusChild; // 18 flops
    }

    eps  = f - ~H*z; // 12*dof flops

    zPlus = z;
    if (!isPrescribed)
        zPlus += G*eps; // 12*dof flops
}

//
// Calculate acceleration in internal coordinates, based on the last set
// of forces that were reduced into epsilon (e.g., see above).
// Base to tip: temp allA_GB does not need to be initialized before
// beginning the iteration.
//
template<int dof, bool noR_FM> void 
RigidBodyNodeSpec<dof, noR_FM>::calcUDotPass2Outward(
    const SBInstanceCache&                  ic,
    const SBTreePositionCache&              pc,
    const SBArticulatedBodyInertiaCache&    abc,
    const SBTreeVelocityCache&              vc,
    const SBDynamicsCache&                  dc,
    const Real*                             allEpsilon,
    SpatialVec*                             allA_GB,
    Real*                                   allUDot,
    Real*                                   allTau) const
{
    const Vec<dof>& eps  = fromU(allEpsilon);
    SpatialVec&     A_GB = allA_GB[nodeNum];
    Vec<dof>&       udot = toU(allUDot);    // pull out this node's udot

    const bool isPrescribed = isUDotKnown(ic);
    const HType&              H   = getH(pc);
    const PhiMatrix&          phi = getPhi(pc);
    const ArticulatedInertia& P   = getP(abc);
    const SpatialVec&         a   = getMobilizerCoriolisAcceleration(vc);
    const Mat<dof,dof>&       DI  = getDI(abc);
    const HType&              G   = getG(abc);

    // Shift parent's acceleration outward (Ground==0). 12 flops
    const SpatialVec& A_GP  = allA_GB[parent->getNodeNum()]; 
    const SpatialVec  APlus = ~phi * A_GP;

    if (isPrescribed) {
        Vec<dof>& tau = updTau(ic,allTau);  // pull out this node's tau
        // This is f - ~H(P*APlus + z); compare Jain 16.17b. Note sign
        // change since our tau is on the LHS while his is on the RHS.
        tau = eps - ~H*(P*APlus); // 66 + 12*dof flops
    } else {
        udot = DI*eps - ~G*APlus; // 2*dof^2 + 11*dof
    }

    A_GB = APlus + H*udot + a;
}

 
//==============================================================================
//                          MULTIPLY BY M INVERSE
//==============================================================================
// Temps do not need to be initialized.
//
// This calculates udot = M^-1 f in two O(n) passes. Note that we are ignoring 
// velocities; if there are any velocity-dependent forces you care about, they 
// should already be in f.
//
// When there is prescribed motion, it is already reflected in the articulated
// body inertias, which were formed with rigid body shifts across the prescribed
// mobilizers. In that case you can think of the udots and generalized forces f
// as partitioned into two sets each: udot={udot_r, udot_p}, f={f_r, f_p}. We
// can discuss them as though the partitions were contiguous although they are
// not and the code deals with that properly. So now you can view the system as
//     [ Mrr Mrp ] [udot_r]   [  0  ]   [f_r]
//     [~Mrp Mpp ] [udot_p] + [tau_p] = [f_p]
// where udot_p, f_r, and f_p are given and the unknowns are udot_r (the free 
// accelerations) and tau_p (the unknown forces that enforce the prescribed 
// motion). This produces two equations when multiplied out:
//     (1) Mrr udot_r = f_r -  Mrp udot_p
//     (2)      tau_p = f_p - ~Mrp udot_r - Mpp udot_p 
//
// Now we can define what the present method does: it returns
//                 [udot_r]   [udot_r]   [Mrr^-1 0] [f_r]
//                 [udot_p] = [   0  ] = [  0    0] [f_p]
// f_p is ignored and won't be examined; udot_p will be set to zero on return.
//
// Cost per body is 
//      30 + 47*ndof_r + 2*ndof_r^2
// where ndof_r is the number of u's for a regular (non-prescribed) mobilizer;
// 0 for a prescribed one.

// Pass 1, to be called from tip to base.
template<int dof, bool noR_FM> void
RigidBodyNodeSpec<dof, noR_FM>::multiplyByMInvPass1Inward(
    const SBInstanceCache&                  ic,
    const SBTreePositionCache&              pc,
    const SBArticulatedBodyInertiaCache&    abc,
    const Real*                             jointForces,
    SpatialVec*                             allZ,
    SpatialVec*                             allZPlus,
    Real*                                   allEpsilon) const
{
    const Vec<dof>&   f     = fromU(jointForces);
    SpatialVec&       z     = allZ[nodeNum];
    SpatialVec&       zPlus = allZPlus[nodeNum];
    Vec<dof>&         eps   = toU(allEpsilon);

    const bool isPrescribed = isUDotKnown(ic);
    const HType&              H = getH(pc);
    const HType&              G = getG(abc);

    z = 0;

    for (unsigned i=0; i<children.size(); i++) {
        const PhiMatrix&  phiChild  = children[i]->getPhi(pc);
        const SpatialVec& zPlusChild = allZPlus[children[i]->getNodeNum()];
        z += phiChild * zPlusChild; // 18 flops
    }

    zPlus = z;
    if (!isPrescribed) {
        eps    = f - ~H*z;  // 12*dof flops
        zPlus += G*eps;     // 12*dof flops
    }
}


// Pass 2 of multiplyByMInv.
// Base to tip: temp allA_GB does not need to be initialized before
// beginning the iteration.
template<int dof, bool noR_FM> void 
RigidBodyNodeSpec<dof, noR_FM>::multiplyByMInvPass2Outward(
    const SBInstanceCache&                  ic,
    const SBTreePositionCache&              pc,
    const SBArticulatedBodyInertiaCache&    abc,
    const Real*                             allEpsilon,
    SpatialVec*                             allA_GB,
    Real*                                   allUDot) const
{
    const Vec<dof>& eps  = fromU(allEpsilon);
    SpatialVec&     A_GB = allA_GB[nodeNum];
    Vec<dof>&       udot = toU(allUDot); // pull out this node's udot

    const bool isPrescribed = isUDotKnown(ic);
    const HType&        H   = getH(pc);
    const PhiMatrix&    phi = getPhi(pc);
    const Mat<dof,dof>& DI  = getDI(abc);
    const HType&        G   = getG(abc);

    // Shift parent's acceleration outward (Ground==0). 12 flops
    const SpatialVec& A_GP  = allA_GB[parent->getNodeNum()]; 
    const SpatialVec  APlus = ~phi * A_GP;

    // For a prescribed mobilizer, set udot==0.
    if (isPrescribed) {
        udot = 0;
        A_GB = APlus;
    } else {
        udot = DI*eps - ~G*APlus;   // 2dof^2 + 11 dof flops
        A_GB = APlus + H*udot;      // 12 dof flops
    }
}



//==============================================================================
//                     CALC BODY ACCELERATIONS FROM UDOT
//==============================================================================
// This method calculates:
//      A_GB = J*udot + Jdot*u
// in O(N) time, where udot is supplied as an argument and Jdot*u is the 
// coriolis acceleration which we get from the velocity cache. This also serves
// as pass 1 for inverse dynamics.
//
// This must be called base to tip. The cost is 12*dof + 18 flops.
template<int dof, bool noR_FM> void 
RigidBodyNodeSpec<dof, noR_FM>::
calcBodyAccelerationsFromUdotOutward(
    const SBTreePositionCache&  pc,
    const SBTreeVelocityCache&  vc,
    const Real*                 allUDot,
    SpatialVec*                 allA_GB) const
{
    const Vec<dof>& udot = fromU(allUDot);
    SpatialVec&     A_GB = allA_GB[nodeNum];

    // Shift parent's A_GB outward. (Ground A_GB is zero.) 12 flops.
    const SpatialVec A_GP = ~getPhi(pc) * allA_GB[parent->getNodeNum()];

    // 12*dof+6 flops.
    A_GB = A_GP + getH(pc)*udot + getMobilizerCoriolisAcceleration(vc); 
}



//==============================================================================
//                            CALC INVERSE DYNAMICS
//==============================================================================
// This algorithm calculates 
//      f = M*udot + C(u) - f_applied 
// in O(N) time, where C(u) are the velocity-dependent coriolis, centrifugal and 
// gyroscopic forces, f_applied are the applied body and joint forces, and udot 
// is given.
//
// Pass 1 is base to tip and is just a calculation of body accelerations arising
// from udot and the coriolis accelerations. It is embodied in the reusable
// calcBodyAccelerationsFromUdotOutward() method above.
//
// Pass 2 is tip to base. It takes body accelerations from pass 1 (including coriolis
// accelerations as well as hinge accelerations), and the applied forces,
// and calculates the additional hinge forces that would be necessary to produce 
// the observed accelerations.
//
// Costs for inverse dynamics include both passes:
//      pass1: 12*dof + 18 flops
//      pass2: 12*dof + 75 flops
//             -----------------
//      total: 24*dof + 93 flops
template<int dof, bool noR_FM> void
RigidBodyNodeSpec<dof, noR_FM>::
calcInverseDynamicsPass2Inward(
    const SBTreePositionCache&  pc,
    const SBTreeVelocityCache&  vc,
    const SpatialVec*           allA_GB,
    const Real*                 jointForces,
    const SpatialVec*           bodyForces,
    SpatialVec*                 allF,   // temp
    Real*                       allTau) const 
{
    const Vec<dof>&   myJointForce  = fromU(jointForces);
    const SpatialVec& myBodyForce   = bodyForces[nodeNum];
    const SpatialVec& A_GB          = allA_GB[nodeNum];
    SpatialVec&       F             = allF[nodeNum];
    Vec<dof>&         tau           = toU(allTau);

    // Start with rigid body force from desired body acceleration and
    // gyroscopic forces due to angular velocity, minus external forces
    // applied directly to this body.
    F = getMk_G(pc)*A_GB + getGyroscopicForce(vc) - myBodyForce; // 57 flops

    // Add in forces on children, shifted to this body.
    for (unsigned i=0; i<children.size(); ++i) {
        const PhiMatrix&  phiChild  = children[i]->getPhi(pc);
        const SpatialVec& FChild    = allF[children[i]->getNodeNum()];
        F += phiChild * FChild;         // 18 flops
    }

    // Project body forces into hinge space and subtract any hinge forces already
    // being applied to get the remaining hinge forces needed.
    tau = ~getH(pc)*F - myJointForce;   // 12*dof flops
}



//==============================================================================
//                             MULTIPLY BY M
//==============================================================================
// The next two methods calculate x=M*v (or f=M*a) in O(N) time, given v.
// The first one is called base to tip.
// Cost is: pass1 12*dof+12 flops
//          pass2 11*dof+63 flops
//          total 23*dof + 75
// TODO: Possibly this could be much faster if composite body inertias R were
// precalculated along with D = ~H*R*H. Then I *think* this could be a single
// pass with tau = D*udot. Whether this is worth it would depend on how much
// this gets re-used after R and D are calculated.
template<int dof, bool noR_FM> void 
RigidBodyNodeSpec<dof, noR_FM>::multiplyByMPass1Outward(
    const SBTreePositionCache&  pc,
    const Real*                 allUDot,
    SpatialVec*                 allA_GB) const
{
    const Vec<dof>& udot = fromU(allUDot);
    SpatialVec&     A_GB = allA_GB[nodeNum];

    // Shift parent's A_GB outward. (Ground A_GB is zero.)
    const SpatialVec A_GP = ~getPhi(pc) * allA_GB[parent->getNodeNum()]; // 12 flops

    A_GB = A_GP + getH(pc)*udot;    // 12*dof flops
}

// Call tip to base after calling multiplyByMPass1Outward() for each
// rigid body.
template<int dof, bool noR_FM> void
RigidBodyNodeSpec<dof, noR_FM>::multiplyByMPass2Inward(
    const SBTreePositionCache&  pc,
    const SpatialVec*           allA_GB,
    SpatialVec*                 allF,   // temp
    Real*                       allTau) const 
{
    const SpatialVec& A_GB  = allA_GB[nodeNum];
    SpatialVec&       F     = allF[nodeNum];
    Vec<dof>&         tau   = toU(allTau);

    // 45 flops because Mk has a nice structure
    F = getMk_G(pc)*A_GB; 

    for (unsigned i=0; i<children.size(); ++i) {
        const PhiMatrix&  phiChild  = children[i]->getPhi(pc);
        const SpatialVec& FChild    = allF[children[i]->getNodeNum()];
        F += phiChild * FChild; // 18 flops
    }

    tau = ~getH(pc)*F;          // 11*dof flops
}



//==============================================================================
//                                  REALIZE Y
//==============================================================================
// To be called base to tip.
// This is calculating what Abhi Jain calls the operational space compliance
// kernel in his 2011 book. This is the inverse of the operational space inertia
// for each body at its body frame. Also, see Equation 20 in Rodriguez,Jain, 
// & Kreutz-Delgado:  A spatial operator algebra 
// for manipulator modeling and control. Intl. J. Robotics Research 
// 10(4):371-381 (1991).
template<int dof, bool noR_FM> void
RigidBodyNodeSpec<dof, noR_FM>::realizeYOutward
   (const SBInstanceCache&                  ic,
    const SBTreePositionCache&              pc,
    const SBArticulatedBodyInertiaCache&    abc,
    SBDynamicsCache&                        dc) const
{
    if (isUDotKnown(ic)) {
        //TODO: (sherm 090810) is this right?
        assert(false);
        //updY(dc) = (~getPhi(pc) * parent->getY(dc)) * getPhi(pc); // rigid shift
        return;
    }

    // Compute psi. Jain has TauBar=I-G*~H but we're negating that to G*~H-I 
    // because we can save 30 flops by just subtracting 1 from the diagonals 
    // rather than having to negate all the off-diagonals. Then Psi ends up 
    // with the wrong sign here also, which doesn't matter because we multiply 
    // by it twice.

    SpatialMat tauBar = getG(abc)*~getH(pc);// 11*dof^2 flops
    tauBar(0,0) -= 1; // subtract identity matrix (only touches diags: 3 flops)
    tauBar(1,1) -= 1; //    "    (3 flops)
    SpatialMat psi = getPhi(pc)*tauBar; // ~100 flops

    // TODO: this is very expensive (~1000 flops?) Could cut be at least half
    // by exploiting symmetry. Also, does Psi have special structure?
    // And does this need to be computed for every body or only those
    // which are loop "base" bodies or some such?


    // Psi here has the opposite sign from Jain's, but we're multiplying twice
    // by it here so it doesn't matter.
    updY(dc) = (getH(pc) * getDI(abc) * ~getH(pc)) 
                + (~psi * parent->getY(dc) * psi);
}



//==============================================================================
//                       MULTIPLY BY SYSTEM JACOBIAN
//==============================================================================
// Calculate product of kinematic Jacobian J=~Phi*H and a mobility-space vector. Requires 
// that Phi and H are available, so this should only be called in Stage::Position or higher.
// This does not change the cache at all.
//
// Note that if the vector v==u (generalized speeds) then the result is V_GB=J*u, the
// body spatial velocities generated by those speeds.
//
// Call base to tip (outward).
//
template<int dof, bool noR_FM> void
RigidBodyNodeSpec<dof, noR_FM>::
multiplyBySystemJacobian(
    const SBTreePositionCache&  pc,
    const Real*                 v,
    SpatialVec*                 Jv) const
{
    const Vec<dof>& in  = fromU(v);
    SpatialVec&     out = Jv[nodeNum];

    // Shift parent's result outward (ground result is 0).
    const SpatialVec outP = ~getPhi(pc) * Jv[parent->getNodeNum()]; // 12 flops

    out = outP + getH(pc)*in;  // 12*dof flops
}

//==============================================================================
//                   MULTIPLY BY SYSTEM JACOBIAN TRANSPOSE
//==============================================================================
// Calculate product of kinematic Jacobian transpose ~J=~H*Phi and a spatial
// forces vector on each of the outboard bodies. Requires that Phi and H are 
// available, so this should only be called in Stage::Position or higher. This 
// does not  change the cache at all.
// NOTE (sherm 060214): I reworked this from the original. This one no longer 
// incorporates applied hinge gradients if there are any; just add those in at 
// the end if you want them.
//
// (sherm 060727) In spatial operators, this calculates ~H*Phi*F where F are the 
// spatial forces applied to each body. See Schwieters Eq. 41. (But with sense 
// of H transposed.)
//
// Call tip to base.
//
template<int dof, bool noR_FM> void
RigidBodyNodeSpec<dof, noR_FM>::
multiplyBySystemJacobianTranspose(
    const SBTreePositionCache&  pc,
    SpatialVec*                 zTmp,
    const SpatialVec*           X, 
    Real*                       JtX) const
{
    const SpatialVec& in  = X[getNodeNum()];
    Vec<dof>&         out = Vec<dof>::updAs(&JtX[getUIndex()]);
    SpatialVec&       z   = zTmp[getNodeNum()];

    z = in;

    for (unsigned i=0; i<children.size(); ++i) {
        const SpatialVec& zChild   = zTmp[children[i]->getNodeNum()];
        const PhiMatrix&  phiChild = children[i]->getPhi(pc);

        z += phiChild * zChild; // 18 flops
    }

    out = ~getH(pc) * z; // 11*dof flops
}

//==============================================================================
//                       CALC EQUIVALENT JOINT FORCES
//==============================================================================
// To be called from tip to base.
// Temps do not need to be initialized.
// (sherm 060727) In spatial operators, this calculates ~H*Phi*(F-(MA+b))
template<int dof, bool noR_FM> void
RigidBodyNodeSpec<dof, noR_FM>::calcEquivalentJointForces(
    const SBTreePositionCache&  pc,
    const SBTreeVelocityCache&  vc,
    const SpatialVec*           bodyForces,
    SpatialVec*                 allZ,
    Real*                       jointForces) const 
{
    const SpatialVec& myBodyForce  = bodyForces[nodeNum];
    SpatialVec&       z            = allZ[nodeNum];
    Vec<dof>&         eps          = toU(jointForces);

    // Centrifugal forces are MA+b where M is body spatial inertia,
    // A is total coriolis acceleration, and b is gyroscopic force.
    z = myBodyForce - getTotalCentrifugalForces(vc);

    for (unsigned i=0; i<children.size(); ++i) {
        const SpatialVec& zChild    = allZ[children[i]->getNodeNum()];
        const PhiMatrix&  phiChild  = children[i]->getPhi(pc);

        z += phiChild * zChild; 
    }

    eps  = ~getH(pc) * z;
}


//==============================================================================
//                      INVERSE DYNAMICS DERIVATIVES
//==============================================================================
// Analytic first-order partials of the inverse dynamics residual. These three
// methods are the per-body halves of the operator driven by
// SimbodyMatterSubsystemRep::calcTreeResidualForceDerivatives(); see
// Simbody/tests/adhoc/InverseDynamicsSandbox.cpp for the derivation and its
// numerical verification, and Jain, "Robot and Multibody Dynamics", Ch. 18 for
// the operator sensitivities cited below.
//
// Spatial vectors are referred to each body's own origin Bo and expressed in
// Ground, as everywhere else in Simbody. W(w) denotes the angular-only cross
// operator diag(w~, w~), Jain's Htilde*_w. Each sensitivity is a commutator
// with W, and commutators are applied rather than formed, using
//     [W X - X W] y = W(X y) - X(W y),
// which keeps every operation below to cross products and structured
// applications of Phi and Mk_G.

// Outward: dH, dV, dA and the u-side eV, eA for this body.
template<int dof, bool noR_FM> void
RigidBodyNodeSpec<dof, noR_FM>::calcInverseDynamicsDerivativesOutward
   (const SBTreePositionCache& pc, const SBTreeVelocityCache& vc,
    const SpatialVec& H_col, int selfMode, InverseDynamicsDerivativesWorkspace& w) const
{
    const MobilizedBodyIndex k = this->getNodeNum();
    const MobilizedBodyIndex p = this->parent->getNodeNum();
    const int                ux0 = this->getUIndex();

    const Vec3& w_i = H_col[0];          // perturbing mode, angular part
    const Vec3& v_i = H_col[1];          // and linear part
    const PhiMatrix&  Phi = this->getPhi(pc);
    const SpatialVec& V_p = this->getV_GP(vc);
    const SpatialVec& V_k = this->getV_GB(vc);

    // rho = p_BoMo in Ground, fixed in this body, and R_GF the orientation of
    // the inboard frame; both are per-body and were built once by the caller.
    // H_PB_G is built from H_FM by rotating with R_GF and shifting from Mo to
    // Bo, and differentiating that construction gives the self-column term.
    const Vec3&     rho  = w.rho[k];
    const Rotation& R_GF = w.R_GF[k];

    //--------------------------------------------------------------------------
    // Hinge-matrix sensitivity.                                  [Jain 18.29b]
    //   i > k : dH = W(w_i) H
    //   i == k: the modes are fixed in F while their line of action passes
    //           through Mo, which is material in this body, so differentiating
    //           H_PB_G = R_GF (H_FM + H_MB_F) at fixed R_GF gives
    //             dH = R_GF dH_FM + (0, -(w_d x rho_F) x H_FM_w
    //                                   - r_MB_F x dH_FM_w)
    //           which reduces to (0, -H_w x (w_i x rho)) when H_FM is constant.
    //--------------------------------------------------------------------------
    // Zero first, so that a mobilizer which cannot supply its partials degrades
    // to the constant-H_FM formulas rather than reading garbage. The Rep
    // validates the whole tree before any sweep runs, so a false return here
    // means the node was driven directly.
    Vec3 dHFMstore[2*dof], dHDotFMstore[2*dof];
    for (int j=0; j < 2*dof; ++j) dHFMstore[j] = dHDotFMstore[j] = Vec3(0);
    const HType& dH_FM    = HType::getAs(dHFMstore);
    const HType& dHDot_FM = HType::getAs(dHDotFMstore);

    const HType& H = this->getH(pc);
    if (selfMode < 0) {
        for (int e=0; e < dof; ++e)
            w.dH[ux0+e] = SpatialVec(w_i % H(e)[0], w_i % H(e)[1]);
    } else {
        bool ok = this->calcAcrossJointVelocityJacobianPartial(
                      *w.sbs, selfMode, dHFMstore);
        ok = ok && this->calcAcrossJointVelocityJacobianDotPartial(
                       *w.sbs, w.u, selfMode, dHDotFMstore);
        assert(ok); (void)ok;
        for (int e=0; e < dof; ++e) {
            // Rotate dH_FM into Ground, then add the moving-Mo correction.
            const SpatialVec dHr(R_GF*dH_FM(e)[0], R_GF*dH_FM(e)[1]);
            w.dH[ux0+e] = dHr
                + SpatialVec(Vec3(0), -(H(e)[0] % (w_i % rho))
                                      + (rho % dHr[0]));
        }
    }

    //--------------------------------------------------------------------------
    // Shift-operator sensitivity, applied not formed.             [Jain 18.29a]
    //   i > k : (~dPhi) y = W(~Phi y) - ~Phi(W y)
    //   i == k: only this joint's own translation enters, dPhi =
    //           [[0, v_i~],[0,0]], whose transpose sends y to (0, -v_i x y_w).
    //--------------------------------------------------------------------------
    const SpatialVec& A_p = w.A_GB[p];
    SpatialVec dPhiT_V, dPhiT_A;
    if (selfMode >= 0) {
        dPhiT_V = SpatialVec(Vec3(0), -(v_i % V_p[0]));
        dPhiT_A = SpatialVec(Vec3(0), -(v_i % A_p[0]));
    } else {
        const SpatialVec sV = ~Phi * V_p, sA = ~Phi * A_p;
        dPhiT_V = SpatialVec(w_i % sV[0], w_i % sV[1])
                - ~Phi * SpatialVec(w_i % V_p[0], w_i % V_p[1]);
        dPhiT_A = SpatialVec(w_i % sA[0], w_i % sA[1])
                - ~Phi * SpatialVec(w_i % A_p[0], w_i % A_p[1]);
    }

    //--------------------------------------------------------------------------
    // Velocity sensitivity, q and u sides.
    //   dV = (~dPhi) V_p + ~Phi dV_p + sum_e dH u
    //   eV = ~Phi eV_p + H_col 1[i == k]
    //--------------------------------------------------------------------------
    SpatialVec dVr(Vec3(0), Vec3(0));
    for (int e=0; e < dof; ++e) dVr += w.dH[ux0+e] * w.u[ux0+e];
    w.dV[k] = dPhiT_V + ~Phi * w.dV[p] + dVr;

    const SpatialVec eVr = (selfMode >= 0) ? H_col
                                           : SpatialVec(Vec3(0), Vec3(0));
    w.eV[k] = ~Phi * w.eV[p] + eVr;

    //--------------------------------------------------------------------------
    // Coriolis-acceleration sensitivity. Simbody's cross-mobilizer term has the
    // closed form
    //   a_c = (0, wp x (v_k - v_p)) + W(wp) Vr + (0, -wr x (wr x rho))
    // with Vr the cross-mobilizer velocity, so differentiating it needs only
    // dV, dVr and d(rho) = w_i x rho. The u side drops the d(rho) term since
    // rho is configuration-only.
    //--------------------------------------------------------------------------
    const SpatialVec& Vr = this->getV_PB_G(vc);
    const Vec3& wp = V_p[0];
    const Vec3& wr = Vr[0];
    const Vec3& dwp = w.dV[p][0];
    const Vec3& ewp = w.eV[p][0];
    const Vec3  dwr = dVr[0], ewr = eVr[0];
    const Vec3  drho = w_i % rho;

    // When H_FM depends on q the closed form above is incomplete: a_c also
    // contains Z = R_GF*(HDot_FM u) together with its Mo shift (0, rho x Z_w).
    // Z is per-body and zero for every mobilizer with HDot_FM identically zero,
    // so the next few lines cost nothing in the common case.
    const SpatialVec& Z = w.Z[k];
    SpatialVec dZ, eZ(Vec3(0), Vec3(0));
    if (selfMode < 0) {
        // An ancestor's coordinate rotates Z rigidly; the F-local HDot_FM and u
        // are untouched.
        dZ = SpatialVec(w_i % Z[0], w_i % Z[1]);
    } else {
        // dZ/dq_d = R_GF*(sum_e dHDot_FM(e) u_e).
        SpatialVec sd(Vec3(0), Vec3(0));
        for (int e=0; e < dof; ++e) sd += dHDot_FM(e) * w.u[ux0+e];
        dZ = SpatialVec(R_GF*sd[0], R_GF*sd[1]);

        // HDot_FM u is quadratic in u, so differentiating it in u gives two
        // terms: dZ/du_d = R_GF*(sum_e dH_FM(e) u_e + HDot_FM(d)).
        SpatialVec se = this->getHDot_FM(vc)(selfMode);
        for (int e=0; e < dof; ++e) se += dH_FM(e) * w.u[ux0+e];
        eZ = SpatialVec(R_GF*se[0], R_GF*se[1]);
    }

    const SpatialVec da_c =
          SpatialVec(Vec3(0), dwp % (V_k[1] - V_p[1])
                            + wp  % (w.dV[k][1] - w.dV[p][1]))
        + SpatialVec(dwp % Vr[0] + wp % dVr[0],
                     dwp % Vr[1] + wp % dVr[1])
        + SpatialVec(Vec3(0), -(dwr % (wr % rho))
                              -(wr  % (dwr % rho))
                              -(wr  % (wr  % drho)))
        + dZ + SpatialVec(Vec3(0), drho % Z[0] + rho % dZ[0]);

    const SpatialVec ea_c =
          SpatialVec(Vec3(0), ewp % (V_k[1] - V_p[1])
                            + wp  % (w.eV[k][1] - w.eV[p][1]))
        + SpatialVec(ewp % Vr[0] + wp % eVr[0],
                     ewp % Vr[1] + wp % eVr[1])
        + SpatialVec(Vec3(0), -(ewr % (wr % rho))
                              -(wr  % (ewr % rho)))
        + eZ + SpatialVec(Vec3(0), rho % eZ[0]);

    //--------------------------------------------------------------------------
    // Acceleration sensitivity. H depends on q alone, so the u side has no
    // dH udot term.
    //--------------------------------------------------------------------------
    SpatialVec dA = dPhiT_A + ~Phi * w.dA[p] + da_c;
    for (int e=0; e < dof; ++e) dA += w.dH[ux0+e] * w.udot[ux0+e];
    w.dA[k] = dA;
    w.eA[k] = ~Phi * w.eA[p] + ea_c;
}

// Inward: dF, eF, this body's Jacobian rows, and the push to the parent.
template<int dof, bool noR_FM> void
RigidBodyNodeSpec<dof, noR_FM>::calcInverseDynamicsDerivativesInward
   (const SBTreePositionCache& pc, const SBTreeVelocityCache& vc,
    const SpatialVec& H_col, int selfMode, InverseDynamicsDerivativesWorkspace& w,
    int col, Matrix& dTauDQ, Matrix& dTauDU) const
{
    const MobilizedBodyIndex k = this->getNodeNum();
    const MobilizedBodyIndex p = this->parent->getNodeNum();
    const int                ux0 = this->getUIndex();

    const Vec3& w_i = H_col[0];
    const Vec3& v_i = H_col[1];
    const PhiMatrix&      Phi = this->getPhi(pc);
    const SpatialInertia& M   = this->getMk_G(pc);
    const HType&          H   = this->getH(pc);

    const Vec3& wk = this->getV_GB(vc)[0];
    const Vec3& dw = w.dV[k][0];
    const Vec3& ew = w.eV[k][0];

    //--------------------------------------------------------------------------
    // Inertia sensitivity, applied not formed.                    [Jain 18.29c]
    // dM = W M - M W, hence dM y = W(M y) - M(W y), with M y precomputed once
    // per body as MA = M A_GB and Mw = M (w,0).
    //--------------------------------------------------------------------------
    const SpatialVec& MA = w.MA[k];
    const SpatialVec& Mw = w.Mw[k];
    const SpatialVec& A  = w.A_GB[k];
    const SpatialVec dMA = SpatialVec(w_i % MA[0], w_i % MA[1])
                         - M * SpatialVec(w_i % A[0], w_i % A[1]);
    const SpatialVec dMw = SpatialVec(w_i % Mw[0], w_i % Mw[1])
                         - M * SpatialVec(w_i % wk, Vec3(0));

    //--------------------------------------------------------------------------
    // Gyroscopic-force sensitivity. b = W(w)[M (w,0)], so
    //   db = W(dw)[M(w,0)] + W(w)[dM(w,0) + M(dw,0)]
    // and the u side drops the dM term.
    //--------------------------------------------------------------------------
    const SpatialVec Mdw = M * SpatialVec(dw, Vec3(0));
    const SpatialVec Mew = M * SpatialVec(ew, Vec3(0));
    const SpatialVec db = SpatialVec(dw % Mw[0], dw % Mw[1])
                        + SpatialVec(wk % (dMw[0]+Mdw[0]),
                                     wk % (dMw[1]+Mdw[1]));
    const SpatialVec eb = SpatialVec(ew % Mw[0], ew % Mw[1])
                        + SpatialVec(wk % Mew[0], wk % Mew[1]);

    //--------------------------------------------------------------------------
    // Transmitted-force sensitivity, then this body's Jacobian rows.
    //   dF += dM A + M dA + db ;  eF += M eA + eb
    //   dtau = [dH] . F + H . dF ;  etau = H . eF              [Singh J3]
    // The [dH] . F term is J3's residue and is nonzero only for multi-DoF
    // joints; there is no [eH] . F counterpart since H is q-only.
    //--------------------------------------------------------------------------
    w.dF[k] += dMA + M * w.dA[k] + db;
    w.eF[k] += M * w.eA[k] + eb;

    const SpatialVec& Fk = w.F[k];
    for (int e=0; e < dof; ++e) {
        dTauDQ(ux0+e, col) = dot(w.dH[ux0+e], Fk) + dot(H(e), w.dF[k]);
        dTauDU(ux0+e, col) = dot(H(e), w.eF[k]);
    }

    //--------------------------------------------------------------------------
    // Push to the parent. On a force the dPhi action runs the other way:
    //   dPhi F = W(Phi F) - Phi(W F), or (v_i x f, 0) when i == k.
    //--------------------------------------------------------------------------
    const SpatialVec PhiF = Phi * Fk;
    const SpatialVec dPhiF = (selfMode >= 0)
        ? SpatialVec(v_i % Fk[1], Vec3(0))
        : SpatialVec(w_i % PhiF[0], w_i % PhiF[1])
          - Phi * SpatialVec(w_i % Fk[0], w_i % Fk[1]);
    w.dF[p] += dPhiF + Phi * w.dF[k];
    w.eF[p] += Phi * w.eF[k];
}

// Inward, above the perturbed body. Outside the perturbed subtree every body
// term vanishes and so does dPhi, since dPhi(k,c) requires the perturbed joint
// to be an ancestor of c; only the transmitted force keeps propagating, and dH
// is zero so dtau loses its first term.
template<int dof, bool noR_FM> void
RigidBodyNodeSpec<dof, noR_FM>::calcInverseDynamicsDerivativesInwardAbove
   (const SBTreePositionCache& pc, InverseDynamicsDerivativesWorkspace& w,
    int col, Matrix& dTauDQ, Matrix& dTauDU) const
{
    const MobilizedBodyIndex k = this->getNodeNum();
    const MobilizedBodyIndex p = this->parent->getNodeNum();
    const int                ux0 = this->getUIndex();
    const HType&             H   = this->getH(pc);
    const PhiMatrix&         Phi = this->getPhi(pc);

    for (int e=0; e < dof; ++e) {
        dTauDQ(ux0+e, col) = dot(H(e), w.dF[k]);
        dTauDU(ux0+e, col) = dot(H(e), w.eF[k]);
    }
    w.dF[p] += Phi * w.dF[k];
    w.eF[p] += Phi * w.eF[k];
}


    ////////////////////
    // INSTANTIATIONS //
    ////////////////////

template class RigidBodyNodeSpec<1, false>;
template class RigidBodyNodeSpec<2, false>;
template class RigidBodyNodeSpec<3, false>;
template class RigidBodyNodeSpec<4, false>;
template class RigidBodyNodeSpec<5, false>;
template class RigidBodyNodeSpec<6, false>;
template class RigidBodyNodeSpec<1, true>;
template class RigidBodyNodeSpec<3, true>;
