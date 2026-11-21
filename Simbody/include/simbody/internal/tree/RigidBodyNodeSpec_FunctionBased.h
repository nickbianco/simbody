#ifndef SimTK_SIMBODY_RIGID_BODY_NODE_SPEC_FUNCTION_BASED_H_
#define SimTK_SIMBODY_RIGID_BODY_NODE_SPEC_FUNCTION_BASED_H_

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

/**@file
 * Define the RigidBodyNode that implements a FunctionBased mobilizer for a
 * scalar type other than Real.
 */

#include "RigidBodyNode.h"
#include "RigidBodyNodeSpec.h"
#include "MobilizerKernels.h"
#include "SimTKcommon/internal/FunctionEvaluator.h"

#include <vector>

namespace SimTK {

    // FUNCTION BASED //

// A MobilizedBody::FunctionBased mobilizer, for a scalar type T other than
// Real. (For Real, it is a Custom mobilizer, whose FunctionBasedImpl
// implementation computes the same things from the State.) The q's are the
// arguments of six Functions, the spatial coordinates, from which
// FunctionBasedMobilizerKernel (shared with FunctionBasedImpl) computes
// X_FM, H_FM, and HDot_FM; see MobilizerKernels.h. The generalized speeds
// are the qdots. The Functions are evaluated with FunctionEvaluator<T>, and
// need first and second derivatives (for H_FM and HDot_FM).
template <class T, int nu>
class RBNodeFunctionBased_ : public RigidBodyNodeSpec_<T, nu, false> {
public:
SimTK_RBNODE_SCALAR_TYPEDEFS(T);
typedef RigidBodyNode_<T> Base;
typedef RigidBodyNodeSpec_<T, nu, false> Spec;
typedef typename Spec::HType HType;
typedef FunctionEvaluator<T> Evaluator;
static_assert(hasFunctionEvaluator<T>,
              "FunctionEvaluator<T> must be specialized.");

const char* type() const override { return "functionBased"; }

// The Functions are owned by the mobilizer, which outlives this node.
RBNodeFunctionBased_(const MassProperties_<T>&             mProps_B,
                     const std::vector<const Function*>&   functions,
                     const std::vector<std::vector<int>>&  coordIndices,
                     const FunctionBasedMobilizerKernel&   kernel,
                     bool                                  isReversed,
                     UIndex&                               nextUSlot,
                     USquaredIndex&                        nextUSqSlot,
                     QIndex&                               nextQSlot)
:   Spec(mProps_B,nextUSlot,nextUSqSlot,nextQSlot,
         Base::QDotIsAlwaysTheSameAsU, Base::QuaternionIsNeverUsed,
         isReversed),
    functions(functions), coordIndices(coordIndices), kernel(kernel)
{
    assert(functions.size() == 6 && coordIndices.size() == 6);
    this->updateSlots(nextUSlot,nextUSqSlot,nextQSlot);
}

// We're not going to attempt to cache any q precalculations.
int calcQPoolSize(const SBModelVars&) const override {return 0;}

void performQPrecalculations(const SBStateDigest_<T>& sbs,
                             const RealP* q, int nq,
                             RealP* qCache,  int nQCache,
                             RealP* qErr,    int nQErr) const override
{
    assert(q && nq==nu && nQCache==0 && nQErr==0);
}

void calcX_FM(const SBStateDigest_<T>& sbs,
              const RealP* q,      int nq,
              const RealP* qCache, int nQCache,
              TransformP&  X_FM) const override
{
    assert(q && nq==nu && nQCache==0);
    X_FM = kernel.calcX_FM(calcSpatialCoords(q));
}

void calcAcrossJointVelocityJacobian(
    const SBStateDigest_<T>& sbs,
    HType&                   H_FM) const override
{
    const RealP* q = sbs.getQ() + this->getQIndex();
    kernel.calcH(calcSpatialCoords(q), calcFq(q), H_FM);
}

void calcAcrossJointVelocityJacobianDot(
    const SBStateDigest_<T>& sbs,
    HType&                   HDot_FM) const override
{
    const RealP* q = sbs.getQ() + this->getQIndex();
    const RealP* u = sbs.getU() + this->getUIndex();
    kernel.calcHDot(calcSpatialCoords(q), calcFq(q), calcFqDot(q, u),
                    Vec<nu,T>::getAs(u), HDot_FM);
}

private:
// The arguments of Function i, from the mobilizer's q's.
std::vector<T> getArguments(int i, const RealP* q) const {
    std::vector<T> x(coordIndices[i].size());
    for (int k=0; k < (int)x.size(); ++k) x[k] = q[coordIndices[i][k]];
    return x;
}

// The spatial coordinates; a constant Function (of no q's) is evaluated as
// a Real.
Vec<6,T> calcSpatialCoords(const RealP* q) const {
    Vec<6,T> s;
    for (int i=0; i < 6; ++i)
        s[i] = coordIndices[i].empty()
            ? T(functions[i]->calcValue(Vector(0)))
            : Evaluator::calcValue(*functions[i], getArguments(i, q));
    return s;
}

// The partials of the spatial coordinates with respect to the q's.
Mat<6,nu,T> calcFq(const RealP* q) const {
    Mat<6,nu,T> Fq(T(0));
    std::vector<int> deriv(1);
    for (int i=0; i < 6; ++i) {
        const int nc = (int)coordIndices[i].size();
        if (nc == 0) continue;
        const std::vector<T> x = getArguments(i, q);
        for (int j=0; j < nc; ++j) {
            deriv[0] = j;
            Fq(i, coordIndices[i][j]) =
                Evaluator::calcDerivative(*functions[i], deriv, x);
        }
    }
    return Fq;
}

// The time derivatives of the partials, the second partials times u.
Mat<6,nu,T> calcFqDot(const RealP* q, const RealP* u) const {
    Mat<6,nu,T> FqDot(T(0));
    std::vector<int> derivs(2);
    for (int i=0; i < 6; ++i) {
        const int nc = (int)coordIndices[i].size();
        if (nc == 0) continue;
        const std::vector<T> x = getArguments(i, q);
        for (int j=0; j < nc; ++j) {
            derivs[0] = j;
            for (int k=0; k < nc; ++k) {
                derivs[1] = k;
                FqDot(i, coordIndices[i][j]) +=
                    Evaluator::calcDerivative(*functions[i], derivs, x)
                    * u[coordIndices[i][k]];
            }
        }
    }
    return FqDot;
}

const std::vector<const Function*>  functions;
const std::vector<std::vector<int>> coordIndices;
const FunctionBasedMobilizerKernel  kernel;
};

} // namespace SimTK

#endif // SimTK_SIMBODY_RIGID_BODY_NODE_SPEC_FUNCTION_BASED_H_
