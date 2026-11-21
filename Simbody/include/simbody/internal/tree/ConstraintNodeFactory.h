#ifndef SimTK_SIMBODY_CONSTRAINT_NODE_FACTORY_H_
#define SimTK_SIMBODY_CONSTRAINT_NODE_FACTORY_H_

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
A visitor used to create the constraints of a SimbodyMatterSubsystem for a
scalar type other than Real, when a ScalarState<T> is realized through
Stage::Instance. This works like RigidBodyNodeFactory:
SimbodyMatterSubsystem::createConstraintNodes() visits each enabled
Constraint, in order of ConstraintIndex, and the Constraint's implementation
calls the factory method for its type with its parameters (a kernel from
ConstraintKernels.h). A factory for a particular scalar type (see
ConstraintNodeFactory_<T> in MultibodyScalarImpl.h) overrides the methods for
the constraint types it supports; the default implementations throw an
exception. **/

#include "simbody/internal/common.h"
#include "simbody/internal/tree/ConstraintKernels.h"

#include <vector>

namespace SimTK {

/** The q's and u's of one of a Constraint's constrained mobilizers, as
contiguous ranges of the Constraint's constrained q's and u's. **/
struct ConstrainedMobilizerNodeInfo {
    MobilizedBodyIndex mobod;
    int firstConstrainedQ = 0, nq = 0;
    int firstConstrainedU = 0, nu = 0;
};

/** The information about one enabled Constraint needed to create its T-typed
counterpart, from the Real State. **/
struct ConstraintNodeInfo {
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
};

/** Abstract visitor that creates T-typed constraints; see the file comment.
The ConstrainedBodyIndex arguments say which constrained bodies the kernel's
bodies are. **/
class ConstraintNodeFactory {
public:
    virtual ~ConstraintNodeFactory() = default;

    virtual void createBall(const ConstraintNodeInfo& i,
                            ConstrainedBodyIndex B1, ConstrainedBodyIndex B2,
                            const BallConstraintKernel&)
    {   unsupported("Ball", i); }
    virtual void createWeld(const ConstraintNodeInfo& i,
                            ConstrainedBodyIndex B, ConstrainedBodyIndex F,
                            const WeldConstraintKernel&)
    {   unsupported("Weld", i); }
    virtual void createPointInPlane(const ConstraintNodeInfo& i,
                                    ConstrainedBodyIndex planeBody,
                                    ConstrainedBodyIndex followerBody,
                                    const PointInPlaneConstraintKernel&)
    {   unsupported("PointInPlane", i); }
    virtual void createPointOnLine(const ConstraintNodeInfo& i,
                                   ConstrainedBodyIndex lineBody,
                                   ConstrainedBodyIndex followerBody,
                                   const PointOnLineConstraintKernel&)
    {   unsupported("PointOnLine", i); }
    virtual void createConstantAngle(const ConstraintNodeInfo& i,
                                     ConstrainedBodyIndex B,
                                     ConstrainedBodyIndex F,
                                     const ConstantAngleConstraintKernel&)
    {   unsupported("ConstantAngle", i); }
    virtual void createRod(const ConstraintNodeInfo& i,
                           ConstrainedBodyIndex F, ConstrainedBodyIndex B,
                           const RodConstraintKernel&)
    {   unsupported("Rod", i); }
    virtual void createNoSlip1D(const ConstraintNodeInfo& i,
                                ConstrainedBodyIndex caseBody,
                                ConstrainedBodyIndex movingBody0,
                                ConstrainedBodyIndex movingBody1,
                                const NoSlip1DConstraintKernel&)
    {   unsupported("NoSlip1D", i); }
    /** A Constraint::CoordinateCoupler: \a function of the constrained q's
    given by \a arguments is zero. The Function is owned by the Constraint,
    which outlives the T-typed constraint. **/
    virtual void createCoordinateCoupler(const ConstraintNodeInfo& i,
        const Function& function,
        const std::vector<ConstrainedQIndex>& arguments)
    {   unsupported("CoordinateCoupler", i); }
    /** Called for the other Constraint types. **/
    virtual void createOther(const ConstraintNodeInfo& i)
    {   unsupported("this", i); }

protected:
    /** Throw an exception saying that this factory doesn't support the given
    constraint type. **/
    void unsupported(const char* constraint,
                     const ConstraintNodeInfo& info) const {
        SimTK_ERRCHK2_ALWAYS(false, "ConstraintNodeFactory",
            "Constraint %d is enabled, and %s constraint type is not yet "
            "supported for this scalar type.", (int)info.index, constraint);
    }
};

} // namespace SimTK

#endif // SimTK_SIMBODY_CONSTRAINT_NODE_FACTORY_H_
