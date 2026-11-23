#ifndef SimTK_SIMBODY_MULTIBODY_SCALAR_IMPL_H_
#define SimTK_SIMBODY_MULTIBODY_SCALAR_IMPL_H_

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
Definitions of the member templates of MultibodySystem, SimbodyMatterSubsystem,
and MobilizedBody that evaluate a system with a ScalarState<T> (see
SimTKcommon/internal/ScalarState.h), for any scalar type T other than Real.
This is header-only. Include it (after the header that defines the traits for
T; see SimTKcommon/internal/RealScalarType.h) in each translation unit that
uses these templates, or include it in one translation unit that explicitly
instantiates them for T with SimTK_INSTANTIATE_MULTIBODY_SCALAR(T). For
example:
@code
    #include "MyScalarTraits.h" // defines the traits for MyScalar
    #include "simbody/internal/tree/MultibodyScalarImpl.h"
    SimTK_INSTANTIATE_MULTIBODY_SCALAR(MyScalar);
@endcode

This header is the opt-in entry point for scalar evaluation: it binds a
ScalarState<T> to the matter subsystem's T-typed data and defines the public
member templates. The matter subsystem keeps its T-typed quantities in a cache
entry of the ScalarState<T>, as it does in a State: a MatterScalarCache<T>,
allocated when the ScalarState<T> is realized through Stage::Instance. Its
MatterScalarStructure<T>, shared by copies, holds the T-typed RigidBodyNodes
(created by a RigidBodyNodeFactory_<T>) and the descriptions of the enabled
constraints (ConstraintEquations.h), which SimbodyMatterSubsystemRep
creates from the Real State; each copy has its own tree cache entries. The
numerical tree sweeps are shared with SimbodyMatterSubsystemRep
(TreeSweeps.h), or mirror its ones with an SBScalarStateDigest<T>.

Note that the headers in this directory are Simbody implementation details
that are templatized on the scalar type (in namespace SimTK), so include them
only where needed. **/

#include "simbody/internal/common.h"
#include "simbody/internal/MultibodySystem.h"
#include "simbody/internal/SimbodyMatterSubsystem.h"
#include "simbody/internal/MobilizedBody.h"

#include "simbody/internal/tree/SimbodyTreeState.h"
#include "simbody/internal/tree/RigidBodyNode.h"
#include "simbody/internal/tree/RigidBodyNodeSpec.h"
#include "simbody/internal/tree/RigidBodyNodeSpecDefs.h"
#include "simbody/internal/tree/RigidBodyNode_Weld.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_Pin.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_Slider.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_Planar.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_Gimbal.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_Universal.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_Cylinder.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_Screw.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_Translation.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_PolarCoords.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_SphericalCoords.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_CantileverFreeBeam.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_Ball.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_Ellipsoid.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_LineOrientation.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_Free.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_FreeLine.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_Bushing.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_FunctionBased.h"
#include "simbody/internal/tree/RigidBodyNodeFactory.h"
#include "simbody/internal/tree/TreeSweeps.h"
#include "simbody/internal/tree/ConstraintEquations.h"
#include "SimTKcommon/internal/FunctionEvaluator.h"

#include <memory>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace SimTK {

//==============================================================================
//                          MATTER SCALAR STRUCTURE
//==============================================================================
/* The constant part of the matter subsystem's cache entry in a ScalarState<T>,
shared by copies: the Model and Instance stage information of the (Real)
system, with T-typed copies of the Instance quantities, and the T-typed
RigidBodyNodes. */
template <class T>
class MatterScalarStructure {
public:
    const SimbodyMatterSubsystem*   matter = nullptr;

    SBTopologyCache         topo;
    SBModelVars             mv;
    SBModelCache            mc;
    SBInstanceCache         icReal; // only used for sizing cache entries
    SBInstanceVars_<T>      iv;
    SBInstanceCache_<T>     ic;
    int                     nq = 0, nu = 0;

    // T-typed nodes, indexed by MobilizedBodyIndex, and organized by level
    // in the tree (Ground is at level 0).
    std::vector<std::unique_ptr<RigidBodyNode_<T>>>    nodes;
    std::vector<std::vector<const RigidBodyNode_<T>*>> levels;

    // The enabled constraints, in order of ConstraintIndex (their equations
    // are evaluated with T), and the total numbers of holonomic,
    // nonholonomic, and acceleration-only equations.
    std::vector<ConstraintDescription> constraints;
    int mHolo = 0, mNonholo = 0, mAccOnly = 0;
    // Where this subsystem's constraint errors start in the system's qErr and
    // uErr.
    int qErrStart = 0, uErrStart = 0;

    int getNumBodies() const {return (int)nodes.size();}
};

//==============================================================================
//                         RIGID BODY NODE FACTORY
//==============================================================================
/* Creates the T-typed RigidBodyNodes for the mobilizer types that support
scalar type T, and connects them. See RigidBodyNodeFactory.h. To support
another mobilizer, override its create method here. */
template <class T>
class RigidBodyNodeFactory_ : public RigidBodyNodeFactory {
public:
    explicit RigidBodyNodeFactory_(MatterScalarStructure<T>& s) : s(s) {}

    void beginTree(const SBTopologyCache&   topology,
                   const SBModelVars&       modelVars,
                   const SBModelCache&      modelCache,
                   const SBInstanceVars&    instanceVars,
                   const SBInstanceCache&   instanceCache) override {
        s.topo   = topology;
        s.mv     = modelVars;
        s.mc     = modelCache;
        s.icReal = instanceCache;
        s.iv     = SBInstanceVars_<T>(instanceVars);
        s.ic     = SBInstanceCache_<T>(instanceCache);
        s.nodes.clear();
        s.nodes.resize(topology.nBodies);
        s.levels.clear();
    }

    void createGround(const RigidBodyNodeInfo& info) override
    {   add(info, new RBGroundBody_<T>()); }
    void createWeld(const RigidBodyNodeInfo& info) override {
        add(info, new RBNodeWeld_<T>(sbCast<T>(info.massProperties),
                                     info.uIndex, info.uSqIndex, info.qIndex));
    }
    // Each of these mobilizers' nodes is constructed from its mass properties,
    // the mobilizer's parameters (if any), and its direction; see addFree().
    void createPin(const RigidBodyNodeInfo& i) override
    {   addFree<RBNodeTorsion_<T>>(i); }
    void createSlider(const RigidBodyNodeInfo& i) override
    {   addFree<RBNodeSlider_<T>>(i); }
    void createPlanar(const RigidBodyNodeInfo& i) override
    {   addFree<RBNodePlanar_<T>>(i); }
    void createGimbal(const RigidBodyNodeInfo& i) override
    {   addFree<RBNodeGimbal_<T>>(i); }
    void createUniversal(const RigidBodyNodeInfo& i) override
    {   addFree<RBNodeUJoint_<T>>(i); }
    void createCylinder(const RigidBodyNodeInfo& i) override
    {   addFree<RBNodeCylinder_<T>>(i); }
    void createTranslation(const RigidBodyNodeInfo& i) override
    {   addFree<RBNodeTranslate_<T>>(i); }
    void createBendStretch(const RigidBodyNodeInfo& i) override
    {   addFree<RBNodeBendStretch_<T>>(i); }
    void createBall(const RigidBodyNodeInfo& i) override
    {   addFree<RBNodeBall_<T>>(i); }
    void createLineOrientation(const RigidBodyNodeInfo& i) override
    {   addFree<RBNodeLineOrientation_<T>>(i); }
    void createFree(const RigidBodyNodeInfo& i) override
    {   addFree<RBNodeFree_<T>>(i); }
    void createFreeLine(const RigidBodyNodeInfo& i) override
    {   addFree<RBNodeFreeLine_<T>>(i); }
    void createBushing(const RigidBodyNodeInfo& i) override
    {   addFree<RBNodeBushing_<T>>(i); }
    void createScrew(const RigidBodyNodeInfo& i, Real pitch) override
    {   addFree<RBNodeScrew_<T>>(i, pitch); }
    void createCantileverFreeBeam(const RigidBodyNodeInfo& i,
                                  Real length) override
    {   addFree<RBNodeCantileverFreeBeam_<T>>(i, length); }
    void createSphericalCoords(const RigidBodyNodeInfo& i,
        Real azimuthOffset, bool azimuthNegated, Real zenithOffset,
        bool zenithNegated, CoordinateAxis translationAxis,
        bool translationNegated) override {
        addFree<RBNodeSphericalCoords_<T>>(i, azimuthOffset, azimuthNegated,
            zenithOffset, zenithNegated, translationAxis, translationNegated);
    }
    // The radii are an Instance-stage variable, from the Real State.
    void createEllipsoid(const RigidBodyNodeInfo& i, const Vec3& radii,
                         int mirrorAxis) override
    {   addFree<RBNodeEllipsoid_<T>>(i, radii, mirrorAxis); }
    // The node is templatized on the number of mobilities, and evaluates the
    // Functions with FunctionEvaluator<T>.
    void createFunctionBased(const RigidBodyNodeInfo& i, int nu,
        const std::vector<const Function*>& functions,
        const std::vector<std::vector<int>>& coordIndices,
        const FunctionBasedMobilizerKernel& kernel) override {
        if constexpr (hasFunctionEvaluator<T>) {
            switch (nu) {
#define SimTK_FUNCTION_BASED_CASE(n)                                           \
            case n: addFree<RBNodeFunctionBased_<T,n>>(i, functions,           \
                                                     coordIndices, kernel);    \
                    break;
            SimTK_FUNCTION_BASED_CASE(1) SimTK_FUNCTION_BASED_CASE(2)
            SimTK_FUNCTION_BASED_CASE(3) SimTK_FUNCTION_BASED_CASE(4)
            SimTK_FUNCTION_BASED_CASE(5) SimTK_FUNCTION_BASED_CASE(6)
#undef SimTK_FUNCTION_BASED_CASE
            default: assert(!"FunctionBased mobilizers have 1 to 6 mobilities");
            }
        } else {
            SimTK_ERRCHK1_ALWAYS(false, "RigidBodyNodeFactory_::create()",
                "Mobilized body %d is FunctionBased, which needs "
                "SimTK::FunctionEvaluator to be specialized for this scalar "
                "type (see SimTKcommon/internal/FunctionEvaluator.h).",
                (int)i.index);
        }
    }

private:
    // Prescribed motion is not yet supported.
    void requireFree(const RigidBodyNodeInfo& info) const {
        const SBInstancePerMobodInfo& mi = 
            s.icReal.getMobodInstanceInfo(info.index);
        SimTK_ERRCHK1_ALWAYS(mi.qMethod == Motion::Free 
            && mi.uMethod == Motion::Free && mi.udotMethod == Motion::Free,
            "RigidBodyNodeFactory_::create()",
            "MobilizedBody %d has prescribed motion, which is not yet "
            "supported for this scalar type.", (int)info.index);
    }

    // Create a node of type Node for a mobilizer without prescribed motion,
    // from the mass properties, the mobilizer's parameters args, and the
    // direction, taking the next u, u-squared, and q slots, and connect it.
    template <class Node, class... Args>
    void addFree(const RigidBodyNodeInfo& info, const Args&... args) {
        requireFree(info);
        UIndex u = info.uIndex; USquaredIndex usq = info.uSqIndex;
        QIndex q = info.qIndex;
        add(info, new Node(sbCast<T>(info.massProperties), args...,
                           info.isReversed, u, usq, q));
    }

    // Take ownership of the node and connect it to its parent, which has
    // already been created. This is what MobilizedBodyImpl::realizeTopology()
    // does for Real.
    void add(const RigidBodyNodeInfo& info, RigidBodyNode_<T>* node) {
        s.nodes[info.index].reset(node);
        node->setNodeNum(info.index);
        node->setLevel(info.level);
        if (info.parent.isValid()) {
            RigidBodyNode_<T>* parent = s.nodes[info.parent].get();
            assert(parent);
            parent->addChild(node);
            node->setParent(parent);
        }
        if ((int)s.levels.size() <= info.level) s.levels.resize(info.level+1);
        s.levels[info.level].push_back(node);
    }

    MatterScalarStructure<T>& s;
};

//==============================================================================
//                            MATTER SCALAR CACHE
//==============================================================================
/* The matter subsystem's cache entry in a ScalarState<T>. */
template <class T>
class MatterScalarCache {
public:
    std::shared_ptr<const MatterScalarStructure<T>> structure;

    std::vector<T>                      qdot;
    SBTreePositionCache_<T>             tpc;
    SBTreeVelocityCache_<T>             tvc;
    SBArticulatedBodyInertiaCache_<T>   abc;
    SBArticulatedBodyVelocityCache_<T>  abvc;
    SBDynamicsCache_<T>                 dc;
    SBTreeAccelerationCache_<T>         tac;
    // Articulated body inertias aren't a stage; they are computed on demand
    // and are valid until the next position kinematics.
    bool                                abiRealized = false;
};

//==============================================================================
//                             MATTER SCALAR REP
//==============================================================================
/* The matter subsystem's computations for a ScalarState<T>, using its cache
entry there. This is created as needed by the member templates below. */
template <class T>
class MatterScalarRep {
public:
    typedef SpatialVec_<T> SpatialVecT;

    // The cache entry is always the matter subsystem's first one.
    static CacheEntryIndex getCacheEntryIndex() {return CacheEntryIndex(0);}

    // Create the matter subsystem's cache entry in the ScalarState if it
    // doesn't exist yet. This is Stage::Instance for the matter subsystem.
    static void realizeInstance(const SimbodyMatterSubsystem& matter,
                                const ScalarState<T>& state);

    // Throws if the ScalarState hasn't been realized through Stage::Instance or
    // belongs to a different matter subsystem.
    MatterScalarRep(const SimbodyMatterSubsystem& matter,
                    const ScalarState<T>& state, const char* methodName);

    const MatterScalarCache<T>& getCache() const {return c;}

    // Throw if the ScalarState hasn't been realized through the given stage.
    void requireStage(Stage required, const char* methodName) const {
        const Stage stage = state.getSystemStage();
        SimTK_ERRCHK3_ALWAYS(stage >= required, methodName,
            "Expected ScalarState to be realized through Stage::%s but it has "
            "only been realized through Stage::%s (in %s).",
            required.getName().c_str(), stage.getName().c_str(), methodName);
    }

    void realizePosition() const;
    void realizeVelocity() const;
    void realizeArticulatedBodyInertias() const;

    T    calcKineticEnergy() const;
    void multiplyBySystemJacobian(const std::vector<T>& u,
                                  std::vector<SpatialVecT>& Ju) const;
    void multiplyBySystemJacobianTranspose(const std::vector<SpatialVecT>& F_G,
                                           std::vector<T>& f) const;
    void multiplyByM(const std::vector<T>& a, std::vector<T>& Ma) const;
    void multiplyByMInv(const std::vector<T>& v, std::vector<T>& MinvV) const;
    void calcTreeAccelerations(const std::vector<T>& mobilityForces,
                               const std::vector<SpatialVecT>& bodyForces,
                               std::vector<T>& udot,
                               std::vector<SpatialVecT>& A_GB) const;
    void calcTreeResidualForces(const std::vector<T>& mobilityForces,
                                const std::vector<SpatialVecT>& bodyForces,
                                const std::vector<T>& knownUdot,
                                std::vector<T>& residual) const;
    void calcConstraintAccelerationErrors(const std::vector<T>& knownUDot,
                                          std::vector<T>& pvaerr) const;
    void calcConstraintForcesFromMultipliers
       (const std::vector<T>& multipliers, std::vector<SpatialVecT>& bodyForces,
        std::vector<T>& mobilityForces, const char* where) const;
    void multiplyByGTranspose(const std::vector<T>& lambda,
                              std::vector<T>& f) const;

private:
    static MatterScalarCache<T>& getMatterCache(
        const SimbodyMatterSubsystem& matter, const ScalarState<T>& state,
        const char* methodName);
    int getNumBodies() const {return s.getNumBodies();}
    int getNumMultipliers() const {return s.mHolo + s.mNonholo + s.mAccOnly;}
    SBScalarStateDigest<T> makeDigest() const;

    // The constrained bodies' poses, velocities, and accelerations (given all
    // the bodies' accelerations in Ground) in the Ancestor frame of the
    // constraint; see ConstraintImpl::getBodyTransformFromState() etc.
    void findConstrainedBodyTransforms(const ConstraintDescription& ci,
        std::vector<Transform_<T>>& X_AB) const;
    void findConstrainedBodyVelocities(const ConstraintDescription& ci,
        std::vector<SpatialVecT>& V_AB) const;
    void findConstrainedBodyAccelerations(const ConstraintDescription& ci,
        const std::vector<SpatialVecT>& A_GB,
        std::vector<SpatialVecT>& A_AB) const;
    // A constraint's constrained q's, qdots, or qdotdots, from the
    // subsystem's (indexed by QIndex).
    static void findConstrainedQ(const ConstraintDescription& ci, const T* all,
                                 std::vector<T>& constrained);
    void realizeConstraintPositionErrors() const;
    void realizeConstraintVelocityErrors() const;

    // Evaluate a constraint's equations (with T): call f with the equations
    // object; see ConstraintEquations.h.
    template <class F>
    static void visit(const ConstraintDescription& ci, F&& f)
    {   std::visit(std::forward<F>(f), ci.equations); }

    // Return in, or a copy of it if it's the same vector as out (an input
    // that's also the output must be read before the output is changed).
    template <class V>
    static const std::vector<V>& unaliased(const std::vector<V>& in,
                                           const std::vector<V>& out,
                                           std::vector<V>& copy) {
        if (&in != &out) return in;
        copy = in;
        return copy;
    }

    // Return v, or a zero vector of length n if v is empty.
    template <class V>
    const std::vector<V>& orZero(const std::vector<V>& v, int n, 
                                 std::vector<V>& zero, const V& z,
                                 const char* what, const char* where) const {
        if (v.empty()) {zero.assign(n, z); return zero;}
        SimTK_ERRCHK3_ALWAYS((int)v.size() == n, where,
            "Expected %d %s but got %d.", n, what, (int)v.size());
        return v;
    }

    const ScalarState<T>&           state;
    const SubsystemIndex            ix;
    MatterScalarCache<T>&           c;
    const MatterScalarStructure<T>& s;
};

//------------------------------------------------------------------------------
//                          MATTER SCALAR REP METHODS
//------------------------------------------------------------------------------
template <class T> void 
MatterScalarRep<T>::realizeInstance(const SimbodyMatterSubsystem& matter,
                                    const ScalarState<T>& state) {
    const SubsystemIndex ix = matter.getMySubsystemIndex();
    if (state.getNumCacheEntries(ix) > 0) return; // already created
    const char* where = "MultibodySystem::realize(ScalarState)";
    SimTK_ERRCHK_ALWAYS(matter.getNumParticles() == 0, where,
        "Particles are not yet supported for this scalar type.");

    auto s = std::make_shared<MatterScalarStructure<T>>();
    s->matter = &matter;
    RigidBodyNodeFactory_<T> factory(*s);
    matter.createRigidBodyNodes(factory, state.getRealState());
    s->nq = state.getNQ(ix);
    s->nu = state.getNU(ix);
    // The enabled constraints (disabled ones, such as OpenSim's coordinate
    // locks, have no effect here). A CoordinateCoupler's Function is
    // evaluated with FunctionEvaluator<T>.
    matter.calcConstraintDescriptions(state.getRealState(), s->constraints);
    if constexpr (!hasFunctionEvaluator<T>) {
        for (const ConstraintDescription& ci : s->constraints)
            SimTK_ERRCHK1_ALWAYS(!std::holds_alternative<
                CoordinateCouplerConstraintEquations>(ci.equations), where,
                "Constraint %d is a CoordinateCoupler, which needs "
                "SimTK::FunctionEvaluator to be specialized for this scalar "
                "type (see SimTKcommon/internal/FunctionEvaluator.h).",
                (int)ci.index);
    }
    s->mHolo    = s->icReal.totalNHolonomicConstraintEquationsInUse;
    s->mNonholo = s->icReal.totalNNonholonomicConstraintEquationsInUse;
    s->mAccOnly = s->icReal.totalNAccelerationOnlyConstraintEquationsInUse;
    s->qErrStart = state.getQErrStart(ix);
    s->uErrStart = state.getUErrStart(ix);

    auto* value = new Value<MatterScalarCache<T>>();
    MatterScalarCache<T>& c = value->upd();
    c.structure = s;
    c.qdot.assign(s->nq, T(0));
    c.tpc.allocate(s->topo, s->mc, s->icReal);
    c.tvc.allocate(s->topo, s->mc, s->icReal);
    c.abc.allocate(s->topo, s->mc, s->icReal);
    c.abvc.allocate(s->topo, s->mc, s->icReal);
    c.dc.allocate(s->topo, s->mc, s->icReal);
    c.tac.allocate(s->topo, s->mc, s->icReal);
    const CacheEntryIndex index = 
        state.allocateCacheEntry(ix, Stage::Instance, value);
    assert(index == getCacheEntryIndex()); (void)index;

    // Give the nodes a chance to initialize cache entries that never change.
    const MatterScalarRep<T> rep(matter, state, where);
    const SBScalarStateDigest<T> sbs = rep.makeDigest();
    for (const auto& level : s->levels)
        for (const RigidBodyNode_<T>* node : level)
            node->realizeInstance(sbs);
}

template <class T>
MatterScalarRep<T>::MatterScalarRep(const SimbodyMatterSubsystem& matter,
                                    const ScalarState<T>& state,
                                    const char* methodName)
:   state(state), ix(matter.getMySubsystemIndex()),
    c(getMatterCache(matter, state, methodName)), s(*c.structure) {
    SimTK_ERRCHK_ALWAYS(matter.isSameSubsystem(*s.matter), methodName,
        "This ScalarState belongs to a different matter subsystem.");
}

template <class T> MatterScalarCache<T>& 
MatterScalarRep<T>::getMatterCache(const SimbodyMatterSubsystem& matter,
                                   const ScalarState<T>& state,
                                   const char* methodName) {
    const SubsystemIndex ix = matter.getMySubsystemIndex();
    SimTK_ERRCHK1_ALWAYS(&state.getSystem() == &matter.getSystem(),
        methodName, "This ScalarState belongs to a different system (in %s).",
        methodName);
    SimTK_ERRCHK1_ALWAYS(matter.getSystem().systemTopologyHasBeenRealized()
        && state.getRealState().getSystemTopologyStageVersion()
               == matter.getSystem().getSystemTopologyCacheVersion(),
        methodName, "The system's topology has changed since this ScalarState "
        "was created (in %s).", methodName);
    SimTK_ERRCHK1_ALWAYS(ix < state.getNumSubsystems()
                         && state.getNumCacheEntries(ix) > 0, methodName,
        "This ScalarState has not been realized through Stage::Instance "
        "(in %s).", methodName);
    // Not getCacheEntry(): this is also used while realizing Stage::Instance.
    return Value<MatterScalarCache<T>>::updDowncast(
        state.updCacheEntry(ix, getCacheEntryIndex())).upd();
}

template <class T> SBScalarStateDigest<T>
MatterScalarRep<T>::makeDigest() const {
    SBScalarStateDigest<T> sbs(s.mv, s.mc, s.iv, s.ic);
    // The ScalarState's stage advances after each stage's computations, and
    // never past Velocity.
    sbs.setStage(state.getSystemStage().next());
    sbs.setQ(state.getQ().data() + state.getQStart(ix));
    sbs.setU(state.getU().data() + state.getUStart(ix));
    sbs.setQDot(c.qdot.data());
    // This subsystem's part of the system's qErr; the nodes write their
    // quaternion normalization errors there (after the constraint errors).
    if (state.getNQErr(ix))
        sbs.setQErr(state.updQErr().data() + state.getQErrStart(ix));
    sbs.setTreePositionCache(&c.tpc);
    sbs.setTreeVelocityCache(&c.tvc);
    sbs.setDynamicsCache(&c.dc);
    sbs.setTreeAccelerationCache(&c.tac);
    return sbs;
}

template <class T> void MatterScalarRep<T>::realizePosition() const {
    const SBScalarStateDigest<T> sbs = makeDigest();
    for (const auto& level : s.levels)
        for (const RigidBodyNode_<T>* node : level)
            node->realizePosition(sbs);
    c.abiRealized = false;
    realizeConstraintPositionErrors();
}

template <class T> void MatterScalarRep<T>::realizeVelocity() const {
    const SBScalarStateDigest<T> sbs = makeDigest();
    for (const auto& level : s.levels)
        for (const RigidBodyNode_<T>* node : level)
            node->realizeVelocity(sbs);
    realizeConstraintVelocityErrors();
}

// See SimbodyMatterSubsystemRep::realizeArticulatedBodyInertias(). 
template <class T> void 
MatterScalarRep<T>::realizeArticulatedBodyInertias() const {
    requireStage(Stage::Position, "realizeArticulatedBodyInertias()");
    if (c.abiRealized) return;
    realizeArticulatedBodyInertiasSweep(s.levels, s.ic, c.tpc, c.abc);
    c.abiRealized = true;
}

// See SimbodyMatterSubsystemRep::calcKineticEnergy().
template <class T> T MatterScalarRep<T>::calcKineticEnergy() const {
    requireStage(Stage::Velocity, "calcKineticEnergy()");
    const auto& levels = s.levels;
    T ke(0);
    for (int i=1; i < (int)levels.size(); ++i) // skip Ground
        for (const RigidBodyNode_<T>* node : levels[i])
            ke += node->calcKineticEnergy(c.tpc, c.tvc);
    return ke;
}

// See SimbodyMatterSubsystemRep::multiplyBySystemJacobian().
template <class T> void MatterScalarRep<T>::multiplyBySystemJacobian
   (const std::vector<T>& v, std::vector<SpatialVecT>& Jv) const {
    const char* where = "multiplyBySystemJacobian()";
    requireStage(Stage::Position, where);
    SimTK_ERRCHK2_ALWAYS((int)v.size() == s.nu, where,
        "Expected %d u's but got %d.", s.nu, (int)v.size());
    Jv.resize(getNumBodies());
    multiplyBySystemJacobianSweep(s.levels, c.tpc, v.data(), Jv.data());
}

// See SimbodyMatterSubsystemRep::multiplyBySystemJacobianTranspose().
template <class T> void MatterScalarRep<T>::multiplyBySystemJacobianTranspose
   (const std::vector<SpatialVecT>& X, std::vector<T>& JtX) const {
    const char* where = "multiplyBySystemJacobianTranspose()";
    requireStage(Stage::Position, where);
    SimTK_ERRCHK2_ALWAYS((int)X.size() == getNumBodies(), where,
        "Expected %d body forces but got %d.", getNumBodies(), (int)X.size());
    std::vector<SpatialVecT> zTmp(getNumBodies(),
                                  SpatialVecT(Vec<3,T>(T(0)), Vec<3,T>(T(0))));
    JtX.assign(s.nu, T(0));
    multiplyBySystemJacobianTransposeSweep(s.levels, c.tpc, zTmp.data(),
                                           X.data(), JtX.data());
}

// See SimbodyMatterSubsystemRep::multiplyByM().
template <class T> void MatterScalarRep<T>::multiplyByM
   (const std::vector<T>& a, std::vector<T>& Ma) const {
    const char* where = "multiplyByM()";
    requireStage(Stage::Position, where);
    SimTK_ERRCHK2_ALWAYS((int)a.size() == s.nu, where,
        "Expected a vector of length %d but got %d.", s.nu, 
        (int)a.size());
    std::vector<SpatialVecT> allA_GB(getNumBodies()), allF(getNumBodies());
    // a and Ma may be the same vector: the sweeps read all of a before they
    // write Ma, and they write every element of Ma.
    Ma.resize(s.nu);
    multiplyByMSweeps(s.levels, c.tpc, a.data(), allA_GB.data(), allF.data(),
                      Ma.data());
}

// See SimbodyMatterSubsystemRep::multiplyByMInv().
template <class T> void MatterScalarRep<T>::multiplyByMInv
   (const std::vector<T>& v, std::vector<T>& MinvV) const {
    const char* where = "multiplyByMInv()";
    requireStage(Stage::Position, where);
    SimTK_ERRCHK2_ALWAYS((int)v.size() == s.nu, where,
        "Expected a vector of length %d but got %d.", s.nu, 
        (int)v.size());
    realizeArticulatedBodyInertias();
    const int nb = getNumBodies(), nu = s.nu;
    std::vector<T> eps(nu, T(0));
    std::vector<SpatialVecT> z(nb), zPlus(nb), A_GB(nb);
    // v and MinvV may be the same vector, as for multiplyByM().
    MinvV.resize(nu);
    multiplyByMInvSweeps(s.levels, s.ic, c.tpc, c.abc, v.data(), eps.data(),
                         z.data(), zPlus.data(), A_GB.data(), MinvV.data());
}

// See SimbodyMatterSubsystemRep::calcTreeAccelerations().
template <class T> void MatterScalarRep<T>::calcTreeAccelerations
   (const std::vector<T>&           mobilityForcesIn,
    const std::vector<SpatialVecT>& bodyForcesIn,
    std::vector<T>&                 udot,
    std::vector<SpatialVecT>&       A_GB) const 
{
    const char* where = "calcAccelerationIgnoringConstraints()";
    requireStage(Stage::Velocity, where);
    const int nb = getNumBodies(), nu = s.nu;
    std::vector<T> zeroF, copyF; std::vector<SpatialVecT> zeroBF;
    // The mobility forces may be udot; the body forces are consumed before
    // A_GB is written.
    const std::vector<T>& mobilityForces = orZero(
        unaliased(mobilityForcesIn, udot, copyF), nu, zeroF, T(0),
        "mobility forces", where);
    const std::vector<SpatialVecT>& bodyForces = orZero(bodyForcesIn, nb, 
        zeroBF, SpatialVecT(Vec<3,T>(0), Vec<3,T>(0)), "body forces", where);

    realizeArticulatedBodyInertias();
    const auto& levels = s.levels;
    // Ground's entries are precalculated so start at level 1.
    for (int i=1; i < (int)levels.size(); ++i)
        for (const RigidBodyNode_<T>* node : levels[i])
            node->realizeArticulatedBodyVelocityCache(c.tpc, c.tvc, c.abc,
                                                      c.abvc);

    udot.assign(nu, T(0));
    T* epsilonPtr = nu ? &c.tac.epsilon[0] : nullptr;
    T* tauPtr     = c.tac.presMotionForces.size() 
                        ? &c.tac.presMotionForces[0] : nullptr;
    SpatialVecT* aPtr = &c.tac.bodyAccelerationInGround[0];

    calcTreeAccelerationsSweeps(levels, s.ic, c.tpc, c.tvc, c.dc, c.abc,
        c.abvc, mobilityForces.data(), bodyForces.data(), c.tac.z.begin(),
        c.tac.zPlus.begin(), epsilonPtr, aPtr, udot.data(), tauPtr);

    A_GB.assign(aPtr, aPtr + nb);
}

// See SimbodyMatterSubsystemRep::calcTreeResidualForces().
template <class T> void MatterScalarRep<T>::calcTreeResidualForces
   (const std::vector<T>&           mobilityForcesIn,
    const std::vector<SpatialVecT>& bodyForcesIn,
    const std::vector<T>&           knownUdotIn,
    std::vector<T>&                 residual) const 
{
    const char* where = "calcResidualForceIgnoringConstraints()";
    requireStage(Stage::Velocity, where);
    const int nb = getNumBodies(), nu = s.nu;
    std::vector<T> zeroF, zeroUDot, copyF, copyUDot;
    std::vector<SpatialVecT> zeroBF;
    // The mobility forces or the udots may be the residual.
    const std::vector<T>& mobilityForces = orZero(
        unaliased(mobilityForcesIn, residual, copyF), nu, zeroF, T(0),
        "mobility forces", where);
    const std::vector<SpatialVecT>& bodyForces = orZero(bodyForcesIn, nb, 
        zeroBF, SpatialVecT(Vec<3,T>(0), Vec<3,T>(0)), "body forces", where);
    const std::vector<T>& knownUdot = orZero(
        unaliased(knownUdotIn, residual, copyUDot), nu, zeroUDot, T(0),
        "udots", where);

    std::vector<SpatialVecT> A_GB(nb), allF(nb);
    residual.assign(nu, T(0));
    calcTreeResidualForcesSweeps(s.levels, c.tpc, c.tvc, knownUdot.data(),
        mobilityForces.data(), bodyForces.data(), A_GB.data(), allF.data(),
        residual.data());
}

// See ConstraintImpl::getBodyTransformFromState() and
// calcConstrainedBodyTransformInAncestor().
template <class T> void MatterScalarRep<T>::findConstrainedBodyTransforms
   (const ConstraintDescription& ci, std::vector<Transform_<T>>& X_AB) const {
    const int ncb = (int)ci.constrainedBodies.size();
    X_AB.resize(ncb);
    if (ci.ancestor == GroundIndex) { // X_AB==X_GB
        for (int b=0; b < ncb; ++b)
            X_AB[b] = c.tpc.getX_GB(ci.constrainedBodies[b]);
        return;
    }
    const Transform_<T>& X_GA = c.tpc.getX_GB(ci.ancestor);
    for (int b=0; b < ncb; ++b) {
        const MobilizedBodyIndex mbx = ci.constrainedBodies[b];
        X_AB[b] = mbx == ci.ancestor ? Transform_<T>()
                                     : Transform_<T>(~X_GA*c.tpc.getX_GB(mbx));
    }
}

// See ConstraintImpl::getBodyVelocityFromState() and
// calcConstrainedBodyVelocityInAncestor().
template <class T> void MatterScalarRep<T>::findConstrainedBodyVelocities
   (const ConstraintDescription& ci, std::vector<SpatialVecT>& V_AB) const {
    const int ncb = (int)ci.constrainedBodies.size();
    V_AB.resize(ncb);
    if (ci.ancestor == GroundIndex) { // V_AB==V_GB
        for (int b=0; b < ncb; ++b)
            V_AB[b] = c.tvc.getV_GB(ci.constrainedBodies[b]);
        return;
    }
    const Transform_<T>& X_GA = c.tpc.getX_GB(ci.ancestor);
    const SpatialVecT&   V_GA = c.tvc.getV_GB(ci.ancestor);
    for (int b=0; b < ncb; ++b) {
        const MobilizedBodyIndex mbx = ci.constrainedBodies[b];
        if (mbx == ci.ancestor) {
            V_AB[b] = SpatialVecT(Vec<3,T>(T(0)), Vec<3,T>(T(0)));
            continue;
        }
        const Transform_<T>& X_GB = c.tpc.getX_GB(mbx);
        const SpatialVecT&   V_GB = c.tvc.getV_GB(mbx);
        const Vec<3,T> p_AB_G     = X_GB.p() - X_GA.p();
        const Vec<3,T> p_AB_G_dot = V_GB[1]  - V_GA[1]; // d/dt p taken in G
        // Relative angular velocity of B in A, exp. in G.
        const Vec<3,T> w_AB_G = V_GB[0] - V_GA[0];
        // Time derivative of p in A, exp in G.
        const Vec<3,T> v_AB_G = p_AB_G_dot - V_GA[0] % p_AB_G;
        V_AB[b] = SpatialVecT(~X_GA.R()*w_AB_G, ~X_GA.R()*v_AB_G); // in A
    }
}

// See ConstraintImpl::convertBodyAccelToConstrainedBodyAccel().
template <class T> void MatterScalarRep<T>::findConstrainedBodyAccelerations
   (const ConstraintDescription& ci, const std::vector<SpatialVecT>& A_GB,
    std::vector<SpatialVecT>& A_AB) const {
    const int ncb = (int)ci.constrainedBodies.size();
    A_AB.resize(ncb);
    if (ci.ancestor == GroundIndex) {
        for (int b=0; b < ncb; ++b)
            A_AB[b] = A_GB[ci.constrainedBodies[b]];
        return;
    }
    const Transform_<T>& X_GA = c.tpc.getX_GB(ci.ancestor);
    const SpatialVecT&   V_GA = c.tvc.getV_GB(ci.ancestor);
    const SpatialVecT&   A_GA = A_GB[ci.ancestor];
    for (int b=0; b < ncb; ++b) {
        const MobilizedBodyIndex mbx = ci.constrainedBodies[b];
        A_AB[b] = mbx == ci.ancestor 
            ? SpatialVecT(Vec<3,T>(T(0)), Vec<3,T>(T(0)))
            : findRelativeAcceleration(X_GA, V_GA, A_GA, c.tpc.getX_GB(mbx),
                                       c.tvc.getV_GB(mbx), A_GB[mbx]);
    }
}

template <class T> void MatterScalarRep<T>::findConstrainedQ
   (const ConstraintDescription& ci, const T* all, std::vector<T>& constrained) {
    constrained.resize(ci.constrainedQ.size());
    for (int i=0; i < (int)ci.constrainedQ.size(); ++i)
        constrained[i] = all[ci.constrainedQ[i]];
}

// See SimbodyMatterSubsystemRep::realizeSubsystemPositionImpl().
template <class T> 
void MatterScalarRep<T>::realizeConstraintPositionErrors() const {
    const auto qErr = state.updQErr();
    const T* q = state.getQ().data() + state.getQStart(ix);
    std::vector<Transform_<T>> X_AB;
    std::vector<T>             cq;
    for (const ConstraintDescription& ci : s.constraints) {
        if (!ci.mp) continue;
        findConstrainedBodyTransforms(ci, X_AB);
        findConstrainedQ(ci, q, cq);
        ConstraintKinematics_<T> k;
        k.X_AB = X_AB.data(); k.q = cq.data();
        visit(ci, [&](const auto& e)
        {   e.calcPositionErrors(k, &qErr[s.qErrStart + ci.holoOffset]); });
    }
}

// See SimbodyMatterSubsystemRep::realizeSubsystemVelocityImpl().
template <class T> 
void MatterScalarRep<T>::realizeConstraintVelocityErrors() const {
    const auto uErr = state.updUErr();
    const T* q = state.getQ().data() + state.getQStart(ix);
    std::vector<Transform_<T>> X_AB;
    std::vector<SpatialVecT>   V_AB;
    std::vector<T>             cq, cqdot;
    for (const ConstraintDescription& ci : s.constraints) {
        if (!(ci.mp || ci.mv)) continue;
        findConstrainedBodyTransforms(ci, X_AB);
        findConstrainedBodyVelocities(ci, V_AB);
        findConstrainedQ(ci, q, cq);
        findConstrainedQ(ci, c.qdot.data(), cqdot);
        ConstraintKinematics_<T> k;
        k.X_AB = X_AB.data(); k.V_AB = V_AB.data();
        k.q = cq.data(); k.qdot = cqdot.data();
        visit(ci, [&](const auto& e) {
            if (ci.mp)
                e.calcPositionDotErrors(k, &uErr[s.uErrStart + ci.holoOffset]);
            if (ci.mv)
                e.calcVelocityErrors(k,
                    &uErr[s.uErrStart + s.mHolo + ci.nonholoOffset]);
        });
    }
}

// See SimbodyMatterSubsystem::calcConstraintAccelerationErrors() and
// SimbodyMatterSubsystemRep::calcConstraintAccelerationErrors().
template <class T> void MatterScalarRep<T>::calcConstraintAccelerationErrors
   (const std::vector<T>& knownUDotIn, std::vector<T>& pvaerr) const {
    const char* where = "calcConstraintAccelerationErrors()";
    requireStage(Stage::Velocity, where);
    const int nb = getNumBodies(), nu = s.nu;
    std::vector<T> zeroUDot;
    const std::vector<T>& knownUDot = orZero(knownUDotIn, nu, zeroUDot,
        T(0), "udots", where);
    pvaerr.assign(getNumMultipliers(), T(0));
    if (pvaerr.empty()) return;

    // Body accelerations and qdotdots consistent with the udots.
    const SBScalarStateDigest<T> sbs = makeDigest();
    std::vector<SpatialVecT> A_GB(nb);
    std::vector<T> qdotdot(s.nq, T(0));
    for (int i=0; i < (int)s.levels.size(); ++i)
        for (const RigidBodyNode_<T>* node : s.levels[i]) {
            node->calcBodyAccelerationsFromUdotOutward(c.tpc, c.tvc,
                knownUDot.data(), A_GB.data());
            if (i > 0) // Ground has no qdots
                node->calcQDotDot(sbs, knownUDot.data() + node->getUIndex(),
                                  qdotdot.data() + node->getQIndex());
        }

    const T* q = state.getQ().data() + state.getQStart(ix);
    std::vector<Transform_<T>> X_AB;
    std::vector<SpatialVecT>   V_AB, A_AB;
    std::vector<T>             cq, cqdot, cqdotdot;
    for (const ConstraintDescription& ci : s.constraints) {
        findConstrainedBodyTransforms(ci, X_AB);
        findConstrainedBodyVelocities(ci, V_AB);
        findConstrainedBodyAccelerations(ci, A_GB, A_AB);
        findConstrainedQ(ci, q, cq);
        findConstrainedQ(ci, c.qdot.data(), cqdot);
        findConstrainedQ(ci, qdotdot.data(), cqdotdot);
        ConstraintKinematics_<T> k;
        k.X_AB = X_AB.data(); k.V_AB = V_AB.data(); k.A_AB = A_AB.data();
        k.q = cq.data(); k.qdot = cqdot.data(); k.qdotdot = cqdotdot.data();
        visit(ci, [&](const auto& e) {
            if (ci.mp) e.calcPositionDotDotErrors(k, &pvaerr[ci.holoOffset]);
            if (ci.mv)
                e.calcVelocityDotErrors(k, &pvaerr[s.mHolo + ci.nonholoOffset]);
        });
    }
}

// See SimbodyMatterSubsystemRep::calcConstraintForcesFromMultipliers() and
// ConstraintImpl::calcConstraintForcesFromMultipliers().
template <class T> void MatterScalarRep<T>::calcConstraintForcesFromMultipliers
   (const std::vector<T>& multipliers, std::vector<SpatialVecT>& bodyForcesInG,
    std::vector<T>& mobilityForces, const char* where) const {
    requireStage(Stage::Velocity, where);
    const int m = getNumMultipliers();
    SimTK_ERRCHK2_ALWAYS((int)multipliers.size() == m, where,
        "Expected %d multipliers but got %d.", m, (int)multipliers.size());
    const SpatialVecT zero(Vec<3,T>(T(0)), Vec<3,T>(T(0)));
    bodyForcesInG.assign(getNumBodies(), zero);
    mobilityForces.assign(s.nu, T(0));

    const SBScalarStateDigest<T> sbs = makeDigest();
    const T* q = state.getQ().data() + state.getQStart(ix);
    std::vector<Transform_<T>> X_AB;
    std::vector<SpatialVecT>   F_A;
    std::vector<T>             cq, qForces, uForces;
    for (const ConstraintDescription& ci : s.constraints) {
        const int ncb = (int)ci.constrainedBodies.size();
        findConstrainedBodyTransforms(ci, X_AB);
        findConstrainedQ(ci, q, cq);
        ConstraintKinematics_<T> k;
        k.X_AB = X_AB.data(); k.q = cq.data();
        F_A.assign(ncb, zero);
        qForces.assign(ci.constrainedQ.size(), T(0));
        visit(ci, [&](const auto& e) {
            if (ci.mp)
                e.addInPositionConstraintForces(k, &multipliers[ci.holoOffset],
                                                F_A.data(), qForces.data());
            if (ci.mv)
                e.addInVelocityConstraintForces(k,
                    &multipliers[s.mHolo + ci.nonholoOffset], F_A.data());
        });

        // Re-express the body forces in Ground.
        if (ci.ancestor != GroundIndex) {
            const Rotation_<T>& R_GA = c.tpc.getX_GB(ci.ancestor).R();
            for (int b=0; b < ncb; ++b)
                F_A[b] = SpatialVecT(R_GA*F_A[b][0], R_GA*F_A[b][1]);
        }
        for (int b=0; b < ncb; ++b)
            bodyForcesInG[ci.constrainedBodies[b]] += F_A[b];

        // Convert the q forces to u forces, fu = ~N fq, for each constrained
        // mobilizer; see ConstraintImpl::convertQForcesToUForces().
        if (ci.constrainedQ.empty()) continue;
        uForces.assign(ci.constrainedU.size(), T(0));
        for (const ConstrainedMobilizerNodeInfo& mi : ci.constrainedMobilizers)
            if (mi.nu)
                s.nodes[mi.mobod]->multiplyByN(sbs, true,
                    &qForces[mi.firstConstrainedQ],
                    &uForces[mi.firstConstrainedU]);
        for (int i=0; i < (int)ci.constrainedU.size(); ++i)
            mobilityForces[ci.constrainedU[i]] += uForces[i];
    }
}

// See SimbodyMatterSubsystem::multiplyByGTranspose(): ~G*lambda is the
// generalized force equivalent to the constraint forces.
template <class T> void MatterScalarRep<T>::multiplyByGTranspose
   (const std::vector<T>& lambda, std::vector<T>& f) const {
    std::vector<SpatialVecT> bodyForcesInG;
    std::vector<T> mobilityForces;
    calcConstraintForcesFromMultipliers(lambda, bodyForcesInG, mobilityForces,
                                        "multiplyByGTranspose()");
    multiplyBySystemJacobianTranspose(bodyForcesInG, f);
    for (int i=0; i < s.nu; ++i) f[i] += mobilityForces[i];
}

} // namespace SimTK

//------------------------------------------------------------------------------
//                  MEMBER TEMPLATES OF THE SIMBODY CLASSES
//------------------------------------------------------------------------------
namespace SimTK {

// The matter subsystem is the only subsystem with T-typed computations, so
// realizing a stage for the system is realizing it for the matter subsystem.
template <class T> ScalarState<T>
MultibodySystem::createScalarState(const State& state) const {
    const char* where = "MultibodySystem::createScalarState()";
    SimTK_ERRCHK_ALWAYS(getMatterSubsystem().isOwnerOf(state), where,
        "The State is not a State of this system (its topology was realized "
        "by another system, or this system's topology has changed since).");
    SimTK_ERRCHK_ALWAYS(getMatterSubsystem().getNumParticles() == 0, where,
        "Particles are not yet supported for this scalar type.");
    return ScalarState<T>(*this, state);
}

template <class T> void 
MultibodySystem::realize(const ScalarState<T>& state, Stage stage) const {
    const char* where = "MultibodySystem::realize(ScalarState)";
    SimTK_ERRCHK_ALWAYS(&state.getSystem() == this, where,
        "This ScalarState belongs to a different system.");
    SimTK_ERRCHK_ALWAYS(state.getRealState().getSystemTopologyStageVersion()
                            == getSystemTopologyCacheVersion(), where,
        "This system's topology has changed since the ScalarState was "
        "created.");
    SimTK_ERRCHK1_ALWAYS(stage <= Stage::Velocity, where,
        "A ScalarState can be realized only through Stage::Velocity, not "
        "Stage::%s, since force subsystems don't support scalar types other "
        "than Real.", stage.getName().c_str());
    const SimbodyMatterSubsystem& matter = getMatterSubsystem();
    // Once the matter subsystem has a cache entry, check that it's this one.
    if (state.getNumCacheEntries(matter.getMySubsystemIndex()) > 0)
        (void)MatterScalarRep<T>(matter, state, where);
    if (stage >= Stage::Instance && state.getSystemStage() < Stage::Instance) {
        MatterScalarRep<T>::realizeInstance(matter, state);
        state.advanceSystemToStage(Stage::Instance);
    }
    if (stage >= Stage::Time && state.getSystemStage() < Stage::Time)
        state.advanceSystemToStage(Stage::Time);
    if (stage >= Stage::Position && state.getSystemStage() < Stage::Position) {
        MatterScalarRep<T>(matter, state, where).realizePosition();
        state.advanceSystemToStage(Stage::Position);
    }
    if (stage >= Stage::Velocity && state.getSystemStage() < Stage::Velocity) {
        MatterScalarRep<T>(matter, state, where).realizeVelocity();
        state.advanceSystemToStage(Stage::Velocity);
    }
}

template <class T> void SimbodyMatterSubsystem::
realizePositionKinematics(const ScalarState<T>& state) const
{   MultibodySystem::downcast(getSystem()).realize(state, Stage::Position); }
template <class T> void SimbodyMatterSubsystem::
realizeVelocityKinematics(const ScalarState<T>& state) const
{   MultibodySystem::downcast(getSystem()).realize(state, Stage::Velocity); }
template <class T> void SimbodyMatterSubsystem::
realizeArticulatedBodyInertias(const ScalarState<T>& state) const {
    const char* where = "realizeArticulatedBodyInertias()";
    MatterScalarRep<T>(*this, state, where).realizeArticulatedBodyInertias();
}
template <class T> T SimbodyMatterSubsystem::
calcKineticEnergy(const ScalarState<T>& state) const {
    const char* where = "calcKineticEnergy()";
    return MatterScalarRep<T>(*this, state, where).calcKineticEnergy();
}
template <class T> void SimbodyMatterSubsystem::
multiplyBySystemJacobian(const ScalarState<T>& state,
                         const std::vector<T>& u,
                         std::vector<SpatialVec_<T>>& Ju) const {
    const char* where = "multiplyBySystemJacobian()";
    MatterScalarRep<T>(*this, state, where).multiplyBySystemJacobian(u, Ju);
}
template <class T> void SimbodyMatterSubsystem::
multiplyBySystemJacobianTranspose(const ScalarState<T>& state,
                                  const std::vector<SpatialVec_<T>>& F_G,
                                  std::vector<T>& f) const {
    const char* where = "multiplyBySystemJacobianTranspose()";
    MatterScalarRep<T>(*this, state, where)
        .multiplyBySystemJacobianTranspose(F_G, f);
}
template <class T> void SimbodyMatterSubsystem::
multiplyByM(const ScalarState<T>& state, const std::vector<T>& a,
            std::vector<T>& Ma) const {
    MatterScalarRep<T>(*this, state, "multiplyByM()").multiplyByM(a, Ma);
}
template <class T> void SimbodyMatterSubsystem::
multiplyByMInv(const ScalarState<T>& state, const std::vector<T>& v,
               std::vector<T>& MinvV) const {
    MatterScalarRep<T>(*this, state, "multiplyByMInv()")
        .multiplyByMInv(v, MinvV);
}
template <class T> void SimbodyMatterSubsystem::
calcAccelerationIgnoringConstraints
   (const ScalarState<T>&               state,
    const std::vector<T>&               appliedMobilityForces,
    const std::vector<SpatialVec_<T>>&  appliedBodyForces,
    std::vector<T>&                     udot,
    std::vector<SpatialVec_<T>>&        A_GB) const {
    const char* where = "calcAccelerationIgnoringConstraints()";
    MatterScalarRep<T>(*this, state, where).calcTreeAccelerations(
        appliedMobilityForces, appliedBodyForces, udot, A_GB);
}
template <class T> void SimbodyMatterSubsystem::
calcResidualForceIgnoringConstraints
   (const ScalarState<T>&               state,
    const std::vector<T>&               appliedMobilityForces,
    const std::vector<SpatialVec_<T>>&  appliedBodyForces,
    const std::vector<T>&               knownUdot,
    std::vector<T>&                     residualMobilityForces) const {
    const char* where = "calcResidualForceIgnoringConstraints()";
    MatterScalarRep<T>(*this, state, where).calcTreeResidualForces(
        appliedMobilityForces, appliedBodyForces, knownUdot,
        residualMobilityForces);
}

template <class T> void SimbodyMatterSubsystem::
calcConstraintAccelerationErrors(const ScalarState<T>&  state,
                                 const std::vector<T>&  knownUDot,
                                 std::vector<T>&        pvaerr) const {
    const char* where = "calcConstraintAccelerationErrors()";
    MatterScalarRep<T>(*this, state, where)
        .calcConstraintAccelerationErrors(knownUDot, pvaerr);
}
template <class T> void SimbodyMatterSubsystem::
calcConstraintForcesFromMultipliers
   (const ScalarState<T>&               state,
    const std::vector<T>&               multipliers,
    std::vector<SpatialVec_<T>>&        bodyForcesInG,
    std::vector<T>&                     mobilityForces) const {
    const char* where = "calcConstraintForcesFromMultipliers()";
    MatterScalarRep<T>(*this, state, where).calcConstraintForcesFromMultipliers(
        multipliers, bodyForcesInG, mobilityForces, where);
}
template <class T> void SimbodyMatterSubsystem::
multiplyByGTranspose(const ScalarState<T>&  state,
                     const std::vector<T>&  lambda,
                     std::vector<T>&        f) const {
    MatterScalarRep<T>(*this, state, "multiplyByGTranspose()")
        .multiplyByGTranspose(lambda, f);
}

template <class T> const Transform_<T>& 
MobilizedBody::getBodyTransform(const ScalarState<T>& state) const {
    const char* where = "MobilizedBody::getBodyTransform()";
    const MatterScalarRep<T> rep(getMatterSubsystem(), state, where);
    rep.requireStage(Stage::Position, where);
    return rep.getCache().tpc.getX_GB(getMobilizedBodyIndex());
}
template <class T> const SpatialVec_<T>& 
MobilizedBody::getBodyVelocity(const ScalarState<T>& state) const {
    const char* where = "MobilizedBody::getBodyVelocity()";
    const MatterScalarRep<T> rep(getMatterSubsystem(), state, where);
    rep.requireStage(Stage::Velocity, where);
    return rep.getCache().tvc.getV_GB(getMobilizedBodyIndex());
}

} // namespace SimTK

/** Explicitly instantiate ScalarState<T> and all the Simbody member templates
that use it, for the scalar type T. Use this at global scope in one
translation unit; others can then include just Simbody.h. **/
#define SimTK_INSTANTIATE_MULTIBODY_SCALAR(T)                                  \
template class SimTK::ScalarState<T>;                                          \
template SimTK::ScalarState<T>                                                 \
SimTK::MultibodySystem::createScalarState<T>(const SimTK::State&) const;       \
template void SimTK::MultibodySystem::realize<T>                               \
    (const SimTK::ScalarState<T>&, SimTK::Stage) const;                        \
template void SimTK::SimbodyMatterSubsystem::realizePositionKinematics<T>      \
    (const SimTK::ScalarState<T>&) const;                                      \
template void SimTK::SimbodyMatterSubsystem::realizeVelocityKinematics<T>      \
    (const SimTK::ScalarState<T>&) const;                                      \
template void SimTK::SimbodyMatterSubsystem::realizeArticulatedBodyInertias<T> \
    (const SimTK::ScalarState<T>&) const;                                      \
template T SimTK::SimbodyMatterSubsystem::calcKineticEnergy<T>                 \
    (const SimTK::ScalarState<T>&) const;                                      \
template void SimTK::SimbodyMatterSubsystem::multiplyBySystemJacobian<T>       \
    (const SimTK::ScalarState<T>&, const std::vector<T>&,                      \
     std::vector<SimTK::SpatialVec_<T>>&) const;                               \
template void SimTK::SimbodyMatterSubsystem::                                  \
    multiplyBySystemJacobianTranspose<T>(const SimTK::ScalarState<T>&,         \
     const std::vector<SimTK::SpatialVec_<T>>&, std::vector<T>&) const;        \
template void SimTK::SimbodyMatterSubsystem::multiplyByM<T>                    \
    (const SimTK::ScalarState<T>&, const std::vector<T>&,                      \
     std::vector<T>&) const;                                                   \
template void SimTK::SimbodyMatterSubsystem::multiplyByMInv<T>                 \
    (const SimTK::ScalarState<T>&, const std::vector<T>&,                      \
     std::vector<T>&) const;                                                   \
template void SimTK::SimbodyMatterSubsystem::                                  \
    calcAccelerationIgnoringConstraints<T>(const SimTK::ScalarState<T>&,       \
     const std::vector<T>&, const std::vector<SimTK::SpatialVec_<T>>&,         \
     std::vector<T>&, std::vector<SimTK::SpatialVec_<T>>&) const;              \
template void SimTK::SimbodyMatterSubsystem::                                  \
    calcResidualForceIgnoringConstraints<T>(const SimTK::ScalarState<T>&,      \
     const std::vector<T>&, const std::vector<SimTK::SpatialVec_<T>>&,         \
     const std::vector<T>&, std::vector<T>&) const;                            \
template void SimTK::SimbodyMatterSubsystem::                                  \
    calcConstraintAccelerationErrors<T>(const SimTK::ScalarState<T>&,          \
     const std::vector<T>&, std::vector<T>&) const;                            \
template void SimTK::SimbodyMatterSubsystem::                                  \
    calcConstraintForcesFromMultipliers<T>(const SimTK::ScalarState<T>&,       \
     const std::vector<T>&, std::vector<SimTK::SpatialVec_<T>>&,               \
     std::vector<T>&) const;                                                   \
template void SimTK::SimbodyMatterSubsystem::multiplyByGTranspose<T>           \
    (const SimTK::ScalarState<T>&, const std::vector<T>&,                      \
     std::vector<T>&) const;                                                   \
template const SimTK::Transform_<T>& SimTK::MobilizedBody::getBodyTransform<T> \
    (const SimTK::ScalarState<T>&) const;                                      \
template const SimTK::SpatialVec_<T>& SimTK::MobilizedBody::getBodyVelocity<T> \
    (const SimTK::ScalarState<T>&) const;                                      \
static_assert(true, "") // require a semicolon

#endif // SimTK_SIMBODY_MULTIBODY_SCALAR_IMPL_H_
