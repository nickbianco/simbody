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

The matter subsystem keeps its T-typed quantities in a cache entry of the
ScalarState<T>, as it does in a State: a MatterScalarCache<T>, allocated when
the ScalarState<T> is realized through Stage::Instance. It holds the T-typed
RigidBodyNodes (created by a RigidBodyNodeFactory_<T> and shared by copies)
and the T-typed tree cache entries. The tree sweeps here mirror the ones in
SimbodyMatterSubsystemRep, but use an SBScalarStateDigest<T>.

Note that the headers in this directory are Simbody implementation details
that are templatized on the scalar type; they declare classes in the global
namespace and contain a "using namespace SimTK" directive, so include them only
where needed. **/

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
#include "simbody/internal/tree/RigidBodyNodeFactory.h"

#include <memory>
#include <type_traits>
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
    void createPin(const RigidBodyNodeInfo& info) override {
        requireFree(info);
        UIndex u = info.uIndex; USquaredIndex usq = info.uSqIndex;
        QIndex q = info.qIndex;
        add(info, new RBNodeTorsion_<T>(sbCast<T>(info.massProperties),
                                        info.isReversed, u, usq, q));
    }
    void createSlider(const RigidBodyNodeInfo& info) override {
        requireFree(info);
        UIndex u = info.uIndex; USquaredIndex usq = info.uSqIndex;
        QIndex q = info.qIndex;
        add(info, new RBNodeSlider_<T>(sbCast<T>(info.massProperties),
                                       info.isReversed, u, usq, q));
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

private:
    static MatterScalarCache<T>& getMatterCache(
        const SimbodyMatterSubsystem& matter, const ScalarState<T>& state,
        const char* methodName);
    int getNumBodies() const {return s.getNumBodies();}
    SBScalarStateDigest<T> makeDigest() const;

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
    SimTK_ERRCHK_ALWAYS(matter.getNumConstraints() == 0, where,
        "Constraints are not yet supported for this scalar type.");
    SimTK_ERRCHK_ALWAYS(matter.getNumParticles() == 0, where,
        "Particles are not yet supported for this scalar type.");

    auto s = std::make_shared<MatterScalarStructure<T>>();
    s->matter = &matter;
    RigidBodyNodeFactory_<T> factory(*s);
    matter.createRigidBodyNodes(factory, state.getRealState());
    s->nq = state.getNQ(ix);
    s->nu = state.getNU(ix);

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
    sbs.setQ(state.getQ().data() + state.getQStart(ix));
    sbs.setU(state.getU().data() + state.getUStart(ix));
    sbs.setQDot(c.qdot.data());
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
}

template <class T> void MatterScalarRep<T>::realizeVelocity() const {
    const SBScalarStateDigest<T> sbs = makeDigest();
    for (const auto& level : s.levels)
        for (const RigidBodyNode_<T>* node : level)
            node->realizeVelocity(sbs);
}

// See SimbodyMatterSubsystemRep::realizeArticulatedBodyInertias(). 
template <class T> void 
MatterScalarRep<T>::realizeArticulatedBodyInertias() const {
    requireStage(Stage::Position, "realizeArticulatedBodyInertias()");
    if (c.abiRealized) return;
    const auto& levels = s.levels;
    for (int i=(int)levels.size()-1; i >= 0; --i)
        for (const RigidBodyNode_<T>* node : levels[i])
            node->realizeArticulatedBodyInertiasInward(s.ic, c.tpc, c.abc);
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
    for (const auto& level : s.levels)
        for (const RigidBodyNode_<T>* node : level)
            node->multiplyBySystemJacobian(c.tpc, v.data(), Jv.data());
}

// See SimbodyMatterSubsystemRep::multiplyBySystemJacobianTranspose().
template <class T> void MatterScalarRep<T>::multiplyBySystemJacobianTranspose
   (const std::vector<SpatialVecT>& X, std::vector<T>& JtX) const {
    const char* where = "multiplyBySystemJacobianTranspose()";
    requireStage(Stage::Position, where);
    SimTK_ERRCHK2_ALWAYS((int)X.size() == getNumBodies(), where,
        "Expected %d body forces but got %d.", getNumBodies(), (int)X.size());
    std::vector<SpatialVecT> zTmp(getNumBodies());
    JtX.assign(s.nu, T(0));
    const auto& levels = s.levels;
    for (int i=(int)levels.size()-1; i >= 0; --i)
        for (const RigidBodyNode_<T>* node : levels[i])
            node->multiplyBySystemJacobianTranspose(c.tpc, zTmp.data(), 
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
    Ma.assign(s.nu, T(0));
    const auto& levels = s.levels;
    for (int i=0; i < (int)levels.size(); ++i)
        for (const RigidBodyNode_<T>* node : levels[i])
            node->multiplyByMPass1Outward(c.tpc, a.data(), allA_GB.data());
    for (int i=(int)levels.size()-1; i >= 0; --i)
        for (const RigidBodyNode_<T>* node : levels[i])
            node->multiplyByMPass2Inward(c.tpc, allA_GB.data(), allF.data(),
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
    MinvV.assign(nu, T(0));
    const auto& levels = s.levels;
    for (int i=(int)levels.size()-1; i >= 0; --i)
        for (const RigidBodyNode_<T>* node : levels[i])
            node->multiplyByMInvPass1Inward(s.ic, c.tpc, c.abc, v.data(),
                z.data(), zPlus.data(), eps.data());
    for (int i=0; i < (int)levels.size(); ++i)
        for (const RigidBodyNode_<T>* node : levels[i])
            node->multiplyByMInvPass2Outward(s.ic, c.tpc, c.abc, 
                eps.data(), A_GB.data(), MinvV.data());
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
    std::vector<T> zeroF; std::vector<SpatialVecT> zeroBF;
    const std::vector<T>& mobilityForces = orZero(mobilityForcesIn, nu, zeroF,
        T(0), "mobility forces", where);
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

    for (int i=(int)levels.size()-1; i >= 0; --i)
        for (const RigidBodyNode_<T>* node : levels[i])
            node->calcUDotPass1Inward(s.ic, c.tpc, c.abc, c.abvc,
                mobilityForces.data(), bodyForces.data(), udot.data(),
                c.tac.z.begin(), c.tac.zPlus.begin(), epsilonPtr);
    for (int i=0; i < (int)levels.size(); ++i)
        for (const RigidBodyNode_<T>* node : levels[i])
            node->calcUDotPass2Outward(s.ic, c.tpc, c.abc, c.tvc, c.dc,
                epsilonPtr, aPtr, udot.data(), tauPtr);

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
    std::vector<T> zeroF, zeroUDot; std::vector<SpatialVecT> zeroBF;
    const std::vector<T>& mobilityForces = orZero(mobilityForcesIn, nu, zeroF,
        T(0), "mobility forces", where);
    const std::vector<SpatialVecT>& bodyForces = orZero(bodyForcesIn, nb, 
        zeroBF, SpatialVecT(Vec<3,T>(0), Vec<3,T>(0)), "body forces", where);
    const std::vector<T>& knownUdot = orZero(knownUdotIn, nu, zeroUDot,
        T(0), "udots", where);

    std::vector<SpatialVecT> A_GB(nb), allF(nb);
    residual.assign(nu, T(0));
    const auto& levels = s.levels;
    for (int i=0; i < (int)levels.size(); ++i)
        for (const RigidBodyNode_<T>* node : levels[i])
            node->calcBodyAccelerationsFromUdotOutward(c.tpc, c.tvc,
                knownUdot.data(), A_GB.data());
    for (int i=(int)levels.size()-1; i >= 0; --i)
        for (const RigidBodyNode_<T>* node : levels[i])
            node->calcInverseDynamicsPass2Inward(c.tpc, c.tvc, A_GB.data(),
                mobilityForces.data(), bodyForces.data(), allF.data(),
                residual.data());
}

} // namespace SimTK

//------------------------------------------------------------------------------
//                  MEMBER TEMPLATES OF THE SIMBODY CLASSES
//------------------------------------------------------------------------------
namespace SimTK {

// The matter subsystem is the only subsystem with T-typed computations, so
// realizing a stage for the system is realizing it for the matter subsystem.
template <class T> void 
MultibodySystem::realize(const ScalarState<T>& state, Stage stage) const {
    const char* where = "MultibodySystem::realize(ScalarState)";
    const State& realState = state.getRealState();
    SimTK_ERRCHK_ALWAYS(
        realState.getSystemTopologyStageVersion() 
            == getSystemTopologyCacheVersion()
        && state.getNumSubsystems() == getNumSubsystems(), where,
        "This ScalarState was not created from a State of this system.");
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
template const SimTK::Transform_<T>& SimTK::MobilizedBody::getBodyTransform<T> \
    (const SimTK::ScalarState<T>&) const;                                      \
template const SimTK::SpatialVec_<T>& SimTK::MobilizedBody::getBodyVelocity<T> \
    (const SimTK::ScalarState<T>&) const;                                      \
static_assert(true, "") // require a semicolon

#endif // SimTK_SIMBODY_MULTIBODY_SCALAR_IMPL_H_
