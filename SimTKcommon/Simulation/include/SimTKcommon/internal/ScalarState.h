#ifndef SimTK_SimTKCOMMON_SCALAR_STATE_H_
#define SimTK_SimTKCOMMON_SCALAR_STATE_H_

/* -------------------------------------------------------------------------- *
 *                       Simbody(tm): SimTKcommon                             *
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
Declares and defines ScalarState<T>, a State whose time and continuous
variables have a scalar type T other than Real. **/

#include "SimTKcommon/internal/common.h"
#include "SimTKcommon/internal/ClonePtr.h"
#include "SimTKcommon/internal/Value.h"
#include "SimTKcommon/internal/State.h"

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimTK {

/** A State whose time and continuous variables (q, u, and z) have a scalar
type T other than Real, such as an automatic differentiation or symbolic type
(e.g. casadi::SX), for evaluating a System's computations with T.

A ScalarState<T> is created from a (Real) State that has been realized through
Stage::Instance. It keeps that State for the System's structure (its Topology,
Model, and Instance stage information, and its discrete variables) and has
T-typed copies of the time and continuous variables, with the same layout.
Subsystems store their T-typed computed quantities in cache entries of the
ScalarState<T>, as they do in a State. Realize a ScalarState<T> with the
System's realize() method for ScalarState<T> (e.g. MultibodySystem::realize()):
@code
    State state = system.realizeTopology();
    system.realize(state, Stage::Instance);

    ScalarState<casadi::SX> sx(state);
    sx.updQ() = q;  sx.updU() = u;          // std::vector<casadi::SX>
    system.realize(sx, Stage::Velocity);
@endcode

Compared to State, a ScalarState<T>:
  - has a single, system-wide realized stage (no per-subsystem stages);
  - has cache entries that are valid whenever the system stage is at least
    the stage they were allocated at, with no lazy evaluation or version
    tracking;
  - provides the discrete variables and Instance stage quantities only
    through the (Real) State it was created from, getRealState();
  - uses std::vector<T> for the continuous variables.

After construction the system stage is Stage::Model: realizing
Stage::Instance lets the subsystems create their T-typed Instance stage
quantities. Changing the time invalidates Stage::Time and above, q
Stage::Position and above, u Stage::Velocity and above, and z Stage::Dynamics
and above.

The scalar type needs traits defined by a header following the recipe in
SimTKcommon/internal/RealScalarType.h. For Real, use State. **/
template <class T>
class ScalarState {
public:
    static_assert(!std::is_same<T,Real>::value, "ScalarState<T> is for "
                  "scalar types other than Real; use State.");

    /** The type of the continuous variables q, u, and z. **/
    typedef std::vector<T> VectorType;

    /** Create a ScalarState from \a state, which must be realized through
    Stage::Instance. The time and continuous variables are copied from
    \a state. The system stage is Stage::Model. **/
    explicit ScalarState(const State& state)
    :   realState(std::make_shared<const State>(state)) {
        SimTK_STAGECHECK_GE_ALWAYS(state.getSystemStage(), Stage::Instance,
                                   "ScalarState::ScalarState()");
        const int nss = state.getNumSubsystems();
        ranges.resize(nss);
        cache.resize(nss);
        for (SubsystemIndex i(0); i < nss; ++i) {
            ranges[i].qStart = state.getQStart(i); ranges[i].nq=state.getNQ(i);
            ranges[i].uStart = state.getUStart(i); ranges[i].nu=state.getNU(i);
            ranges[i].zStart = state.getZStart(i); ranges[i].nz=state.getNZ(i);
        }
        time = T(state.getTime());
        convert(state.getQ(), q);
        convert(state.getU(), u);
        convert(state.getZ(), z);
    }

    /** The (Real) State that this was created from. It provides the
    System's structure, including the discrete variables. **/
    const State& getRealState() const {return *realState;}

    /** @name Stages **/
    /**@{**/
    int getNumSubsystems() const {return (int)ranges.size();}
    /** The highest stage that has been realized. **/
    const Stage& getSystemStage() const {return stage;}
    /** If the system stage is at or above \a g, back up to the stage just
    prior to \a g. **/
    void invalidateAll(Stage g) {invalidateAllCacheAtOrAbove(g);}
    /** Like invalidateAll(), for use by the System's realize() methods. **/
    void invalidateAllCacheAtOrAbove(Stage g) const
    {   if (stage >= g) stage = g.prev(); }
    /** Advance the system stage to \a g, which must be one stage higher than
    the current system stage. For use by the System's realize() methods. **/
    void advanceSystemToStage(Stage g) const {
        SimTK_ERRCHK2_ALWAYS(g == stage.next(),
            "ScalarState::advanceSystemToStage()",
            "Expected to advance to Stage::%s but got Stage::%s.",
            stage.next().getName().c_str(), g.getName().c_str());
        stage = g;
    }
    /**@}**/

    /** @name Time and continuous variables **/
    /**@{**/
    const T& getTime() const {return time;}
    /** Invalidates Stage::Time and above. **/
    void setTime(const T& t) {invalidateAll(Stage::Time); time = t;}

    int getNQ() const {return (int)q.size();}
    int getNU() const {return (int)u.size();}
    int getNZ() const {return (int)z.size();}

    const VectorType& getQ() const {return q;}
    const VectorType& getU() const {return u;}
    const VectorType& getZ() const {return z;}
    /** Invalidates Stage::Position and above. **/
    VectorType& updQ() {invalidateAll(Stage::Position); return q;}
    /** Invalidates Stage::Velocity and above. **/
    VectorType& updU() {invalidateAll(Stage::Velocity); return u;}
    /** Invalidates Stage::Dynamics and above. **/
    VectorType& updZ() {invalidateAll(Stage::Dynamics); return z;}
    void setQ(const VectorType& v) {checkSize(v, q, "setQ()"); updQ() = v;}
    void setU(const VectorType& v) {checkSize(v, u, "setU()"); updU() = v;}
    void setZ(const VectorType& v) {checkSize(v, z, "setZ()"); updZ() = v;}

    /** Each subsystem's continuous variables are a contiguous range of the
    system's, as in a State. **/
    SystemQIndex getQStart(SubsystemIndex i) const {return ranges[i].qStart;}
    int          getNQ(SubsystemIndex i)     const {return ranges[i].nq;}
    SystemUIndex getUStart(SubsystemIndex i) const {return ranges[i].uStart;}
    int          getNU(SubsystemIndex i)     const {return ranges[i].nu;}
    SystemZIndex getZStart(SubsystemIndex i) const {return ranges[i].zStart;}
    int          getNZ(SubsystemIndex i)     const {return ranges[i].nz;}
    /**@}**/

    /** @name Cache entries
    For use by subsystems, to store their T-typed computed quantities. A
    cache entry is allocated at a stage, and is valid whenever the system
    stage is at or above that stage. **/
    /**@{**/
    /** Take ownership of \a value, a heap-allocated cache entry for
    subsystem \a i that is valid at Stage \a g and above. **/
    CacheEntryIndex allocateCacheEntry(SubsystemIndex i, Stage g,
                                       AbstractValue* value) const {
        cache[i].push_back(CacheEntry{g, ClonePtr<AbstractValue>(value)});
        return CacheEntryIndex((int)cache[i].size() - 1);
    }
    int getNumCacheEntries(SubsystemIndex i) const
    {   return (int)cache[i].size(); }
    const AbstractValue& getCacheEntry(SubsystemIndex i,
                                       CacheEntryIndex j) const {
        const char* where = "ScalarState::getCacheEntry()";
        const CacheEntry& e = getEntry(i, j, where);
        SimTK_STAGECHECK_GE_ALWAYS(stage, e.stage, where);
        return *e.value;
    }
    /** The cache entry can be modified in a const ScalarState. This does not
    check the stage, so that subsystems can compute an entry's value while
    realizing its stage. **/
    AbstractValue& updCacheEntry(SubsystemIndex i, CacheEntryIndex j) const {
        getEntry(i, j, "ScalarState::updCacheEntry()"); // check the indices
        return *cache[i][j].value;
    }
    /**@}**/

private:
    struct Ranges {
        SystemQIndex qStart; SystemUIndex uStart; SystemZIndex zStart;
        int nq = 0, nu = 0, nz = 0;
    };
    struct CacheEntry {
        Stage                    stage;
        ClonePtr<AbstractValue>  value;
    };

    static void convert(const Vector& v, VectorType& out) {
        out.resize(v.size());
        for (int i=0; i < v.size(); ++i) out[i] = T(v[i]);
    }
    static void checkSize(const VectorType& v, const VectorType& current,
                          const char* where) {
        SimTK_ERRCHK2_ALWAYS(v.size() == current.size(), where,
            "Expected a vector of length %d but got %d.",
            (int)current.size(), (int)v.size());
    }
    const CacheEntry& getEntry(SubsystemIndex i, CacheEntryIndex j,
                               const char* where) const {
        SimTK_INDEXCHECK_ALWAYS(i, getNumSubsystems(), where);
        SimTK_INDEXCHECK_ALWAYS(j, (int)cache[i].size(), where);
        return cache[i][j];
    }

    // The System's structure; shared by copies since it doesn't change.
    std::shared_ptr<const State>                    realState;
    std::vector<Ranges>                             ranges;
    T                                               time;
    VectorType                                      q, u, z;
    // Like a State's cache, these can change in a const ScalarState.
    mutable Stage                                   stage = Stage::Model;
    mutable std::vector<std::vector<CacheEntry>>    cache;
};

} // namespace SimTK

#endif // SimTK_SimTKCOMMON_SCALAR_STATE_H_
