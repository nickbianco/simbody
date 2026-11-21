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
#include "SimTKcommon/internal/System.h"

#include <algorithm>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimTK {

/** A State whose time and continuous variables (q, u, and z) have a scalar
type T other than Real, such as an automatic differentiation or symbolic type
(e.g. casadi::SX), for evaluating a System's computations with T.

A ScalarState<T> belongs to a System, and is created from a (Real) State of
that System that has been realized through Stage::Instance. It keeps that
State for the System's structure (its Topology, Model, and Instance stage
information, and its discrete variables) and has T-typed copies of the time
and continuous variables, with the same, fixed layout. Subsystems store their
T-typed computed quantities in cache entries of the ScalarState<T>, as they do
in a State. Create and realize a ScalarState<T> with the System's methods for
it (e.g. MultibodySystem::createScalarState() and realize()):
@code
    State state = system.realizeTopology();
    system.realize(state, Stage::Instance);

    ScalarState<casadi::SX> sx = system.createScalarState<casadi::SX>(state);
    sx.setQ(q);  sx.setU(u);                // std::vector<casadi::SX>
    sx.updQ()[0] = q0;                      // or change elements in place
    system.realize(sx, Stage::Velocity);
@endcode

Compared to State, a ScalarState<T>:
  - has a single, system-wide realized stage (no per-subsystem stages);
  - has cache entries that are valid whenever the system stage is at least
    the stage they were allocated at, with no lazy evaluation or version
    tracking;
  - provides the discrete variables and Instance stage quantities only
    through the (Real) State it was created from, getRealState();
  - uses std::vector<T> for the continuous variables, which can be read as
    vectors, set as whole vectors of the right length, or changed in place
    through fixed-length views (std::span<T>);
  - has T-typed position and velocity constraint errors (qErr and uErr, with
    the same layout as in a State), but no acceleration constraint errors,
    since a ScalarState can't be realized past Stage::Velocity.

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
    /** A writable view of a vector of variables, whose elements can be
    changed but whose length can't. **/
    typedef std::span<T> VectorView;

    /** Create a ScalarState for \a system from \a state, which must be a
    State of \a system (with its current topology) realized through
    Stage::Instance. The time and continuous variables are copied from
    \a state. The system stage is Stage::Model. A System that can realize a
    ScalarState usually provides a method that calls this one (e.g.
    MultibodySystem::createScalarState()). **/
    ScalarState(const System& system, const State& state)
    :   system(&system), realState(std::make_shared<const State>(state)) {
        const char* where = "ScalarState::ScalarState()";
        SimTK_STAGECHECK_GE_ALWAYS(state.getSystemStage(), Stage::Instance,
                                   where);
        SimTK_ERRCHK_ALWAYS(
            state.getSystemTopologyStageVersion()
                == system.getSystemTopologyCacheVersion()
            && state.getNumSubsystems() == system.getNumSubsystems(), where,
            "The State is not a State of this System, or the System's "
            "topology has changed since the State was created.");
        const int nss = state.getNumSubsystems();
        ranges.resize(nss);
        cache.resize(nss);
        for (SubsystemIndex i(0); i < nss; ++i) {
            ranges[i].qStart = state.getQStart(i); ranges[i].nq=state.getNQ(i);
            ranges[i].uStart = state.getUStart(i); ranges[i].nu=state.getNU(i);
            ranges[i].zStart = state.getZStart(i); ranges[i].nz=state.getNZ(i);
            ranges[i].qErrStart = state.getQErrStart(i);
            ranges[i].nqErr     = state.getNQErr(i);
            ranges[i].uErrStart = state.getUErrStart(i);
            ranges[i].nuErr     = state.getNUErr(i);
        }
        time = T(state.getTime());
        convert(state.getQ(), q);
        convert(state.getU(), u);
        convert(state.getZ(), z);
        qErr.assign(state.getNQErr(), T(NaN));
        uErr.assign(state.getNUErr(), T(NaN));
    }

    /** The System that this belongs to. **/
    const System& getSystem() const {return *system;}
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
    /** A view of q for changing its elements. Invalidates Stage::Position
    and above. **/
    VectorView updQ() {invalidateAll(Stage::Position); return VectorView(q);}
    /** Invalidates Stage::Velocity and above. **/
    VectorView updU() {invalidateAll(Stage::Velocity); return VectorView(u);}
    /** Invalidates Stage::Dynamics and above. **/
    VectorView updZ() {invalidateAll(Stage::Dynamics); return VectorView(z);}
    /** Set all of q, which must have the System's number of q's.
    Invalidates Stage::Position and above. **/
    void setQ(const VectorType& v) {set(v, q, Stage::Position, "setQ()");}
    /** Invalidates Stage::Velocity and above. **/
    void setU(const VectorType& v) {set(v, u, Stage::Velocity, "setU()");}
    /** Invalidates Stage::Dynamics and above. **/
    void setZ(const VectorType& v) {set(v, z, Stage::Dynamics, "setZ()");}

    /** Each subsystem's continuous variables are a contiguous range of the
    system's, as in a State. **/
    SystemQIndex getQStart(SubsystemIndex i) const {return ranges[i].qStart;}
    int          getNQ(SubsystemIndex i)     const {return ranges[i].nq;}
    SystemUIndex getUStart(SubsystemIndex i) const {return ranges[i].uStart;}
    int          getNU(SubsystemIndex i)     const {return ranges[i].nu;}
    SystemZIndex getZStart(SubsystemIndex i) const {return ranges[i].zStart;}
    int          getNZ(SubsystemIndex i)     const {return ranges[i].nz;}
    /**@}**/

    /** @name Constraint errors
    The position (qErr) and velocity (uErr) constraint errors, computed by
    the subsystems when the ScalarState is realized through Stage::Position
    and Stage::Velocity. Each subsystem's errors are a contiguous range of the
    system's, as in a State. **/
    /**@{**/
    int getNQErr() const {return (int)qErr.size();}
    int getNUErr() const {return (int)uErr.size();}
    /** Requires Stage::Position. **/
    const VectorType& getQErr() const {
        SimTK_STAGECHECK_GE_ALWAYS(stage, Stage::Position,
                                   "ScalarState::getQErr()");
        return qErr;
    }
    /** Requires Stage::Velocity. **/
    const VectorType& getUErr() const {
        SimTK_STAGECHECK_GE_ALWAYS(stage, Stage::Velocity,
                                   "ScalarState::getUErr()");
        return uErr;
    }
    /** For use by subsystems while realizing Stage::Position; these can be
    modified in a const ScalarState, like a cache entry. **/
    VectorView updQErr() const {return VectorView(qErr);}
    /** For use by subsystems while realizing Stage::Velocity. **/
    VectorView updUErr() const {return VectorView(uErr);}

    SystemQErrIndex getQErrStart(SubsystemIndex i) const
    {   return ranges[i].qErrStart; }
    int getNQErr(SubsystemIndex i) const {return ranges[i].nqErr;}
    SystemUErrIndex getUErrStart(SubsystemIndex i) const
    {   return ranges[i].uErrStart; }
    int getNUErr(SubsystemIndex i) const {return ranges[i].nuErr;}
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
        SystemQErrIndex qErrStart; SystemUErrIndex uErrStart;
        int nqErr = 0, nuErr = 0;
    };
    struct CacheEntry {
        Stage                    stage;
        ClonePtr<AbstractValue>  value;
    };

    static void convert(const Vector& v, VectorType& out) {
        out.resize(v.size());
        for (int i=0; i < v.size(); ++i) out[i] = T(v[i]);
    }
    // The variables' lengths are fixed (by the System's layout) when the
    // ScalarState is created.
    void set(const VectorType& v, VectorType& vars, Stage g,
             const char* where) {
        SimTK_ERRCHK2_ALWAYS(v.size() == vars.size(), where,
            "Expected a vector of length %d but got %d.",
            (int)vars.size(), (int)v.size());
        invalidateAll(g);
        std::copy(v.begin(), v.end(), vars.begin());
    }
    const CacheEntry& getEntry(SubsystemIndex i, CacheEntryIndex j,
                               const char* where) const {
        SimTK_INDEXCHECK_ALWAYS(i, getNumSubsystems(), where);
        SimTK_INDEXCHECK_ALWAYS(j, (int)cache[i].size(), where);
        return cache[i][j];
    }

    const System*                                   system;
    // The System's structure; shared by copies since it doesn't change.
    std::shared_ptr<const State>                    realState;
    std::vector<Ranges>                             ranges;
    T                                               time;
    VectorType                                      q, u, z;
    // Like a State's constraint errors, these are computed by realization.
    mutable VectorType                              qErr, uErr;
    // Like a State's cache, these can change in a const ScalarState.
    mutable Stage                                   stage = Stage::Model;
    mutable std::vector<std::vector<CacheEntry>>    cache;
};

} // namespace SimTK

#endif // SimTK_SimTKCOMMON_SCALAR_STATE_H_
