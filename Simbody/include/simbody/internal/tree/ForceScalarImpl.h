#ifndef SimTK_SIMBODY_FORCE_SCALAR_IMPL_H_
#define SimTK_SIMBODY_FORCE_SCALAR_IMPL_H_

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
The implementation of the Force_<T>, Force_<T>::Custom, and
GeneralForceSubsystem_<T> templates. Simbody compiles them for Real. To use
another scalar type T, include this header in one translation unit and write
SimTK_INSTANTIATE_FORCE_SCALAR(T) at global scope (see MultibodyScalarImpl.h
for the requirements on T). **/

#include "SimTKcommon.h"
#include "SimTKcommon/internal/PrivateImplementation_Defs.h"
#include "simbody/internal/common.h"
#include "simbody/internal/ForceSubsystem.h"
#include "simbody/internal/ForceSubsystemGuts.h"
#include "simbody/internal/GeneralForceSubsystem.h"
#include "simbody/internal/MultibodySystem.h"
#include "simbody/internal/SimbodyMatterSubsystem.h"
#include "simbody/internal/Force.h"
#include "simbody/internal/Force_Custom.h"

#include <cassert>
#include <vector>

namespace SimTK {

// The Real general force subsystem has its own implementation (with parallel
// evaluation and caching of position-only forces) in GeneralForceSubsystem.cpp.
template <> class GeneralForceSubsystemRep_<Real>;

namespace ForceScalarDetail {
// Size and zero force vectors of either type.
inline void resizeAndZero(Vector_<SpatialVec>& v, int n)
{   v.resize(n); v.setToZero(); }
inline void resizeAndZero(Vector_<Vec3>& v, int n)
{   v.resize(n); v.setToZero(); }
inline void resizeAndZero(Vector& v, int n)
{   v.resize(n); v.setToZero(); }
template <class T>
void resizeAndZero(std::vector<T>& v, int n) {v.assign(n, T(0));}
template <class T>
void resizeAndZero(std::vector<Vec<3,T>>& v, int n)
{   v.assign(n, Vec<3,T>(T(0))); }
template <class T>
void resizeAndZero(std::vector<SpatialVec_<T>>& v, int n)
{   v.assign(n, SpatialVec_<T>(Vec<3,T>(T(0)), Vec<3,T>(T(0)))); }

// The Real State holding the structure and discrete variables.
inline const State& getRealState(const State& s) {return s;}
template <class T>
const State& getRealState(const ScalarState<T>& s) {return s.getRealState();}
} // namespace ForceScalarDetail

//==============================================================================
//                                FORCE IMPL
//==============================================================================
/** The base class of the implementations of Force_<T> elements. calcForce()
and calcPotentialEnergy() take the state and force vector types for T; the
realize...() methods always take the Real State. **/
template <class T>
class ForceImpl_ : public PIMPLImplementation<Force_<T>, ForceImpl_<T>> {
public:
    using StateType           = typename Force_<T>::StateType;
    using BodyForceVector     = typename Force_<T>::BodyForceVector;
    using ParticleForceVector = typename Force_<T>::ParticleForceVector;
    using MobilityForceVector = typename Force_<T>::MobilityForceVector;

    ForceImpl_() : forces(0), defaultDisabled(false) {}
    ForceImpl_(const ForceImpl_& clone) {*this = clone;}

    void setDisabledByDefault(bool shouldBeDisabled)
    {   invalidateTopologyCache();
        defaultDisabled = shouldBeDisabled; }

    bool isDisabledByDefault() const
    {   return defaultDisabled; }

    virtual ~ForceImpl_() {}
    virtual ForceImpl_* clone() const = 0;
    virtual bool dependsOnlyOnPositions() const {
        return false;
    }
    virtual bool shouldBeParallelIfPossible() const{
        return false;
    }
    ForceIndex getForceIndex() const {return index;}
    const GeneralForceSubsystem_<T>& getForceSubsystem() const
    {   assert(forces); return *forces; }
    void setForceSubsystem(GeneralForceSubsystem_<T>& frcsub, ForceIndex ix) {
        forces = &frcsub;
        index  = ix;
    }
    void invalidateTopologyCache() const {
        if (forces) forces->invalidateSubsystemTopologyCache();
    }

    // Every force element must implement this method (it is pure virtual) to
    // add its forces to the given body, particle, and mobility forces.
    virtual void calcForce(const StateType&     state,
                           BodyForceVector&     bodyForces,
                           ParticleForceVector& particleForces,
                           MobilityForceVector& mobilityForces) const = 0;

    // Every force element must implement this method (it is pure virtual) to
    // return its potential energy, or zero.
    virtual T calcPotentialEnergy(const StateType& state) const = 0;

    virtual void realizeTopology    (State& state) const {}
    virtual void realizeModel       (State& state) const {}
    virtual void realizeInstance    (const State& state) const {}
    virtual void realizeTime        (const State& state) const {}
    virtual void realizePosition    (const State& state) const {}
    virtual void realizeVelocity    (const State& state) const {}
    virtual void realizeDynamics    (const State& state) const {}
    virtual void realizeAcceleration(const State& state) const {}
    virtual void realizeReport      (const State& state) const {}

    virtual void calcDecorativeGeometryAndAppend
       (const State& s, Stage stage, Array_<DecorativeGeometry>& geom) const {}

private:
    GeneralForceSubsystem_<T>* forces;  // just a reference; no delete on destruction
    ForceIndex                 index;
    bool                       defaultDisabled;
};

//==============================================================================
//                                CUSTOM IMPL
//==============================================================================
template <class T>
class Force_<T>::CustomImpl : public ForceImpl_<T> {
    using Implementation      = typename Force_<T>::Custom::Implementation;
public:
    using StateType           = typename Force_<T>::StateType;
    using BodyForceVector     = typename Force_<T>::BodyForceVector;
    using ParticleForceVector = typename Force_<T>::ParticleForceVector;
    using MobilityForceVector = typename Force_<T>::MobilityForceVector;

    CustomImpl(Implementation* implementation)
    :   implementation(implementation) {}
    CustomImpl* clone() const override {
        return new CustomImpl(*this);
    }
    bool dependsOnlyOnPositions() const override {
        return implementation->dependsOnlyOnPositions();
    }
    void calcForce(const StateType& state, BodyForceVector& bodyForces,
                   ParticleForceVector& particleForces,
                   MobilityForceVector& mobilityForces) const override {
        implementation->calcForce(state, bodyForces, particleForces,
                                  mobilityForces);
    }
    T calcPotentialEnergy(const StateType& state) const override {
        return implementation->calcPotentialEnergy(state);
    }
    bool shouldBeParallelIfPossible() const override {
        return implementation->shouldBeParallelIfPossible();
    }
    ~CustomImpl() {
        delete implementation;
    }
    const Implementation& getImplementation() const {
        return *implementation;
    }
    Implementation& updImplementation() {
        return *implementation;
    }
protected:
    void realizeTopology(State& state) const override {
        implementation->realizeTopology(state);
    }
    void realizeModel(State& state) const override {
        implementation->realizeModel(state);
    }
    void realizeInstance(const State& state) const override {
        implementation->realizeInstance(state);
    }
    void realizeTime(const State& state) const override {
        implementation->realizeTime(state);
    }
    void realizePosition(const State& state) const override {
        implementation->realizePosition(state);
    }
    void realizeVelocity(const State& state) const override {
        implementation->realizeVelocity(state);
    }
    void realizeDynamics(const State& state) const override {
        implementation->realizeDynamics(state);
    }
    void realizeAcceleration(const State& state) const override {
        implementation->realizeAcceleration(state);
    }
    void realizeReport(const State& state) const override {
        implementation->realizeReport(state);
    }
    void calcDecorativeGeometryAndAppend(const State& state, Stage stage,
        Array_<DecorativeGeometry>& geom) const override
    {
        implementation->calcDecorativeGeometryAndAppend(state,stage,geom);
    }
private:
    Implementation* implementation;
};

//==============================================================================
//                       GENERAL FORCE SUBSYSTEM REP
//==============================================================================
/* The implementation of a GeneralForceSubsystem_<T> for a scalar type T other
than Real. It does the bookkeeping that the Real implementation does (which
force elements are enabled, and the force elements' realize...() methods), but
none of its force evaluation, since the Real System's dynamics are only Real.
The T forces are evaluated by calcForceContributionsSum(). */
template <class T>
class GeneralForceSubsystemRep_ : public ForceSubsystem::Guts {
public:
    using StateType           = typename Force_<T>::StateType;
    using BodyForceVector     = typename Force_<T>::BodyForceVector;
    using ParticleForceVector = typename Force_<T>::ParticleForceVector;
    using MobilityForceVector = typename Force_<T>::MobilityForceVector;

    GeneralForceSubsystemRep_()
    :   ForceSubsystemRep("GeneralForceSubsystem", "0.0.1") {}

    ~GeneralForceSubsystemRep_() {
        for (int i = (int) forces.size()-1; i >= 0; --i)
            delete forces[i];
    }

    ForceIndex adoptForce(Force_<T>& force) {
        invalidateSubsystemTopologyCache();
        const ForceIndex index((int) forces.size());
        forces.push_back(new Force_<T>()); // grow
        Force_<T>& f = *forces.back(); // refer to the empty handle we just created
        force.disown(f); // transfer ownership to f
        return index;
    }

    int getNumForces() const {return forces.size();}

    const Force_<T>& getForce(ForceIndex index) const {
        assert(index >= 0 && index < forces.size());
        return *forces[index];
    }
    Force_<T>& updForce(ForceIndex index) {
        assert(index >= 0 && index < forces.size());
        return *forces[index];
    }

    bool isForceDisabled(const State& state, ForceIndex index) const {
        return !getForceEnabled(state)[index];
    }
    void setForceIsDisabled
       (State& s, ForceIndex index, bool shouldDisable) const {
        Array_<bool>& forceEnabled = Value< Array_<bool> >::updDowncast
            (updDiscreteVariable(s, forceEnabledIndex));
        forceEnabled[index] = !shouldDisable;
    }

    void calcForceContributionsSum(const StateType& s,
        const Array_<ForceIndex>& forceIndexes,
        BodyForceVector& bodyForces, MobilityForceVector& mobilityForces) const
    {
        const SimbodyMatterSubsystem& matter =
                getMultibodySystem().getMatterSubsystem();
        ForceScalarDetail::resizeAndZero(bodyForces, matter.getNumBodies());
        ForceScalarDetail::resizeAndZero(mobilityForces,
                                         matter.getNumMobilities());
        ParticleForceVector particleForces; // unused
        const State& realState = ForceScalarDetail::getRealState(s);
        for (const auto& index : forceIndexes) {
            if (!isForceDisabled(realState, index)) {
                forces[index]->getImpl().calcForce(
                    s, bodyForces, particleForces, mobilityForces);
            }
        }
    }

    GeneralForceSubsystemRep_* cloneImpl() const override
    {   return new GeneralForceSubsystemRep_(*this); }

    int realizeSubsystemTopologyImpl(State& s) const override {
        Array_<bool> forceEnabled(getNumForces());
        for (int i = 0; i < (int)forces.size(); ++i)
            forceEnabled[i] = !(forces[i]->isDisabledByDefault());
        forceEnabledIndex = allocateDiscreteVariable(s, Stage::Instance,
            new Value<Array_<bool> >(forceEnabled));
        for (int i = 0; i < (int) forces.size(); ++i)
            forces[i]->getImpl().realizeTopology(s);
        return 0;
    }
    int realizeSubsystemModelImpl(State& s) const override {
        for (int i = 0; i < (int) forces.size(); ++i)
            forces[i]->getImpl().realizeModel(s);
        return 0;
    }
    int realizeSubsystemInstanceImpl(const State& s) const override {
        forEachEnabled(s, [&](const ForceImpl_<T>& f) {f.realizeInstance(s);});
        return 0;
    }
    int realizeSubsystemTimeImpl(const State& s) const override {
        forEachEnabled(s, [&](const ForceImpl_<T>& f) {f.realizeTime(s);});
        return 0;
    }
    int realizeSubsystemPositionImpl(const State& s) const override {
        forEachEnabled(s, [&](const ForceImpl_<T>& f) {f.realizePosition(s);});
        return 0;
    }
    int realizeSubsystemVelocityImpl(const State& s) const override {
        forEachEnabled(s, [&](const ForceImpl_<T>& f) {f.realizeVelocity(s);});
        return 0;
    }
    // The T forces aren't added to the System's Real forces.
    int realizeSubsystemDynamicsImpl(const State& s) const override {
        forEachEnabled(s, [&](const ForceImpl_<T>& f) {f.realizeDynamics(s);});
        return 0;
    }
    int realizeSubsystemAccelerationImpl(const State& s) const override {
        forEachEnabled(s,
            [&](const ForceImpl_<T>& f) {f.realizeAcceleration(s);});
        return 0;
    }
    int realizeSubsystemReportImpl(const State& s) const override {
        forEachEnabled(s, [&](const ForceImpl_<T>& f) {f.realizeReport(s);});
        return 0;
    }

    // Likewise, the T potential energy isn't part of the System's Real
    // potential energy; use Force_<T>::calcPotentialEnergyContribution().
    Real calcPotentialEnergy(const State&) const override {return 0;}

    int calcDecorativeGeometryAndAppendImpl
       (const State& s, Stage stage, Array_<DecorativeGeometry>& geom) const
       override
    {
        forEachEnabled(s, [&](const ForceImpl_<T>& f)
                          {f.calcDecorativeGeometryAndAppend(s, stage, geom);});
        return 0;
    }

private:
    const Array_<bool>& getForceEnabled(const State& s) const {
        return Value< Array_<bool> >::downcast
            (getDiscreteVariable(s, forceEnabledIndex));
    }
    template <class F>
    void forEachEnabled(const State& s, F&& f) const {
        const Array_<bool>& enabled = getForceEnabled(s);
        for (int i = 0; i < (int) forces.size(); ++i)
            if (enabled[i]) f(forces[i]->getImpl());
    }

    Array_<Force_<T>*>              forces;
    mutable DiscreteVariableIndex   forceEnabledIndex;
};

//==============================================================================
//                                  FORCE
//==============================================================================
template <class T>
void Force_<T>::setDisabledByDefault(bool shouldBeDisabled)
{   this->updImpl().setDisabledByDefault(shouldBeDisabled); }
template <class T>
bool Force_<T>::isDisabledByDefault() const
{   return this->getImpl().isDisabledByDefault(); }

template <class T>
void Force_<T>::disable(State& state) const
{   getForceSubsystem().setForceIsDisabled(state, getForceIndex(), true); }
template <class T>
void Force_<T>::enable(State& state) const
{   getForceSubsystem().setForceIsDisabled(state, getForceIndex(), false); }
template <class T>
bool Force_<T>::isDisabled(const State& state) const
{   return getForceSubsystem().isForceDisabled(state, getForceIndex()); }

template <class T>
void Force_<T>::calcForceContribution(const StateType&      state,
                                      BodyForceVector&      bodyForces,
                                      ParticleForceVector&  particleForces,
                                      MobilityForceVector&  mobilityForces) const
{
    const MultibodySystem& mbs = getForceSubsystem().getMultibodySystem();
    const SimbodyMatterSubsystem& matter = mbs.getMatterSubsystem();

    // Resize if necessary, and set all forces to zero.
    ForceScalarDetail::resizeAndZero(bodyForces, matter.getNumBodies());
    ForceScalarDetail::resizeAndZero(particleForces, matter.getNumParticles());
    ForceScalarDetail::resizeAndZero(mobilityForces,
                                     matter.getNumMobilities());
    if (isDisabled(ForceScalarDetail::getRealState(state))) return;

    // Add in force element contributions.
    this->getImpl().calcForce(state,bodyForces,particleForces,mobilityForces);
}

template <class T>
T Force_<T>::calcPotentialEnergyContribution(const StateType& state) const {
    if (isDisabled(ForceScalarDetail::getRealState(state))) return T(0);
    return this->getImpl().calcPotentialEnergy(state);
}

template <class T>
const GeneralForceSubsystem_<T>& Force_<T>::getForceSubsystem() const
{   return this->getImpl().getForceSubsystem(); }
template <class T>
ForceIndex Force_<T>::getForceIndex() const
{   return this->getImpl().getForceIndex(); }

//==============================================================================
//                                  CUSTOM
//==============================================================================
template <class T>
auto Force_<T>::Custom::getImpl() const -> const CustomImpl& {
    return SimTK_DYNAMIC_CAST_DEBUG<const CustomImpl&>(Force_<T>::getImpl());
}
template <class T>
auto Force_<T>::Custom::updImpl() -> CustomImpl& {
    return SimTK_DYNAMIC_CAST_DEBUG<CustomImpl&>(Force_<T>::updImpl());
}
template <class T>
const Force_<T>& Force_<T>::Custom::upcast() const {
    return static_cast<const Force_<T>&>(*this);
}
template <class T>
Force_<T>& Force_<T>::Custom::updUpcast() {
    return static_cast<Force_<T>&>(*this);
}
template <class T>
bool Force_<T>::Custom::isInstanceOf(const Force_<T>& p) {
    return dynamic_cast<const CustomImpl*>(&p.getImpl()) != 0;
}
template <class T>
auto Force_<T>::Custom::downcast(const Force_<T>& p) -> const Custom& {
    assert(isInstanceOf(p));
    return static_cast<const Custom&>(p);
}
template <class T>
auto Force_<T>::Custom::updDowncast(Force_<T>& p) -> Custom& {
    assert(isInstanceOf(p));
    return static_cast<Custom&>(p);
}

template <class T>
Force_<T>::Custom::Custom(GeneralForceSubsystem_<T>& forces,
                          Implementation* implementation)
:   Force_<T>(new CustomImpl(implementation)) {
    updImpl().setForceSubsystem(forces, forces.adoptForce(*this));
}

template <class T>
auto Force_<T>::Custom::getImplementation() const -> const Implementation& {
    return getImpl().getImplementation();
}
template <class T>
auto Force_<T>::Custom::updImplementation() -> Implementation& {
    return updImpl().updImplementation();
}

//==============================================================================
//                         GENERAL FORCE SUBSYSTEM
//==============================================================================
template <class T>
bool GeneralForceSubsystem_<T>::isInstanceOf(const ForceSubsystem& s) {
    return dynamic_cast<const GeneralForceSubsystemRep_<T>*>(&s.getRep())
           != nullptr;
}
template <class T>
auto GeneralForceSubsystem_<T>::downcast(const ForceSubsystem& s)
    -> const GeneralForceSubsystem_& {
    assert(isInstanceOf(s));
    return static_cast<const GeneralForceSubsystem_&>(s);
}
template <class T>
auto GeneralForceSubsystem_<T>::updDowncast(ForceSubsystem& s)
    -> GeneralForceSubsystem_& {
    assert(isInstanceOf(s));
    return static_cast<GeneralForceSubsystem_&>(s);
}

template <class T>
const GeneralForceSubsystemRep_<T>& GeneralForceSubsystem_<T>::getRep() const {
    return SimTK_DYNAMIC_CAST_DEBUG<const GeneralForceSubsystemRep_<T>&>
        (ForceSubsystem::getRep());
}
template <class T>
GeneralForceSubsystemRep_<T>& GeneralForceSubsystem_<T>::updRep() {
    return SimTK_DYNAMIC_CAST_DEBUG<GeneralForceSubsystemRep_<T>&>
        (ForceSubsystem::updRep());
}

// Create Subsystem but don't associate it with any System. This isn't much
// use except for making std::vector's, which require a default constructor
// to be available.
template <class T>
GeneralForceSubsystem_<T>::GeneralForceSubsystem_()
:   ForceSubsystem()
{   this->adoptSubsystemGuts(new GeneralForceSubsystemRep_<T>()); }

template <class T>
GeneralForceSubsystem_<T>::GeneralForceSubsystem_(MultibodySystem& mbs)
:   ForceSubsystem()
{   this->adoptSubsystemGuts(new GeneralForceSubsystemRep_<T>());
    mbs.addForceSubsystem(*this); } // steal ownership

template <class T>
ForceIndex GeneralForceSubsystem_<T>::adoptForce(Force_<T>& force)
{   return updRep().adoptForce(force); }

template <class T>
int GeneralForceSubsystem_<T>::getNumForces() const
{   return getRep().getNumForces(); }

template <class T>
const Force_<T>& GeneralForceSubsystem_<T>::getForce(ForceIndex index) const
{   return getRep().getForce(index); }

template <class T>
Force_<T>& GeneralForceSubsystem_<T>::updForce(ForceIndex index)
{   return updRep().updForce(index); }

template <class T>
bool GeneralForceSubsystem_<T>::isForceDisabled
   (const State& state, ForceIndex index) const
{   return getRep().isForceDisabled(state, index); }

template <class T>
void GeneralForceSubsystem_<T>::setForceIsDisabled
   (State& state, ForceIndex index, bool disabled) const
{   getRep().setForceIsDisabled(state, index, disabled); }

template <class T>
void GeneralForceSubsystem_<T>::setNumberOfThreads(unsigned numThreads)
    requires std::is_same_v<T, Real>
{   updRep().setNumberOfThreads(numThreads); }

template <class T>
int GeneralForceSubsystem_<T>::getNumberOfThreads() const
    requires std::is_same_v<T, Real>
{   return getRep().getNumberOfThreads(); }

template <class T>
void GeneralForceSubsystem_<T>::calcForceContributionsSum(
    const StateType& s, const Array_<ForceIndex>& forceIndexes,
    BodyForceVector& bodyForces, MobilityForceVector& mobilityForces) const
{
    getRep().calcForceContributionsSum(
        s, forceIndexes, bodyForces, mobilityForces);
}

template <class T>
const MultibodySystem& GeneralForceSubsystem_<T>::getMultibodySystem() const
{   return MultibodySystem::downcast(this->getSystem()); }

} // namespace SimTK

/** Instantiate the force element and general force subsystem templates for
the scalar type T. Write this once, at global scope, in a translation unit that
includes this header. **/
#define SimTK_INSTANTIATE_FORCE_SCALAR(T)                                      \
template class SimTK::PIMPLHandle<SimTK::Force_<T>, SimTK::ForceImpl_<T>,      \
                                  true>;                                       \
template class SimTK::PIMPLImplementation<SimTK::Force_<T>,                    \
                                          SimTK::ForceImpl_<T>>;               \
template class SimTK::Force_<T>;                                               \
template class SimTK::GeneralForceSubsystem_<T>;                               \
template class SimTK::GeneralForceSubsystemRep_<T>

#endif // SimTK_SIMBODY_FORCE_SCALAR_IMPL_H_
