#ifndef SimTK_SIMBODY_RIGID_BODY_NODE_FACTORY_H_
#define SimTK_SIMBODY_RIGID_BODY_NODE_FACTORY_H_

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
A visitor used to create the RigidBodyNodes of a SimbodyMatterSubsystem for a
scalar type other than Real, when a ScalarState<T> is realized through
Stage::Instance (see MultibodySystem::realize(const ScalarState<T>&, Stage)).

For Real, each MobilizedBody implementation creates its own RigidBodyNode in
MobilizedBodyImpl::createRigidBodyNode(). That can't be done for an arbitrary
scalar type T, since a virtual method can't be a template. Instead,
SimbodyMatterSubsystem::createRigidBodyNodes() gives a RigidBodyNodeFactory
the subsystem's Model and Instance stage information, and then visits each
mobilized body in order of MobilizedBodyIndex (so parents before children).
Each MobilizedBody implementation calls the factory method for its mobilizer
type. A factory for a particular scalar type (see RigidBodyNodeFactory_<T> in
MultibodyScalarImpl.h) overrides the methods for the mobilizers that it
supports; the default implementations throw an exception. **/

#include "simbody/internal/common.h"
#include "SimbodyTreeState.h"

namespace SimTK {

/** The information about one mobilized body needed to create its
RigidBodyNode. The slots are those of the Real RigidBodyNode. **/
struct RigidBodyNodeInfo {
    MobilizedBodyIndex  index;
    MobilizedBodyIndex  parent;         ///< invalid for Ground
    int                 level = 0;      ///< distance from Ground
    MassProperties      massProperties; ///< in the body frame, about its origin
    bool                isReversed = false;
    UIndex              uIndex;
    USquaredIndex       uSqIndex;
    QIndex              qIndex;
};

/** Abstract visitor that creates RigidBodyNodes; see the file comment. **/
class RigidBodyNodeFactory {
public:
    virtual ~RigidBodyNodeFactory() = default;

    /** Called first, with the subsystem's topology and its Model and Instance
    stage variables and cache entries. These refer to the Real State passed to
    SimbodyMatterSubsystem::createRigidBodyNodes(); copy what's needed. **/
    virtual void beginTree(const SBTopologyCache&   topology,
                           const SBModelVars&       modelVars,
                           const SBModelCache&      modelCache,
                           const SBInstanceVars&    instanceVars,
                           const SBInstanceCache&   instanceCache) = 0;
    /** Called last, after every mobilized body has been visited. **/
    virtual void endTree() {}

    // One method per built-in mobilizer type. Mobilizers with additional
    // parameters (e.g. the pitch of a Screw) will get additional arguments
    // when they are supported for other scalar types.
    virtual void createGround(const RigidBodyNodeInfo& i)   {unsupported("Ground", i);}
    virtual void createWeld(const RigidBodyNodeInfo& i)     {unsupported("Weld", i);}
    virtual void createPin(const RigidBodyNodeInfo& i)      {unsupported("Pin", i);}
    virtual void createSlider(const RigidBodyNodeInfo& i)   {unsupported("Slider", i);}
    virtual void createUniversal(const RigidBodyNodeInfo& i){unsupported("Universal", i);}
    virtual void createCylinder(const RigidBodyNodeInfo& i) {unsupported("Cylinder", i);}
    virtual void createBendStretch(const RigidBodyNodeInfo& i)
    {   unsupported("BendStretch", i); }
    virtual void createPlanar(const RigidBodyNodeInfo& i)   {unsupported("Planar", i);}
    virtual void createSphericalCoords(const RigidBodyNodeInfo& i)
    {   unsupported("SphericalCoords", i); }
    virtual void createGimbal(const RigidBodyNodeInfo& i)   {unsupported("Gimbal", i);}
    virtual void createBushing(const RigidBodyNodeInfo& i)  {unsupported("Bushing", i);}
    virtual void createBall(const RigidBodyNodeInfo& i)     {unsupported("Ball", i);}
    virtual void createEllipsoid(const RigidBodyNodeInfo& i){unsupported("Ellipsoid", i);}
    virtual void createTranslation(const RigidBodyNodeInfo& i)
    {   unsupported("Translation", i); }
    virtual void createFree(const RigidBodyNodeInfo& i)     {unsupported("Free", i);}
    virtual void createLineOrientation(const RigidBodyNodeInfo& i)
    {   unsupported("LineOrientation", i); }
    virtual void createFreeLine(const RigidBodyNodeInfo& i) {unsupported("FreeLine", i);}
    virtual void createScrew(const RigidBodyNodeInfo& i)    {unsupported("Screw", i);}
    virtual void createCantileverFreeBeam(const RigidBodyNodeInfo& i)
    {   unsupported("CantileverFreeBeam", i); }
    virtual void createCustom(const RigidBodyNodeInfo& i)   {unsupported("Custom", i);}

protected:
    /** Throw an exception saying that this factory doesn't support the given
    mobilizer type. **/
    void unsupported(const char* mobilizer,
                     const RigidBodyNodeInfo& info) const {
        SimTK_ERRCHK2_ALWAYS(false, "RigidBodyNodeFactory",
            "MobilizedBody %d is a %s mobilizer, which is not yet supported "
            "for this scalar type.", (int)info.index, mobilizer);
    }
};

} // namespace SimTK

#endif // SimTK_SIMBODY_RIGID_BODY_NODE_FACTORY_H_
