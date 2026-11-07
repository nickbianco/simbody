#ifndef SimTK_SIMBODY_FORCE_TYPES_H_
#define SimTK_SIMBODY_FORCE_TYPES_H_

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
Declarations of the force element and general force subsystem templates on
the scalar type T, the Real typedefs that keep their usual names (Force,
GeneralForceSubsystem), and the types T is evaluated with. Include this
instead of writing forward declarations of these classes. **/

#include "SimTKcommon.h"
#include "simbody/internal/common.h"

#include <vector>

namespace SimTK {

template <class T> class Force_;
template <class T> class ForceImpl_;
template <class T> class GeneralForceSubsystem_;
template <class T> class GeneralForceSubsystemRep_;

/** The force element type used by Simbody's ordinary (Real) computations. **/
using Force                    = Force_<Real>;
using ForceImpl                = ForceImpl_<Real>;
/** The general force subsystem used by Simbody's ordinary (Real)
computations. **/
using GeneralForceSubsystem    = GeneralForceSubsystem_<Real>;
using GeneralForceSubsystemRep = GeneralForceSubsystemRep_<Real>;

/** The state and force vector types that force elements with scalar type T
are evaluated with. For Real these are Simbody's usual State and Vector types.
For any other T (such as casadi::SX), the time and continuous variables are in
a ScalarState<T>, and forces are std::vectors, because Vector_ supports only
the built-in scalar types. (These would go away if State were templatized.) **/
template <class T>
struct ForceScalarTypes {
    using StateType           = ScalarState<T>;
    using BodyForceVector     = std::vector<SpatialVec_<T>>;
    using ParticleForceVector = std::vector<Vec<3,T>>;
    using MobilityForceVector = std::vector<T>;
};

template <>
struct ForceScalarTypes<Real> {
    using StateType           = State;
    using BodyForceVector     = Vector_<SpatialVec>;
    using ParticleForceVector = Vector_<Vec3>;
    using MobilityForceVector = Vector;
};

} // namespace SimTK

#endif // SimTK_SIMBODY_FORCE_TYPES_H_
