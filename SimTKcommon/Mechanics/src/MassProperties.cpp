/* -------------------------------------------------------------------------- *
 *                       Simbody(tm): SimTKcommon                             *
 * -------------------------------------------------------------------------- *
 * This is part of the SimTK biosimulation toolkit originating from           *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org/home/simbody.  *
 *                                                                            *
 * Portions copyright (c) 2005-12 Stanford University and the Authors.        *
 * Authors: Michael Sherman                                                   *
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
 * Implementations of non-inline methods of MassProperties classes.
 */

#include "SimTKcommon/internal/common.h"
#include "SimTKcommon/internal/MassProperties.h"

#include <iostream>

namespace SimTK {
    /////////////////////////
    //       INERTIA       //
    /////////////////////////

// Instantiate so we catch bugs now.
template class Inertia_<float>;
template class Inertia_<double>;

    /////////////////////////
    //     UNIT INERTIA    //
    /////////////////////////

// Instantiate so we catch bugs now.
template class UnitInertia_<float>;
template class UnitInertia_<double>;

    /////////////////////////
    //   MASS PROPERTIES   //
    /////////////////////////

// Instantiate so we catch bugs now.
template class MassProperties_<float>;
template class MassProperties_<double>;

    /////////////////////////
    //   SPATIAL INERTIA   //
    /////////////////////////

// Instantiate so we catch bugs now.
template class SpatialInertia_<float>;
template class SpatialInertia_<double>;

    /////////////////////////
    // ARTICULATED INERTIA //
    /////////////////////////

// ArticulatedInertia_::shift() and shiftInPlace() are defined inline in
// MassProperties.h so that they are available for any scalar type.

// Instantiate so we catch bugs now.
template class ArticulatedInertia_<float>;
template class ArticulatedInertia_<double>;



} // namespace SimTK

