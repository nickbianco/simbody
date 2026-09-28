/* -------------------------------------------------------------------------- *
 *                               Simbody(tm)                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the SimTK biosimulation toolkit originating from           *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org/home/simbody.  *
 *                                                                            *
 * Portions copyright (c) 2005-15 Stanford University and the Authors.        *
 * Authors: Michael Sherman                                                   *
 * Contributors: Derived from IVM code written by Charles Schwieters          *
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

/* This file contains implementations of the base class methods for the
templatized class RigidBodyNodeSpec_<P, dof, noR_FM> for the default Real
precision, for all possible values of the arguments. The definitions are in
RigidBodyNodeSpecDefs.h. */

#include "SimbodyMatterSubsystemRep.h"
#include "simbody/internal/tree/RigidBodyNode.h"
#include "simbody/internal/tree/RigidBodyNodeSpec.h"
#include "simbody/internal/tree/RigidBodyNodeSpecDefs.h"

#include "simbody/internal/tree/RigidBodyNodeSpec_Pin.h"
#include "simbody/internal/tree/RigidBodyNodeSpec_Slider.h"
#include "RigidBodyNodeSpec_Ball.h"
#include "RigidBodyNodeSpec_Free.h"
#include "RigidBodyNodeSpec_Custom.h"

    ////////////////////
    // INSTANTIATIONS //
    ////////////////////

template class RigidBodyNodeSpec_<Real, 1, false>;
template class RigidBodyNodeSpec_<Real, 2, false>;
template class RigidBodyNodeSpec_<Real, 3, false>;
template class RigidBodyNodeSpec_<Real, 4, false>;
template class RigidBodyNodeSpec_<Real, 5, false>;
template class RigidBodyNodeSpec_<Real, 6, false>;
template class RigidBodyNodeSpec_<Real, 1, true>;
template class RigidBodyNodeSpec_<Real, 3, true>;
