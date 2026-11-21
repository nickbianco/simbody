#ifndef SimTK_SIMBODY_TREE_SWEEPS_H_
#define SimTK_SIMBODY_TREE_SWEEPS_H_

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
Ordered sweeps over the multibody tree that are shared by the Real operators
(SimbodyMatterSubsystemRep) and those for other scalar types
(MatterScalarRep<T> in MultibodyScalarImpl.h). Each caller finds its own
caches and provides its own temporaries, then calls these, which visit the
RigidBodyNodes in the right order. \a levels is the tree's nodes by level,
from Ground outward: a sequence of sequences of node pointers. **/

#include "simbody/internal/tree/RigidBodyNode.h"

namespace SimTK {

/** Ma = M a, with temporaries A_GB and F (one per body). Ma (of length nu)
may be the same array as a, since a is read before Ma is written. **/
template <class T, class Levels>
void multiplyByMSweeps(const Levels&                   levels,
                       const SBTreePositionCache_<T>&  tpc,
                       const T*                        a,
                       SpatialVec_<T>*                 A_GB,
                       SpatialVec_<T>*                 F,
                       T*                              Ma) {
    for (int i=0; i < (int)levels.size(); ++i)
        for (int j=0; j < (int)levels[i].size(); ++j)
            levels[i][j]->multiplyByMPass1Outward(tpc, a, A_GB);
    for (int i=(int)levels.size()-1; i >= 0; --i)
        for (int j=0; j < (int)levels[i].size(); ++j)
            levels[i][j]->multiplyByMPass2Inward(tpc, A_GB, F, Ma);
}

/** MInvf = M^-1 f, with temporaries eps (one per mobility) and z, zPlus,
and A_GB (one per body). The articulated-body inertias must have been
realized. MInvf (of length nu) may be the same array as f, since f is read
before MInvf is written. **/
template <class T, class Levels>
void multiplyByMInvSweeps(const Levels&                             levels,
                          const SBInstanceCache_<T>&                ic,
                          const SBTreePositionCache_<T>&            tpc,
                          const SBArticulatedBodyInertiaCache_<T>&  abc,
                          const T*                                  f,
                          T*                                        eps,
                          SpatialVec_<T>*                           z,
                          SpatialVec_<T>*                           zPlus,
                          SpatialVec_<T>*                           A_GB,
                          T*                                        MInvf) {
    for (int i=(int)levels.size()-1; i >= 0; --i)
        for (int j=0; j < (int)levels[i].size(); ++j)
            levels[i][j]->multiplyByMInvPass1Inward(ic, tpc, abc, f, z,
                                                    zPlus, eps);
    for (int i=0; i < (int)levels.size(); ++i)
        for (int j=0; j < (int)levels[i].size(); ++j)
            levels[i][j]->multiplyByMInvPass2Outward(ic, tpc, abc, eps,
                                                     A_GB, MInvf);
}

} // namespace SimTK

#endif // SimTK_SIMBODY_TREE_SWEEPS_H_
