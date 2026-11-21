#ifndef SimTK_SimTKCOMMON_FUNCTION_EVALUATOR_H_
#define SimTK_SimTKCOMMON_FUNCTION_EVALUATOR_H_

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
Declares FunctionEvaluator<T>, the customization point for evaluating a (Real)
Function at arguments of another scalar type T. **/

#include "SimTKcommon/internal/Function.h"

#include <vector>

namespace SimTK {

/** Evaluates a (Real) Function, and its partial derivatives, at arguments of
a scalar type T other than Real. A Function's methods take Real arguments, so
computations with T that use a Function (such as those of a
Constraint::CoordinateCoupler) need a way to evaluate it with T arguments that
depends on T. The project that defines the traits for T (see
SimTKcommon/internal/RealScalarType.h) can provide it by specializing this
template, before instantiating Simbody's templates for T:
@code
namespace SimTK {
template <> struct FunctionEvaluator<MyScalar> {
    // The value of f at x.
    static MyScalar calcValue(const Function& f,
                              const std::vector<MyScalar>& x);
    // A partial derivative of f at x, as in Function::calcDerivative().
    static MyScalar calcDerivative(const Function& f,
                                   const std::vector<int>& derivComponents,
                                   const std::vector<MyScalar>& x);
};
}
@endcode
For example, a forward-mode automatic differentiation type can apply the chain
rule using the Function's derivatives, and a symbolic type such as
casadi::SX can wrap the Function as an external function. Either way, the
derivatives of the result with respect to x are only available to the order
that the Function provides (Function::getMaxDerivativeOrder()), so
computations that use the Function's derivatives have correct derivatives
only to correspondingly lower order.

The primary template is empty: features that need it are unavailable for a
scalar type that doesn't specialize it. **/
template <class T>
struct FunctionEvaluator {};

/** Whether FunctionEvaluator has been specialized for scalar type T. **/
template <class T>
inline constexpr bool hasFunctionEvaluator = requires
   (const Function& f, const std::vector<int>& c, const std::vector<T>& x) {
    {FunctionEvaluator<T>::calcValue(f, x)};
    {FunctionEvaluator<T>::calcDerivative(f, c, x)};
};

} // namespace SimTK

#endif // SimTK_SimTKCOMMON_FUNCTION_EVALUATOR_H_
