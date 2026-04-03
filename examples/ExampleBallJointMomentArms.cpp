/*-----------------------------------------------------------------------------
                Simbody(tm) Example: Moment Arms when qdot != u
-------------------------------------------------------------------------------
 Copyright (c) 2026 Authors.
 Authors: Nicholas Bianco

 Licensed under the Apache License, Version 2.0 (the "License"); you may
 not use this file except in compliance with the License. You may obtain a
 copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ----------------------------------------------------------------------------*/

#include "Simbody.h"
#include <cassert>
#include <fstream>
#include <vector>
using namespace SimTK;

// This example represents an application where it would be valuable to have
// methods for calculating moment arms across mobilizers where qdot != u. We
// first simulate a 3D pendulum (mobilized by a MobilizedBody::Ball) driven by
// an actuator whose tension is applied along a path defined by a CableSpan.
// Then, we use least-squares to fit a set of multivariate polynomial
// coefficients to the path lengths and moment arms calculated from the states
// trajectory from the simulation. Finally, we create a new pendulum system that
// is actuated based on the polynomial functions we created and verify that
// the two systems produce equal inverse dynamics moments.
//
// The main takeaway from this example is the use of the N matrix defining the
// kinematical differential equations for the system. For a ball joint,
// N != I, whether using Euler angles or quaternions to represent the
// orientation.
//
// We will have a function f(q) that gives us the path length of the cable as a
// function of the generalized coordinates q:
//
// l = f(q)
//
// In this example, we use choose to calculate moment arms based on path speed
// formulation described in Sherman et al. (2023)
//
// rᵢ = l̇ / q̇ᵢ
//
// Since q̇ != u, we need to use the N matrix to calculate q̇ from u. First, we
// set q̇ for the coordinate of interest to 1 and all other coordinates to 0:
//
// q̇= 0
// q̇ᵢ = 1
//
// Then we calculate u from q̇ using the N matrix:
//
// u = N⁻¹ q̇
//
// Finally, we calculate the moment arm for coordinate i as the path speed
// divided by q̇ᵢ:
//
// l̇ <- q, u
// rᵢ = l̇ / q̇ᵢ
//
// We can also use the transpose of N to compute the contribution of a scalar
// tension, T, to the the mobility forces, fu, for a function based path:
//
// fq = T [r₀, r₁, r₂, ... ]
// fu = Nᵀ fq
//
// While this won't produce the same body forces as the original system, we will
// show in this example that we can produce the same inverse dynamics moments as
// the original cable-actuated system.

// The code for representing multivariate polynomials is based on OpenSim's
// MultivariatePolynomial class:
// https://github.com/opensim-org/opensim-core/blob/main/OpenSim/Common/MultivariatePolynomialFunction.cpp

// The code for estimating polynomial coefficients is based on OpenSim's
// PolynomialPathFitter class:
// https://github.com/opensim-org/opensim-core/blob/main/OpenSim/Actuators/PolynomialPathFitter.cpp

//=============================================================================
//                          MULTIVARIATE POLYNOMIAL
//=============================================================================
/**
 * A helper class to construct and manipulate multivariate polynomials using
 * symbolic operations.
 *
 * A multivariate polynomial is represented as a map of monomials to
 * coefficients. Monomials are represented as maps of variable names to
 * exponents. For example, the polynomial
 *
 * f = 0.1 x^2 y + 0.7 y^3 z + 0.5 x z^5
 *
 * can be constructed represented as
 *
 * @code
 * MultivariatePolynomial f;
 * Monomial m0 = {{"x", 2}, {"y", 1}};
 * Monomial m1 = {{"y", 3}, {"z", 1}};
 * Monomial m2 = {{"x", 1}, {"z", 5}};
 * f[m0] = 0.1;
 * f[m1] = 0.7;
 * f[m2] = 0.5;
 * @endcode
 */
using Monomial = std::map<std::string, int>;
class MultivariatePolynomial : public std::map<Monomial, double> {

public:
    MultivariatePolynomial() = default;

    /**
     * Construct a multivariate polynomial of a given dimension and order with
     * variables named according to the provided vector of strings, `var`. All
     * possible monomials of the polynomial are generated  in the same order as
     * `MultivariatePolynomialFunction` and assigned a coefficient corresponding
     * to the index of the monomial in the polynomial.
     */
    MultivariatePolynomial(int dimension, int order,
            const std::vector<std::string>& vars) {

        assert(static_cast<int>(vars.size()) == dimension);

        int numCoefficients = choose(dimension + order, order);
        std::vector<Monomial> monomials;
        monomials.resize(numCoefficients);

        std::vector<std::vector<int>> combinations;
        generateCombinations(dimension, order, combinations);
        for (int i = 0; i < numCoefficients; ++i) {
            Monomial monomial;
            for (int j = 0; j < dimension; ++j) {
                if (combinations[i][j] > 0) {
                    monomial[vars[j]] = combinations[i][j];
                }
            }
            monomials.at(i) = monomial;
        }

        for (int i = 0; i < numCoefficients; ++i) {
            (*this)[monomials[i]] = i;
        }
    }

    // GET METHODS
    /**
     * Get the coefficient of a monomial in the polynomial. If the monomial is
     * not in the polynomial, the coefficient is zero.
     */
    double getCoefficient(const Monomial& monomial) const {
        auto it = find(monomial);
        if (it != end()) {
            return it->second;
        }
        return 0.0;
    }

    // CALC METHODS
    /**
     * Calculate the order of the polynomial.
     */
    int calcOrder() const {
        int order = 0;
        // Iterate through all monomials.
        for (auto it = begin(); it != end(); it++) {
            // For this monomial, calculate the sum of the exponents of all
            // variables.
            const Monomial& monomial = it->first;
            int sum = 0;
            for (auto itv = monomial.begin(); itv != monomial.end(); itv++) {
                sum += itv->second;
            }
            // Update the order if the sum is greater than the current order.
            if (sum > order) {
                order = sum;
            }
        }
        return order;
    }

    /**
     * Calculate the vector of coefficients of the polynomial if it had a given
     * order and dimension. This vector matches the coefficients vector of
     * MultivariatePolynomialFunction; therefore it will contain coefficients
     * for all possible monomials in the same order as a
     * MultivariatePolynomialFunction. If the MultivariatePolynomial does not
     * contain a monomial for a corresponding slot in the coefficients vector,
     * coefficient is set to zero.
     */
    Vector calcCoefficients(int dimension, int order,
            const std::vector<std::string>& vars) const {

        assert(static_cast<int>(vars.size()) == dimension);

        // Create a temporary polynomial with the same dimension and order as
        // the desired polynomial.
        MultivariatePolynomial temp(dimension, order, vars);
        // Initialize the coefficients to zero.
        int numCoefficients = choose(dimension + order, order);
        Vector coefficients(numCoefficients, 0.0);
        // If the current polynomial contains a monomial that matches a monomial
        // in the temporary polynomial, set its coefficient in the full
        // coefficients vector.
        for (auto it2 = temp.begin(); it2 != temp.end(); ++it2) {
            const Monomial& monomial = it2->first;
            int index = static_cast<int>(it2->second);
            coefficients[index] = getCoefficient(monomial);
        }

        return coefficients;
    }

    // POLYNOMIAL OPERATIONS
    /**
     * Get a new polynomial that is the first derivative of the current
     * polynomial with respect to the specified variable.
     */
    MultivariatePolynomial getDerivative(const std::string& var) const {
        const MultivariatePolynomial& poly = *this;
        MultivariatePolynomial deriv;
        // Iterate through all monomials in the polynomial.
        for (auto itp = poly.begin(); itp != poly.end(); itp++) {
            Monomial monomial = itp->first;
            double coeff = getCoefficient(monomial);
            // Iterate through all variables in the monomial.
            for (auto itm = monomial.begin(); itm != monomial.end(); itm++) {
                Monomial derivMonomial = monomial;
                // If the variable is in the monomial, take the derivative.
                if (itm->first == var) {
                    // If the exponent is 1, remove the variable from the
                    // monomial. Otherwise, reduce the exponent by 1.
                    int exponent = itm->second;
                    derivMonomial.erase(itm->first);
                    if (exponent > 1) {
                        derivMonomial[itm->first] = exponent - 1;
                    }
                    // Add the derivative term to the derivative polynomial.
                    // The coefficient is the product of the original
                    // coefficient and previous exponent of the monomial.
                    deriv[derivMonomial] = coeff * exponent;
                }
            }
        }
        return deriv;
    }

    /**
     * Given a variable, factor the current polynomial into two polynomials: one
     * that contains all the terms did not contain the variable, and one that
     * contains all the terms that did contain the variable with the variable
     * factored out. The former polynomial is returned as the first element of
     * the pair, and the latter polynomial is returned as the second element.
     *
     * The variable is factored out to only the first order. For example, if the
     * variable is X, the second polynomial X^3 + X^2 Y would be factored as
     * X (X^2 + X Y) (the first polynomial here is 0).
     */
    std::pair<MultivariatePolynomial, MultivariatePolynomial>
    factorVariable(const std::string& var) const {
        // All the terms that do not contain the variable.
        MultivariatePolynomial left;
        // All the terms that contain the variable.
        MultivariatePolynomial right;

        const MultivariatePolynomial& poly = *this;
        for (auto it = poly.begin(); it != poly.end(); it++) {
            Monomial monomial = it->first;
            // If the variable is not in the monomial, add the term to the left
            // polynomial. Otherwise, factor out the variable and add the term
            // to the right polynomial.
            double coeff = getCoefficient(monomial);
            if (monomial.find(var) == monomial.end()) {
                left[monomial] = coeff;
            } else {
                // Factor out the variable.
                monomial[var] -= 1;
                // Remove the variable if the exponent is zero.
                if (monomial[var] == 0) {
                    monomial.erase(var);
                }
                // Add the term to the right polynomial.
                right[monomial] = coeff;
            }
        }
        return std::make_pair(left, right);
    }

    // HELPER METHODS
    /**
     * Generate all possible combinations of `k` elements from a set of `n`
     * total elements.
     */
    static int choose(int n, int k) {
        if (k == 0) { return 1; }
        return (n * choose(n - 1, k - 1)) / k;
    }

    /**
     * Generate all possible combinations of powers representing the terms of a
     * multivariate polynomial. Each combination is represented as a vector of
     * integers, where the `i`-th element is the power of the `i`-th variable.
     * For example, the combination (2, 1, 0) represents the term X^2 Y Z.
     */
    static void generateCombinations(int dimension, int order,
            std::vector<std::vector<int>>& combinations) {
        combinations.clear();
        generateCombinations(std::vector<int>(), 0, 0,
                dimension, order, combinations);
    }

private:
    static void generateCombinations(std::vector<int> current, int level,
            int sum, int dimension, int order,
            std::vector<std::vector<int>>& combinations) {
        if (level == dimension) {
            if (sum <= order) {
                combinations.push_back(current);
            }
            return;
        }
        for (int i = 0; i <= order - sum; ++i) {
            current.push_back(i);
            generateCombinations(current, level + 1, sum + i,
                    dimension, order, combinations);
            current.pop_back();
        }
    }
};

//=============================================================================
//                      MULTIVARIATE POLYNOMIAL FUNCTION
//=============================================================================
/**
 * A multivariate polynomial function.
 *
 * This implementation allows computation of first-order derivatives only.
 *
 * For a third-order polynomial that is a function of three components
 * (X, Y, Z), the order is as follows:
 *
 * <pre>
 * Index | X  Y  Z
 * 0     | 0  0  0
 * 1     | 0  0  1
 * 2     | 0  0  2
 * 3     | 0  0  3
 * 4     | 0  1  0
 * 5     | 0  1  1
 * 6     | 0  1  2
 * 7     | 0  2  0
 * 8     | 0  2  1
 * 9     | 0  3  0
 * 10    | 1  0  0
 * 11    | 1  0  1
 * 12    | 1  0  2
 * 13    | 1  1  0
 * 14    | 1  1  1
 * 15    | 1  2  0
 * 16    | 2  0  0
 * 17    | 2  0  1
 * 18    | 2  1  0
 * 19    | 3  0  0
 * </pre>
 * Assuming c6 the index 6 coefficient, the corresponding term is Y Z^2.
 *
 * @param dimension the number of independent components
 * @param order the polynomial order (the largest sum of exponents in a single
 *        term)
 * @param coefficients the polynomial coefficients in order of ascending
 *        powers starting from the last independent component.
 *
 * @note The order of coefficients for this class is the opposite from the
 *       order used in the univariate PolynomialFunction.
 */
template <class T>
class MultivariatePolynomialFunction : public Function_<T> {
public:
    MultivariatePolynomialFunction(const Vector_<T>& coefficients,
            const int& dimension, const int& order)
            : m_coefficients(coefficients), m_dimension(dimension),
              m_order(order) {

        assert(dimension >= 0);
        assert(order >= 0);

        int numCoefs = MultivariatePolynomial::choose(dimension + order, order);
        assert(static_cast<int>(coefficients.size()) == numCoefs);

        MultivariatePolynomial::generateCombinations(m_dimension, m_order,
                m_combinations);

        // Construct a Horner's scheme for the polynomial.
        MultivariatePolynomial poly;
        std::vector<std::string> vars;
        for (int i = 0; i < dimension; ++i) {
            vars.push_back("x" + std::to_string(i));
        }
        for (int i = 0; i < numCoefs; ++i) {
            if (m_coefficients[i] != 0) {
                Monomial monomial;
                for (int j = 0; j < dimension; ++j) {
                    if (m_combinations[i][j] > 0) {
                        monomial[vars[j]] = m_combinations[i][j];
                    }
                }
                poly[monomial] = m_coefficients[i];
            }
        }
        m_value = createHornerScheme(vars, poly);

        // Construct a Horner's scheme for each polynomial derivatives.
        for (int i = 0; i < dimension; ++i) {
            std::string var = "x" + std::to_string(i);
            MultivariatePolynomial deriv = poly.getDerivative(var);
            m_derivatives.push_back(createHornerScheme(vars, deriv));
        }

    }

    T calcValue(const Vector& x) const override {
        return m_value->calcValue(x);
    }

    T calcDerivative(const Array_<int>& derivComponent,
            const Vector& x) const override {
        return m_derivatives[derivComponent[0]]->calcValue(x);
    }

    Vector_<T> calcMonomialValues(const Vector& x) const {
        Vector_<T> values(m_coefficients.size(), T(0.0));
        for (int i = 0; i < m_coefficients.size(); ++i) {
            T value = static_cast<T>(1);
            for (int j = 0; j < m_dimension; ++j) {
                value *= std::pow(x[j], m_combinations[i][j]);
            }
            values[i] = value;
        }

        return values;
    }

    Vector_<T> calcMonomialDerivatives(
            const Array_<int>& derivComponent,
            const Vector& x) const {
        Vector_<T> derivatives(m_coefficients.size(), T(0.0));
        for (int i = 0; i < m_coefficients.size(); ++i) {
            if (m_combinations[i][derivComponent[0]] > 0) {
                T deriv = static_cast<T>(1);
                for (int j = 0; j < m_dimension; ++j) {
                    if (j == derivComponent[0]) {
                        deriv *= m_combinations[i][j];
                        deriv *= std::pow(x[j], m_combinations[i][j] - 1);
                    } else {
                        deriv *= std::pow(x[j], m_combinations[i][j]);
                    }
                }
                derivatives[i] = deriv;
            }
        }
        return derivatives;
    }

    int getArgumentSize() const override { return m_dimension; }
    int getMaxDerivativeOrder() const override { return 1; }
    MultivariatePolynomialFunction* clone() const override {
        return new MultivariatePolynomialFunction(*this);
    }

private:
    Vector_<T> m_coefficients;
    int m_dimension;
    int m_order;
    std::vector<std::vector<int>> m_combinations;

    // HORNER'S SCHEME

    // A base class representing a node for a binary tree representation of
    // Horner's scheme for a multivariate polynomial.
    class HornerSchemeNode {
        public:
            virtual ~HornerSchemeNode() = default;
            virtual T calcValue(const Vector_<T>& x) const = 0;

            // ClonePtr requires a clone method.
            virtual HornerSchemeNode* clone() const = 0;
    };

    // A node representing a polynomial factorization. The value of the node is
    // the sum of the left node and the product of the right node and the
    // variable at the specified index.
    class FactorNode : public HornerSchemeNode {
        public:
            FactorNode(ClonePtr<HornerSchemeNode> left,
                    ClonePtr<HornerSchemeNode> right, int index) :
                    m_left(std::move(left)), m_right(std::move(right)),
                    m_index(index) {}

            T calcValue(const Vector_<T>& x) const override {
                return m_left->calcValue(x) +
                        x[m_index] * m_right->calcValue(x);
            }

            HornerSchemeNode* clone() const override {
                return new FactorNode(
                    ClonePtr<HornerSchemeNode>(m_left->clone()),
                    ClonePtr<HornerSchemeNode>(m_right->clone()),
                    m_index);
            }
        private:
            ClonePtr<HornerSchemeNode> m_left;
            ClonePtr<HornerSchemeNode> m_right;
            int m_index;
    };

    // A node representing a monomial. The value of the node is the constant
    // term plus the sum of the product of the polynomial coefficients and the
    // variables.
    class MonomialNode : public HornerSchemeNode {
        public:
            MonomialNode(T constant, const Vector_<T>& coefficients) :
                    m_constant(constant), m_coefficients(coefficients),
                    m_numCoefficients(coefficients.size()) {}

            T calcValue(const Vector_<T>& x) const override {
                T result = m_constant;
                for (int i = 0; i < m_numCoefficients; ++i) {
                    result += m_coefficients[i] * x[i];
                }
                return result;
            }

            HornerSchemeNode* clone() const override {
                return new MonomialNode(m_constant, m_coefficients);
            }
        private:
            T m_constant;
            Vector_<T> m_coefficients;
            int m_numCoefficients;
    };

    // Construct Horner's scheme for a multivariate polynomial. Uses the
    // "greedy" algorithm described in Ceberio and Kreinovich (2004), "Greedy
    // Algorithms for Optimizing Multivariate Horner Schemes". The algorithm is
    // represented by a binary tree where each successive node is constructed
    // by factoring out the variable that appears in the most terms of the
    // current polynomial.
    ClonePtr<HornerSchemeNode> createHornerScheme(
            const std::vector<std::string>& vars,
            const MultivariatePolynomial& poly) {

        int order = poly.calcOrder();
        if (order == 0) {
            // For a zeroth order polynomial, construct a Monomial node with a
            // constant term and no coefficients. If no constant term exists,
            // the constant is set to zero.
            T constant = 0.0;
            if (poly.count({})) {
                constant = static_cast<T>(poly.at({}));
            }
            MonomialNode* monomial = new MonomialNode(
                    constant, Vector_<T>());
            return ClonePtr<MonomialNode>(monomial);

        } else if (order == 1) {
            // For a first order polynomial, construct a Monomial node with a
            // constant term and coefficients equal to the polynomial
            // coefficients.
            int dimension = static_cast<int>(vars.size());
            Vector_<T> coefficients =
                    poly.calcCoefficients(dimension, order, vars);

            // Reverse the coefficients of the monomial to match the
            // convention of the MultivariatePolynomialFunction.
            Vector_<T> x_coefs(dimension);
            int index = 0;
            for (int i = dimension; i > 0; --i) {
                x_coefs[index] = coefficients[i];
                ++index;
            }

            MonomialNode* monomial = new MonomialNode(
                    static_cast<T>(coefficients[0]), x_coefs);
            return ClonePtr<MonomialNode>(monomial);

        } else {
            // For a polynomial of order greater than 1, factor out the variable
            // that appears in the most terms of the polynomial. Construct two
            // FactorNodes: the "left" node represents the polynomial terms
            // that do not contain the variable, and the "right" node represents
            // the polynomial terms that did contain the variable with it
            // factored out to the first order.
            int index = 0;
            int maxCount = 0;
            for (int i = 0; i < static_cast<int>(vars.size()); ++i) {
                int count = 0;
                for (const auto& term : poly) {
                    if (term.first.find(vars[i]) != term.first.end()) {
                        ++count;
                    }
                }
                if (count > maxCount) {
                    maxCount = count;
                    index = i;
                }
            }

            // Factor the polynomial. This returns a pair of polynomials: the
            // "left" polynomial and the "right" polynomial described above.
            auto factors = poly.factorVariable(vars[index]);

            // Recursively continue constructing the binary tree.
            ClonePtr<HornerSchemeNode> left =
                    createHornerScheme(vars, factors.first);
            ClonePtr<HornerSchemeNode> right =
                    createHornerScheme(vars, factors.second);

            // Construct a FactorNode.
            FactorNode* factor = new FactorNode(left, right, index);
            return ClonePtr<FactorNode>(factor);
        }
    }

    ClonePtr<HornerSchemeNode> m_value;
    Array_<ClonePtr<HornerSchemeNode>> m_derivatives;
};

//=============================================================================
//                              CABLE ACTUATOR
//=============================================================================
class CableActuator : public Force::Custom::Implementation {
public:

CableActuator(const CableSpan& cable, Real tension) : m_cable(cable),
                                                      m_tension(tension) {}
virtual void calcForce(const State& state,
                        Vector_<SpatialVec>& bodyForces,
                        Vector_<Vec3>& particleForces,
                        Vector& mobilityForces) const override {
    m_cable.applyBodyForces(state, m_tension, bodyForces);
}

virtual Real calcPotentialEnergy(const State& state) const override {
    return 0.0;
}

private:
    const CableSpan& m_cable;
    const Real m_tension;
};

//=============================================================================
//                         FUNCTION BASED ACTUATOR
//=============================================================================
class FunctionBasedActuator : public Force::Custom::Implementation {
public:

FunctionBasedActuator(const SimTK::MultibodySystem& system,
                      const MultivariatePolynomialFunction<Real>& function,
                      Real tension) :
         m_system(system), m_matter(system.getMatterSubsystem()),
         m_function(function), m_tension(tension) {}

virtual void calcForce(const State& state,
                        Vector_<SpatialVec>& bodyForces,
                        Vector_<Vec3>& particleForces,
                        Vector& mobilityForces) const override {

    // Retrieve the current coordinate values from the state. If the system is
    // using Euler angles, adjust the size of the coordinate vector accordingly.
    Vector q = state.getQ();
    if (m_matter.getUseEulerAngles(state)) {
        q.resizeKeep(q.size()-1);
    }

    // Compute the moment arms.
    Vector momentArms(q.size(), 0.0);
    for (int i = 0; i < q.size(); ++i) {
        // Negative sign to obey the OpenSim convention.
        momentArms[i] = -m_function.calcDerivative({i}, q);
    }

    // Compute the q-space force by multiplying the moment arms by the tension.
    Vector fq = m_tension * momentArms;

    // Convert the "q-like" forces to mobility forces.
    m_system.multiplyByNTranspose(state, fq, mobilityForces);
}

virtual Real calcPotentialEnergy(const State& state) const override {
    return 0.0;
}

private:
    const SimTK::MultibodySystem& m_system;
    const SimTK::SimbodyMatterSubsystem& m_matter;
    MultivariatePolynomialFunction<Real> m_function;
    const Real m_tension;
};

//=============================================================================
//                            3D PENDULUM SYSTEM
//=============================================================================
class Pendulum {
public:
    Pendulum(Real tension) : m_matter(m_system), m_forces(m_system),
            m_cables(m_system), m_gravity(m_forces, m_matter, -YAxis, 9.8) {

        // Describe mass and visualization properties for a generic body.
        Body::Rigid bodyInfo(MassProperties(1.0, Vec3(0), UnitInertia(1)));
        bodyInfo.addDecoration(Transform(), DecorativeSphere(0.1));

        // Add a 3D ball joint pendulum to the system.
        m_pendulum = MobilizedBody::Ball(m_matter.Ground(),
            Transform(Vec3(0)), bodyInfo, Transform(Vec3(0, 0.5, 0)));

        // Construct a new cable.
        m_cable = CableSpan(m_cables,
            m_matter.Ground(), Vec3{0.1, 0., 0.2},
            m_pendulum, Vec3{0.1, 0.1, 0.1});

        // Add a CableActuator to the system.
        if (tension != 0) {
            CableActuator* cableActuator = new CableActuator(m_cable, tension);
            Force::Custom(m_forces, cableActuator);
        }

        // Realize to topology.
        m_system.realizeTopology();
    }

    int getNQ(const State& s) const {
        return (m_matter.getUseEulerAngles(s)) ? s.getNQ()-1 : s.getNQ();
    }

    Vector getQ(const State& s) const {
        Vector q = s.getQ();
        if (m_matter.getUseEulerAngles(s)) {
            q.resizeKeep(q.size()-1);
        }
        return q;
    }

    void setInitialState(State& state) {
        // Set the initial conditions.
        Rotation rotation(BodyRotationSequence, 0.05*Pi, XAxis, -0.05*Pi, ZAxis);
        m_pendulum.setQToFitRotation(state, rotation);
        m_pendulum.setOneU(state, 0, 0.1);
    }

    Real calcLength(const State& state) const {
        return m_cable.calcLength(state);
    }

    Vector calcMomentArms(const State& state) const {
        // Local modifiable copy of the state.
        State s = state;
        s.updQ() = state.getQ();
        int NQ = getNQ(state);

        // Compute the moment arms.
        Vector momentArms(NQ, 0.0);
        Vector qdot = s.getQDot();
        for (int i = 0; i < NQ; ++i) {
            // Set the derivative of this coordinate to 1 and all other
            // coordinate derivatives to 0.
            qdot = 0.;
            qdot[i] = 1.;

            // Calculate the corresponding change to u and update the state.
            Vector u = s.getU();
            u = 0.;
            m_matter.multiplyByNInv(s, false, qdot, u);
            s.updU() = u;

            // Realize the system to the velocity stage and retrieve the moment
            // arm. Negative sign to match the OpenSim convention. Note that in
            // the general case we would also need to ensure here that all
            // kinematic constraints are satisfied.
            m_system.realize(s, Stage::Velocity);
            momentArms[i] = -m_cable.calcLengthDot(s);
        }

        return momentArms;
    }

    MultibodySystem          m_system;
    SimbodyMatterSubsystem   m_matter;
    GeneralForceSubsystem    m_forces;
    CableSubsystem           m_cables;
    Force::Gravity           m_gravity;

    MobilizedBody::Ball      m_pendulum;
    CableSpan                m_cable;
};

//=============================================================================
//                         FIT FUNCTION BASED PATH
//=============================================================================
int choose(int n, int k) {
    if (k == 0) { return 1; }
    return (n * choose(n - 1, k - 1)) / k;
}

Vector fitPathCoefficients(const Pendulum& pendulum, std::vector<State> states,
        int order) {

    // Initialize the 'b' vector. This will contain path lengths and moment arms
    // for each coordinate at all time points in the trajectory.
    int NQ = pendulum.getNQ(states[0]);
    int NT = states.size();
    Vector b(NT * (NQ + 1), 0.0);

    // Compute the number of polynomial coefficients for a given order and
    // number of coordinates.
    int NC = choose(NQ + order, order);

    // Initialize the 'A' matrix. This will contain the polynomial terms for the
    // path length and moment arm functions (the values adjacent) to each
    // polynomial coefficient.
    Matrix A(NT * (NQ + 1), NC, 0.0);

    // Fill in the A matrix and the b vector.
    Vector coefficients(NC, 1.0);
    MultivariatePolynomialFunction<Real> dummy(coefficients, NQ, order);
    coefficients = 0.0;
    for (int i = 0; i < NT; ++i) {
        State state = states[i];
        pendulum.m_system.realize(state, Stage::Position);
        Vector q = pendulum.getQ(state);

        A(i, 0, 1, NC) = dummy.calcMonomialValues(q);
        b(i) = pendulum.calcLength(state);

        Vector momentArms = pendulum.calcMomentArms(state);
        for (int iq = 0; iq < NQ; ++iq) {
            Vector derivatives =
                    dummy.calcMonomialDerivatives({iq}, q);
            derivatives *= -1;
            A((iq+1)*NT + i, 0, 1, NC) = derivatives;
            b((iq+1)*NT + i) = momentArms[iq];
        }
    }

    // Solve the least-squares problem.
    FactorQTZ factor(A);
    factor.solve(b, coefficients);

    // Calculate the RMS error.
    Vector b_fit = A * coefficients;
    Vector error = b - b_fit;

    std::cout << "Path fitting RMSE: "
              << sqrt(error.normSqr() / error.size()) << std::endl;

    // Write original and fitted path lengths and moment arms to a CSV file.
    std::ofstream file("function_based_path_fit.csv");
    file << std::setprecision(10);

    // Write header.
    file << "time,length_original,length_fit";
    for (int iq = 0; iq < NQ; ++iq) {
        file << ",ma_" << iq << "_original,ma_" << iq << "_fit";
    }
    file << "\n";

    // Write one row per time point.
    for (int i = 0; i < NT; ++i) {
        file << states[i].getTime();
        file << "," << b(i) << "," << b_fit(i);
        for (int iq = 0; iq < NQ; ++iq) {
            int row = (iq + 1) * NT + i;
            file << "," << b(row) << "," << b_fit(row);
        }
        file << "\n";
    }
    file.close();

    return coefficients;
}

//=============================================================================
//                                SIMULATE
//=============================================================================
std::vector<State> simulate(Pendulum& pendulum, double finalTime,
        double timeStep, bool visualize = false,
        bool useEulerAngles = true) {

    MultibodySystem& system = pendulum.m_system;
    SimbodyMatterSubsystem& matter = pendulum.m_matter;

    // Set up visualization before realizeTopology() so that adding the event
    // reporter does not invalidate the topology stage.
    if (visualize) {
        system.setUseUniformBackground(true);
        Visualizer* viz = new Visualizer(system);
        viz->setMode(Visualizer::RealTime);
        system.addEventReporter(new Visualizer::Reporter(*viz, timeStep));
    }

    // Initialize the system.
    system.realizeTopology();
    State state = system.getDefaultState();
    matter.setUseEulerAngles(state, useEulerAngles);
    system.realizeModel(state);

    // Simulate and collect the states trajectory.
    RungeKuttaMersonIntegrator integ(system);
    TimeStepper ts(system, integ);
    pendulum.setInitialState(state);
    ts.initialize(state);
    std::vector<State> states;
    for (double t = 0; t < finalTime; t += timeStep) {
        ts.stepTo(t);
        states.push_back(ts.getState());
    }

    return states;
}

int main() {

    // Settings
    // --------
    bool visualize = false;

    // Set this to false to use a quaternion representation for the ball joint
    // orientation instead of Euler angles.
    bool useEulerAngles = false;

    // The tension in the path, whether cable-based or function-based.
    Real tension = 10.0;

    // The order of the fitted multivariate polynomial path function.
    int order = 5;

    // Create and simulate a cable-actuated pendulum system.
    Pendulum pendulum(tension);
    std::vector<State> states = simulate(pendulum, 5.0, 0.01, visualize,
        useEulerAngles);

    // Fit a multivariate polynomial function whose values equal the cable
    // lengths and whose derivatives equal the moment arms at the simulated
    // states.
    Vector coefficients = fitPathCoefficients(pendulum, states, order);
    MultivariatePolynomialFunction<Real> pathFunction(coefficients,
        pendulum.getNQ(states[0]), order);

    // Create a new pendulum system with the same properties as the original
    // system but with a FunctionBasedActuator that uses the fitted function to
    // compute moment arms and mobility forces instead of the original
    // CableActuator. Note that the CableSpan is still present in the system,
    // but it is not actuated and therefore does not contribute to the system's
    // dynamics.
    Pendulum pendulumFunctionBased(0.0);
    FunctionBasedActuator* functionBasedActuator =
        new FunctionBasedActuator(pendulumFunctionBased.m_system,
            pathFunction, tension);
    Force::Custom(pendulumFunctionBased.m_forces, functionBasedActuator);

    // Check that the function-based path model produces the same inverse
    // dynamics forces as the original model for the simulated states trajectory.
    auto calcResiduals = [](const Pendulum& pendulum,
            SimTK::State state) -> SimTK::Vector {
        pendulum.m_system.realize(state, Stage::Acceleration);
        SimTK::Vector residuals(3, 0.0);
        pendulum.m_matter.calcResidualForce(state,
                SimTK::Vector(3, 0.0),
                SimTK::Vector_<SimTK::SpatialVec>(2,
                    SimTK::SpatialVec(SimTK::Vec3(0), SimTK::Vec3(0))),
                state.getUDot(),
                SimTK::Vector(0),
                residuals);
        return residuals;
    };

    int NU = states[0].getNU();
    std::ofstream residualsFile("function_based_path_residuals.csv");
    residualsFile << std::setprecision(10);

    // Write header.
    residualsFile << "time";
    for (int i = 0; i < NU; ++i) {
        residualsFile << ",residual_original_" << i;
    }
    for (int i = 0; i < NU; ++i) {
        residualsFile << ",residual_functionbased_" << i;
    }
    residualsFile << "\n";

    // Write one row per time point.
    double averageResidualError = 0.0;
    for (const auto& state : states) {
        SimTK::Vector residuals = calcResiduals(pendulum, state);
        SimTK::Vector residualsFunctionBased =
                calcResiduals(pendulumFunctionBased, state);
        averageResidualError += (residuals - residualsFunctionBased).norm();

        residualsFile << state.getTime();
        for (int i = 0; i < NU; ++i) {
            residualsFile << "," << residuals[i];
        }
        for (int i = 0; i < NU; ++i) {
            residualsFile << "," << residualsFunctionBased[i];
        }
        residualsFile << "\n";
    }
    residualsFile.close();

    std::cout << "Average residual error between original and function-based "
              << "models: " << averageResidualError / states.size() << std::endl;

    return EXIT_SUCCESS;
}
