/*-----------------------------------------------------------------------------
                Simbody(tm) Test: Cable with Moving Stations
-------------------------------------------------------------------------------
 Copyright (c) 2026 Authors.
 Authors: Nicholas Bianco
 Contributors:

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

using namespace SimTK;

/**
This file contains tests for cable stations whose location in the body frame is
a function of generalized coordinates, i.e., "moving" stations. A moving station
must:
- reduce to a fixed station when the function is constant,
- produce a lengthening speed consistent with the cable length,
- apply forces (body forces plus mobility forces) that do the same virtual
  work as the cable tension, i.e., f_u = -tension * dL/du,
- produce a cable power equal to -tension * lengthDot.
**/

// A smooth test function of n coordinates with analytic first derivatives:
// f(x) = c + sum_k A_k sin(x_k + phi_k) + d * x_0 * x_{n-1}.
class TestStationFunction : public Function_<Vec3> {
public:
    TestStationFunction(
        const Vec3& c,
        const Array_<Vec3>& A,
        const Array_<Real>& phi,
        const Vec3& d) :
        m_c(c), m_A(A), m_phi(phi), m_d(d)
    {}

    Vec3 calcValue(const Vector& x) const override
    {
        Vec3 value = m_c + m_d * x[0] * x[getArgumentSize() - 1];
        for (int k = 0; k < getArgumentSize(); ++k) {
            value += m_A[k] * std::sin(x[k] + m_phi[k]);
        }
        return value;
    }

    Vec3 calcDerivative(
        const Array_<int>& derivComponents,
        const Vector& x) const override
    {
        SimTK_ASSERT_ALWAYS(
            derivComponents.size() == 1,
            "TestStationFunction only supports first derivatives.");
        const int k = derivComponents[0];
        const int n = getArgumentSize();
        Real dProduct = 0.;
        if (k == 0) {
            dProduct += x[n - 1];
        }
        if (k == n - 1) {
            dProduct += x[0];
        }
        return m_A[k] * std::cos(x[k] + m_phi[k]) + m_d * dProduct;
    }

    int getArgumentSize() const override
    {
        return (int)m_A.size();
    }

    int getMaxDerivativeOrder() const override
    {
        return 1;
    }

private:
    Vec3 m_c;
    Array_<Vec3> m_A;
    Array_<Real> m_phi;
    Vec3 m_d;
};

// A constant function of n coordinates.
class ConstantStationFunction : public Function_<Vec3> {
public:
    ConstantStationFunction(const Vec3& value, int n) :
        m_value(value), m_n(n)
    {}

    Vec3 calcValue(const Vector&) const override
    {
        return m_value;
    }

    Vec3 calcDerivative(const Array_<int>&, const Vector&) const override
    {
        return Vec3(0.);
    }

    int getArgumentSize() const override
    {
        return m_n;
    }

    int getMaxDerivativeOrder() const override
    {
        return 1;
    }

private:
    Vec3 m_value;
    int m_n;
};

// A two-body test system: bodyA is pinned to ground, and bodyB is attached to
// bodyA with a ball joint using Euler angles (so that qdot != u).
struct TestSystem {
    TestSystem() :
        matter(system),
        cables(system),
        body(MassProperties(1., Vec3(0), Inertia(1))),
        bodyA(matter.Ground(), Transform(Vec3(0., 0., 0.)), body, Transform()),
        bodyB(bodyA, Transform(Vec3(1., 0., 0.)), body, Transform())
    {}

    State createState(const Vector& q, const Vector& u) const
    {
        system.realizeTopology();
        State s = system.getDefaultState();
        matter.setUseEulerAngles(s, true);
        system.realizeModel(s);
        s.updQ() = q;
        s.updU() = u;
        system.realize(s, Stage::Velocity);
        return s;
    }

    MultibodySystem system;
    SimbodyMatterSubsystem matter;
    CableSubsystem cables;
    Body::Rigid body;
    MobilizedBody::Pin bodyA;
    MobilizedBody::Ball bodyB;
};

// Coordinates of the moving stations: (bodyA q0), and (bodyA q0, bodyB q0-q2).
Array_<MobilizedBodyIndex> coordBodies(
    const TestSystem& ts,
    const Array_<const MobilizedBody*>& bodies)
{
    Array_<MobilizedBodyIndex> indices;
    for (const MobilizedBody* body : bodies) {
        indices.push_back(body->getMobilizedBodyIndex());
    }
    return indices;
}

std::shared_ptr<const Function_<Vec3>> makeTestFunction(int n, Real seed)
{
    Array_<Vec3> A;
    Array_<Real> phi;
    for (int k = 0; k < n; ++k) {
        A.push_back(0.1 * Vec3(std::sin(seed + k), std::cos(2 * seed + k),
            std::sin(3 * seed - k)));
        phi.push_back(0.3 * (seed + k));
    }
    return std::make_shared<TestStationFunction>(
        Vec3(0.1 * seed, 0.5, -0.1 * seed),
        A,
        phi,
        Vec3(0.05, -0.02, 0.03));
}

// Add moving stations at the cable origin, a via point, and the termination.
CableSpan createMovingCable(TestSystem& ts)
{
    CableSpan cable(
        ts.cables,
        ts.matter.Ground(),
        Vec3(0., 1., 0.),
        ts.bodyB,
        Vec3(0.5, 0., 0.));
    // Origin on ground, driven by bodyA.
    cable.setOriginStation(
        makeTestFunction(1, 1.),
        coordBodies(ts, {&ts.bodyA}),
        {MobilizerQIndex(0)});
    // Via point on bodyA, driven by bodyA and bodyB.
    cable.addViaPoint(
        ts.bodyA,
        makeTestFunction(4, 2.),
        coordBodies(ts, {&ts.bodyA, &ts.bodyB, &ts.bodyB, &ts.bodyB}),
        {MobilizerQIndex(0), MobilizerQIndex(0), MobilizerQIndex(1),
            MobilizerQIndex(2)});
    // Termination on bodyB, driven by bodyB.
    cable.setTerminationStation(
        makeTestFunction(3, 3.),
        coordBodies(ts, {&ts.bodyB, &ts.bodyB, &ts.bodyB}),
        {MobilizerQIndex(0), MobilizerQIndex(1), MobilizerQIndex(2)});
    return cable;
}

const Vector c_q = Vector(Vec4(0.3, -0.2, 0.4, 0.1));
const Vector c_u = Vector(Vec4(0.7, -1.1, 0.5, 0.9));

// A moving station with a constant function must match the fixed station.
void testConstantMovingStationsMatchFixed()
{
    TestSystem tsFixed;
    CableSpan fixed(
        tsFixed.cables,
        tsFixed.matter.Ground(),
        Vec3(0., 1., 0.),
        tsFixed.bodyB,
        Vec3(0.5, 0.2, 0.));
    fixed.addViaPoint(tsFixed.bodyA, Vec3(0.3, 0.6, 0.1));

    TestSystem tsMoving;
    CableSpan moving(
        tsMoving.cables,
        tsMoving.matter.Ground(),
        Vec3(NaN),
        tsMoving.bodyB,
        Vec3(NaN));
    moving.setOriginStation(
        std::make_shared<ConstantStationFunction>(Vec3(0., 1., 0.), 1),
        coordBodies(tsMoving, {&tsMoving.bodyA}),
        {MobilizerQIndex(0)});
    moving.addViaPoint(
        tsMoving.bodyA,
        std::make_shared<ConstantStationFunction>(Vec3(0.3, 0.6, 0.1), 1),
        coordBodies(tsMoving, {&tsMoving.bodyB}),
        {MobilizerQIndex(1)});
    moving.setTerminationStation(
        std::make_shared<ConstantStationFunction>(Vec3(0.5, 0.2, 0.), 2),
        coordBodies(tsMoving, {&tsMoving.bodyA, &tsMoving.bodyB}),
        {MobilizerQIndex(0), MobilizerQIndex(2)});

    const State sFixed = tsFixed.createState(c_q, c_u);
    const State sMoving = tsMoving.createState(c_q, c_u);

    const Real lengthError =
        std::abs(fixed.calcLength(sFixed) - moving.calcLength(sMoving));
    const Real lengthDotError =
        std::abs(fixed.calcLengthDot(sFixed) - moving.calcLengthDot(sMoving));
    SimTK_ASSERT1_ALWAYS(lengthError < 1e-14,
        "testConstantMovingStationsMatchFixed failed: length error = %e",
        lengthError);
    SimTK_ASSERT1_ALWAYS(lengthDotError < 1e-14,
        "testConstantMovingStationsMatchFixed failed: lengthDot error = %e",
        lengthDotError);

    const int nb = tsFixed.matter.getNumBodies();
    Vector_<SpatialVec> forcesFixed(nb, SpatialVec(Vec3(0.), Vec3(0.)));
    Vector_<SpatialVec> forcesMoving(nb, SpatialVec(Vec3(0.), Vec3(0.)));
    fixed.applyBodyForces(sFixed, 2., forcesFixed);
    moving.applyBodyForces(sMoving, 2., forcesMoving);
    Vector mobilityForces(sMoving.getNU(), 0.);
    moving.applyMobilityForces(sMoving, 2., mobilityForces);
    for (int i = 0; i < nb; ++i) {
        SimTK_ASSERT1_ALWAYS(
            (forcesFixed[i] - forcesMoving[i]).norm() < 1e-14,
            "testConstantMovingStationsMatchFixed failed: body force error = %e",
            (forcesFixed[i] - forcesMoving[i]).norm());
    }
    SimTK_ASSERT1_ALWAYS(mobilityForces.normInf() < 1e-14,
        "testConstantMovingStationsMatchFixed failed: mobility force = %e",
        mobilityForces.normInf());
}

// Central finite difference of the cable length along qdot = N u.
Real calcLengthDotByFiniteDifference(
    const TestSystem& ts,
    const CableSpan& cable,
    const Vector& q,
    const Vector& u)
{
    const State s = ts.createState(q, u);
    Vector qdot;
    ts.matter.multiplyByN(s, false, u, qdot);
    const Real h = 1e-6;
    const State sPlus = ts.createState(q + h * qdot, u);
    const State sMinus = ts.createState(q - h * qdot, u);
    return (cable.calcLength(sPlus) - cable.calcLength(sMinus)) / (2. * h);
}

// The lengthening speed of a cable with moving stations must match the time
// derivative of its length.
void testMovingStationsLengthDot()
{
    TestSystem ts;
    CableSpan cable = createMovingCable(ts);

    const State s = ts.createState(c_q, c_u);
    const Real expected = calcLengthDotByFiniteDifference(ts, cable, c_q, c_u);
    const Real got = cable.calcLengthDot(s);
    SimTK_ASSERT3_ALWAYS(std::abs(got - expected) < 1e-8,
        "testMovingStationsLengthDot failed: expected %f, got %f (error %e)",
        expected, got, got - expected);

    // The fixed-station result differs, i.e., the stations do move.
    SimTK_ASSERT_ALWAYS(std::abs(got) > 1e-3,
        "testMovingStationsLengthDot failed: lengthDot is unexpectedly zero");
}

// The generalized forces of the cable (from body forces and mobility forces)
// must do the same virtual work as the tension: f_u . u = -tension * lengthDot,
// and the cable power must agree.
void testMovingStationsVirtualWork()
{
    TestSystem ts;
    CableSpan cable = createMovingCable(ts);
    const Real tension = 3.;

    const State s = ts.createState(c_q, c_u);
    const int nb = ts.matter.getNumBodies();
    Vector_<SpatialVec> bodyForces(nb, SpatialVec(Vec3(0.), Vec3(0.)));
    cable.applyBodyForces(s, tension, bodyForces);
    Vector mobilityForces(s.getNU(), 0.);
    cable.applyMobilityForces(s, tension, mobilityForces);

    Vector generalizedForces;
    ts.matter.multiplyBySystemJacobianTranspose(s, bodyForces,
        generalizedForces);
    generalizedForces += mobilityForces;

    const Real lengthDot = calcLengthDotByFiniteDifference(ts, cable, c_q, c_u);
    const Real expectedPower = -tension * lengthDot;
    const Real gotPower = ~generalizedForces * c_u;
    SimTK_ASSERT3_ALWAYS(std::abs(gotPower - expectedPower) < 1e-7,
        "testMovingStationsVirtualWork failed: expected power %f, got %f "
        "(error %e)",
        expectedPower, gotPower, gotPower - expectedPower);

    const Real cablePower = cable.calcCablePower(s, tension);
    SimTK_ASSERT3_ALWAYS(std::abs(cablePower - expectedPower) < 1e-7,
        "testMovingStationsVirtualWork failed: expected cable power %f, got %f "
        "(error %e)",
        expectedPower, cablePower, cablePower - expectedPower);

    // Without the mobility forces, the virtual work would not match.
    Vector bodyOnly;
    ts.matter.multiplyBySystemJacobianTranspose(s, bodyForces, bodyOnly);
    SimTK_ASSERT_ALWAYS(std::abs(~bodyOnly * c_u - expectedPower) > 1e-3,
        "testMovingStationsVirtualWork failed: mobility forces are negligible");
}

// Same as above, with a wrapping obstacle between the via point and the
// termination.
void testMovingStationsWithObstacle()
{
    TestSystem ts;
    CableSpan cable(
        ts.cables,
        ts.matter.Ground(),
        Vec3(0., 1., 0.),
        ts.bodyB,
        Vec3(0.5, 0., 0.));
    cable.setOriginStation(
        makeTestFunction(1, 1.),
        coordBodies(ts, {&ts.bodyA}),
        {MobilizerQIndex(0)});
    cable.setTerminationStation(
        std::make_shared<TestStationFunction>(
            Vec3(1.2, -0.2, 0.),
            Array_<Vec3>{Vec3(0.05, 0., 0.02), Vec3(0., 0.05, 0.),
                Vec3(0.02, 0., 0.05)},
            Array_<Real>{0.1, 0.2, 0.3},
            Vec3(0.01, 0.01, 0.)),
        coordBodies(ts, {&ts.bodyB, &ts.bodyB, &ts.bodyB}),
        {MobilizerQIndex(0), MobilizerQIndex(1), MobilizerQIndex(2)});
    // A sphere on bodyA, just below the straight line between the origin and
    // termination, that the cable wraps over.
    cable.addObstacle(
        ts.bodyA,
        Transform(Vec3(0.6, 0.32, -0.08)),
        std::make_shared<ContactGeometry::Sphere>(0.2),
        Vec3(0., 0.2, 0.));
    cable.setCurveSegmentAccuracy(1e-12);
    cable.setSmoothnessTolerance(1e-10);

    const Vector q(Vec4(0.1, 0.05, -0.05, 0.1));
    const Vector u(Vec4(0.3, -0.4, 0.2, 0.5));
    const State s = ts.createState(q, u);
    SimTK_ASSERT_ALWAYS(cable.isInContactWithObstacle(s,
        CableSpanObstacleIndex(0)),
        "testMovingStationsWithObstacle failed: cable not in contact");

    const Real lengthDot = calcLengthDotByFiniteDifference(ts, cable, q, u);
    SimTK_ASSERT2_ALWAYS(std::abs(cable.calcLengthDot(s) - lengthDot) < 1e-6,
        "testMovingStationsWithObstacle failed: expected lengthDot %f, got %f",
        lengthDot, cable.calcLengthDot(s));

    const Real tension = 2.;
    Vector_<SpatialVec> bodyForces(ts.matter.getNumBodies(),
        SpatialVec(Vec3(0.), Vec3(0.)));
    cable.applyBodyForces(s, tension, bodyForces);
    Vector mobilityForces(s.getNU(), 0.);
    cable.applyMobilityForces(s, tension, mobilityForces);
    Vector generalizedForces;
    ts.matter.multiplyBySystemJacobianTranspose(s, bodyForces,
        generalizedForces);
    generalizedForces += mobilityForces;
    const Real expectedPower = -tension * lengthDot;
    SimTK_ASSERT2_ALWAYS(
        std::abs(~generalizedForces * u - expectedPower) < 1e-6,
        "testMovingStationsWithObstacle failed: expected power %f, got %f",
        expectedPower, ~generalizedForces * u);
}

int main()
{
    testConstantMovingStationsMatchFixed();
    testMovingStationsLengthDot();
    testMovingStationsVirtualWork();
    testMovingStationsWithObstacle();
}
