/*-----------------------------------------------------------------------------
            Simbody(tm) Test: Cable Over Smooth Surfaces and Via Points
 -------------------------------------------------------------------------------
 Copyright (c) 2024 Authors.
 Authors: Pepijn van den Bos
 Contributors: Nicholas Bianco

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
This file contains tests for the CableSpan's path over different obstacles and
via points.
**/

/* A helper class for drawing interesting things of a CableSpan. */
class CableDecorator : public SimTK::DecorationGenerator {
public:
    CableDecorator(MultibodySystem& mbs, const CableSpan& cable) :
        m_mbs(&mbs), m_cable(cable)
    {
        for (CableSpanObstacleIndex ix(0); ix < m_cable.getNumObstacles();
            ++ix) {
            m_obstacleDecorations.push_back(
                m_cable.getObstacleContactGeometry(ix)
                    .createDecorativeGeometry()
                    .setResolution(3));
            m_obstacleDecorationsOffsets.push_back(
                m_obstacleDecorations.back().getTransform());
        }
    }

    void generateDecorations(
        const State& state,
        Array_<DecorativeGeometry>& decorations) override
    {
        for (CableSpanObstacleIndex ix(0); ix < m_cable.getNumObstacles();
            ++ix) {
            // Draw the obstacle surface.
            const ContactGeometry& geometry =
                m_cable.getObstacleContactGeometry(ix);
            // If cable is not in contact with the surface grey it out.
            const bool isInContactWithSurface =
                m_cable.isInContactWithObstacle(state, ix);
            const Vec3 color   = isInContactWithSurface ? Yellow : Gray;
            const Real opacity = isInContactWithSurface ? 0.5 : 0.25;
            // Transform from Ground to obstacle body.
            Transform X_GB =
                m_mbs->getMatterSubsystem()
                    .getMobilizedBody(m_cable.getObstacleMobilizedBodyIndex(ix))
                    .getBodyTransform(state);
            // Transform from Ground to obstacle contact surface offset frame.
            const Transform X_GS =
                X_GB.compose(m_cable.getObstacleTransformSurfaceToBody(ix));
            // Transform from ground to decoration surface.
            const Transform X_GD =
                X_GS.compose(m_obstacleDecorationsOffsets.at(ix));
            // Draw the obstacle's local frame.
            // This is the frame that you define the contact point hint in.
            decorations.push_back(
                DecorativeFrame(0.5).setTransform(X_GS).setColor(Purple));
            // Draw the obstacle contact geometry.
            decorations.push_back(m_obstacleDecorations.at(ix)
                                    .setTransform(X_GD)
                                    .setColor(color)
                                    .setOpacity(opacity));

            // Draw the initial contact point hints (these are user-defined) as
            // a line and a point.
            const Vec3 x_PS = m_cable.getObstacleContactPointHint(ix);
            decorations.push_back(
                DecorativeLine(X_GS.p(), X_GS.shiftFrameStationToBase(x_PS))
                    .setColor(Green)
                    .setLineThickness(3));
            decorations.push_back(
                DecorativePoint(X_GS.shiftFrameStationToBase(x_PS))
                    .setColor(Green));
        }

        for (CableSpanViaPointIndex ix(0); ix < m_cable.getNumViaPoints();
                ++ix) {
            // Draw the via point.
            const Vec3 x_G = m_cable.calcViaPointLocation(state, ix);
            decorations.push_back(
                DecorativeSphere(0.1)
                    .setTransform(x_G)
                    .setRepresentation(DecorativeGeometry::DrawWireframe)
                    .setColor(Cyan)
                    .setOpacity(0.1));
        }
    }

    MultibodySystem* m_mbs;
    CableSpan m_cable;
    Array_<DecorativeGeometry, CableSpanObstacleIndex> m_obstacleDecorations;
    Array_<Transform, CableSpanObstacleIndex> m_obstacleDecorationsOffsets;
};

/** Simple CableSpan path with known solution.

Wrap a cable over (in order):
1. Torus
2. Ellipsoid
3. Torus
4. Cylinder

We wrap the cable conveniently over the obstacles such that each curve
segment becomes a circular-arc shape. This allows us to check the results by
hand. **/
void testSimpleCable()
{
    const bool show = true;

    // Create the system.
    MultibodySystem system;
    SimbodyMatterSubsystem matter(system);
    CableSubsystem cables(system);

    // A dummy body.
    Body::Rigid aBody(MassProperties(1., Vec3(0), Inertia(1)));

    // Mobilizer for path origin.
    MobilizedBody::Translation cableOriginBody(
        matter.Ground(),
        Vec3(0.),
        aBody,
        Transform());

    // Mobilizer for path termination.
    MobilizedBody::Translation cableTerminationBody(
        matter.Ground(),
        Transform(Vec3(-0.1, 0.2, -0.05)),
        aBody,
        Transform());

    // Construct a new cable.
    CableSpan cable(
        cables,
        cableOriginBody,
        Vec3{0.},
        cableTerminationBody,
        Vec3{0.});
    cable.setCurveSegmentAccuracy(1e-10);
    cable.setSmoothnessTolerance(1e-5);

    // Add ellipsoid obstacle.
    MobilizedBody::Translation ellipsoidBody(
        matter.Ground(),
        Transform(),
        aBody,
        Transform());

    // Rotation rotation(0.25*Pi, YAxis);
    cable.addObstacle(
        ellipsoidBody,
        Transform(Vec3{-0.35, 0.15, -0.05}),
        std::shared_ptr<ContactGeometry>(
            new ContactGeometry::Ellipsoid({0.1, 0.1, 0.2})),
        {0., 0.125, 0.});


    // Optionally visualize the system.
    system.setUseUniformBackground(true); // no ground plane in display
    std::unique_ptr<Visualizer> viz(show ? new Visualizer(system) : nullptr);

    if (viz) {
        viz->setShowFrameNumber(true);
        viz->addDecorationGenerator(new CableDecorator(system, cable));
    }

    // Initialize the system and state.
    system.realizeTopology();
    State s = system.getDefaultState();

    // Compute the CableSpan's path.
    system.realize(s, Stage::Report);
    cable.calcLength(s);
    if (viz) {
        viz->report(s);
    }
}

int main()
{
    testSimpleCable();
}
