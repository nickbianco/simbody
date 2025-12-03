#include "Simbody.h"
using namespace SimTK;
int main() {
    // Define the system.
    MultibodySystem system;
    SimbodyMatterSubsystem matter(system);
    GeneralForceSubsystem forces(system);
    Force::Gravity gravity(forces, matter, -YAxis, 9.8);

    // Describe mass and visualization properties for a generic body.
    Body::Rigid bodyInfo(MassProperties(1.0, Vec3(0), UnitInertia(1)));
    bodyInfo.addDecoration(Transform(), DecorativeTorus(0.5, 0.1));

    // Create the moving (mobilized) bodies of the pendulum.
    MobilizedBody::Weld weld(matter.Ground(), Transform(Vec3(0)),
            bodyInfo, Transform(Vec3(0, 0, 0)));

    // Set up visualization.
    system.setUseUniformBackground(true);
    Visualizer viz(system);
    system.addEventReporter(new Visualizer::Reporter(viz, 0.01));

    // Initialize the system and state.
    State state = system.realizeTopology();

    // Simulate for 20 seconds.
    RungeKuttaMersonIntegrator integ(system);
    TimeStepper ts(system, integ);
    ts.initialize(state);
    ts.stepTo(20.0);
}