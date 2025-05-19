/* -------------------------------------------------------------------------- *
 *                     Simbody(tm) Example: Gait3D                            *
 * -------------------------------------------------------------------------- *
 * This is part of the SimTK biosimulation toolkit originating from           *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org/home/simbody.  *
 *                                                                            *
 * Portions copyright (c) 2011-12 Stanford University and the Authors.        *
 * Authors: Nicholas Bianco                                                   *
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

// -----------------------------------------------------------------------------
// Replicating SCONE's Gait3D example (H1922 model) for studying performance.
// -----------------------------------------------------------------------------
#include "Simbody.h"

#include <utility>
#include <map>

using namespace SimTK;


class PointPathMuscle : public Force::Custom::Implementation {
public:
    PointPathMuscle(const SimbodyMatterSubsystem& matter);

    void addPoint(MobilizedBodyIndex body, const Vec3& station);

    virtual void calcForce(const State& state,
                            Vector_<SpatialVec>& bodyForces,
                            Vector_<Vec3>& particleForces,
                            Vector& mobilityForces) const override;

    virtual Real calcPotentialEnergy(const State& state) const override;

    void addDecorativeLines(DecorationSubsystem& viz,
                            const DecorativeLine& line) const {
        for (int i=1; i < m_bodies.size(); ++i) { 
            viz.addRubberBandLine(m_bodies[i-1], m_stations[i-1], 
                m_bodies[i], m_stations[i], line);
        }
    }

    const MobilizedBody& getMobilizedBody(MobilizedBodyIndex body) const {
        return m_matter.getMobilizedBody(body);
    }

private:
    Array_<MobilizedBodyIndex> m_bodies;
    Array_<Vec3> m_stations;

    const SimbodyMatterSubsystem& m_matter;

    Real m_k = 5000;
    Real m_d = 100;
    Real m_x0 = 0.1;
};


class Gait3D {
public:
    enum BodyType {LeftFoot=0, RightFoot,
                   LeftShank, RightShank,
                   LeftThigh, RightThigh,
                   Pelvis, Torso};
    enum Muscle   {GlutMed_R, AddMag_R, Hamstrings_R, Bifemsh_R,
                   GlutMax_R, Iliopsoas_R, RectFem_R, Vasti_R,
                   Gastroc_R, Soleus_R, TibAnt_R, GlutMed_L,
                   AddMag_L, Hamstrings_L, Bifemsh_L,
                   GlutMax_L, Iliopsoas_L, RectFem_L, Vasti_L,
                   Gastroc_L, Soleus_L, TibAnt_L};

    static const int NBodyType = Torso-LeftFoot+1;

    Gait3D();

    void loadDefaultState(State& state);

    MultibodySystem             m_system;
    SimbodyMatterSubsystem      m_matter;
    GeneralForceSubsystem       m_forces;
    ContactTrackerSubsystem     m_tracker;
    CompliantContactSubsystem   m_contactForces;
    DecorationSubsystem         m_viz;
    Force::Gravity              m_gravity;

    Vector m_mass;             // index by BodyType
    Vector_<Vec3> m_inertia;  // index by BodyType

    std::map<BodyType, Body>            m_body;
    std::map<BodyType, MobilizedBody>   m_mobod;
    std::map<Muscle, PointPathMuscle*>  m_muscles;

private:
    static Real massData[NBodyType];
    static Vec3 inertiaData[NBodyType];
};


//////////////////////////////////////////////////////////////////////////
int main() {
  try {
    Gait3D model;

    MultibodySystem& system = model.m_system; 
    
    // Add visualization.
    // Visualizer viz(system);
    // system.addEventReporter(new Visualizer::Reporter(viz, 0.01));
     
    // Initialize the system and state.
    system.realizeTopology();
    State state = system.getDefaultState();
    model.loadDefaultState(state);
    Assembler(system).assemble(state);

    // viz.report(state);

    // Simulate it.
    // RungeKuttaMersonIntegrator integ(system);
    SemiExplicitEuler2Integrator integ(system);
    integ.setAccuracy(.01);
    TimeStepper ts(system, integ);
    ts.initialize(state);

    double cpuStart = cpuTime();
    double realStart = realTime();
    double finalTime = 20.0;
    ts.stepTo(finalTime);
    double cpu_time = cpuTime()-cpuStart;
    double real_time = realTime()-realStart;
    double realTimeFactor = finalTime/(real_time);
    std::cout << "cpu time:  "        << cpu_time << std::endl;
    std::cout << "real time: "        << real_time << std::endl;
    std::cout << "real time factor: " << realTimeFactor << std::endl;
    std::cout << "steps:     "        << integ.getNumStepsTaken() << std::endl;

  } catch (const std::exception& exc) {
      std::cout << "EXCEPTION: " << exc.what() << std::endl;
  }
}

Real Gait3D::massData[] = {1.25, 1.25, 3.7075, 3.7075, 9.3014, 9.3014, 
                           11.777, 34.2366};

Vec3 Gait3D::inertiaData[] = {Vec3(0.0014, 0.0039, 0.0041),
                             Vec3(0.0014, 0.0039, 0.0041),
                             Vec3(0.0504, 0.0051, 0.0511),
                             Vec3(0.0504, 0.0051, 0.0511),
                             Vec3(0.1339, 0.0351, 0.1412),
                             Vec3(0.1339, 0.0351, 0.1412),
                             Vec3(0.1028, 0.0871, 0.0579),
                             Vec3(1.4745, 0.7555, 1.4314)};


Gait3D::Gait3D()
:   m_matter(m_system), m_forces(m_system), m_tracker(m_system), 
    m_contactForces(m_system, m_tracker), m_viz(m_system),
    m_gravity(m_forces, m_matter, -YAxis, 9.81),
    m_mass(NBodyType, massData), m_inertia(NBodyType, inertiaData)
{
    const Real transitionVelocity = .1;
    m_contactForces.setTransitionVelocity(transitionVelocity);

    // Create bodies
    m_body[Pelvis] = Body::Rigid(MassProperties(m_mass[Pelvis], Vec3(0),
        Inertia(m_inertia[Pelvis])));
    m_body[Torso] = Body::Rigid(MassProperties(m_mass[Torso], Vec3(0),
        Inertia(m_inertia[Torso])));
    m_body[LeftThigh] = Body::Rigid(MassProperties(m_mass[LeftThigh], Vec3(0),
        Inertia(m_inertia[LeftThigh])));
    m_body[LeftShank] = Body::Rigid(MassProperties(m_mass[LeftShank], Vec3(0),
        Inertia(m_inertia[LeftShank])));
    m_body[LeftFoot] = Body::Rigid(MassProperties(m_mass[LeftFoot],  Vec3(0),
        Inertia(m_inertia[LeftFoot])));
    m_body[RightThigh] = Body::Rigid(MassProperties(m_mass[RightThigh], Vec3(0),
        Inertia(m_inertia[RightThigh])));
    m_body[RightShank] = Body::Rigid(MassProperties(m_mass[RightShank], Vec3(0),
        Inertia(m_inertia[RightShank])));
    m_body[RightFoot] = Body::Rigid(MassProperties(m_mass[RightFoot],  Vec3(0),
        Inertia(m_inertia[RightFoot])));


    // Add ContactSurfaces to the feet.
    ContactCliqueId clique1 = ContactSurface::createNewContactClique();
    ContactMaterial material(5e6,  // stiffness
                             1.0,  // dissipation
                             0.9,  // mu_static
                             0.6,  // mu_dynamic
                             0.0); // mu_viscous

    // Left foot
    // ---------
    // Heel sphere
    m_body[LeftFoot].addContactSurface(Vec3(-0.085, -0.015, 0.005),
        ContactSurface(ContactGeometry::Sphere(0.03), material)
        .joinClique(clique1));
    m_body[LeftFoot].addDecoration(Vec3(-0.085, -0.015, 0.005),
        DecorativeSphere(0.03).setColor(Green));

    // Lateral toe sphere
    m_body[LeftFoot].addContactSurface(Vec3(0.0425, -0.03, -0.041),
        ContactSurface(ContactGeometry::Sphere(0.02), material)
        .joinClique(clique1));
    m_body[LeftFoot].addDecoration(Vec3(0.0425, -0.03, -0.041),
        DecorativeSphere(0.02).setColor(Green));

    // Medial toe sphere
    m_body[LeftFoot].addContactSurface(Vec3(0.085, -0.03, 0.0275),
        ContactSurface(ContactGeometry::Sphere(0.02), material)
        .joinClique(clique1));
    m_body[LeftFoot].addDecoration(Vec3(0.085, -0.03, 0.0275),
        DecorativeSphere(0.02).setColor(Green));

    // Right foot
    // ----------
    // Heel sphere
    m_body[RightFoot].addContactSurface(Vec3(-0.085, -0.015, -0.005),
        ContactSurface(ContactGeometry::Sphere(0.03), material)
        .joinClique(clique1));
    m_body[RightFoot].addDecoration(Vec3(-0.085, -0.015, -0.005),
        DecorativeSphere(0.03).setColor(Green));

    // Lateral toe sphere
    m_body[RightFoot].addContactSurface(Vec3(0.0425, -0.03, 0.041),
        ContactSurface(ContactGeometry::Sphere(0.02), material)
        .joinClique(clique1));
    m_body[RightFoot].addDecoration(Vec3(0.0425, -0.03, 0.041),
        DecorativeSphere(0.02).setColor(Green));

    // Medial toe sphere
    m_body[RightFoot].addContactSurface(Vec3(0.085, -0.03, -0.0275),
        ContactSurface(ContactGeometry::Sphere(0.02), material)
        .joinClique(clique1));
    m_body[RightFoot].addDecoration(Vec3(0.085, -0.03, -0.0275),
        DecorativeSphere(0.02).setColor(Green));

    // Half space
    // ----------
    // Half space normal is -x; must rotate to make it +y.
    m_matter.Ground().updBody().addContactSurface(Rotation(-Pi/2,ZAxis),
       ContactSurface(ContactGeometry::HalfSpace(), material));

    // Now create the MobilizedBodies (bodies + joints).
    m_mobod[Pelvis] = MobilizedBody::Free(m_matter.Ground(), m_body[Pelvis]);

    // Add torso.
    m_mobod[Torso] = MobilizedBody::Ball(
            m_mobod[Pelvis], Vec3(-0.03, 0.0815, 0), 
            m_body[Torso], Vec3(0.03, -0.32, 0));

    // Add left leg.
    m_mobod[LeftThigh] = MobilizedBody::Ball(
            m_mobod[Pelvis], Vec3(0, -0.0661, -0.0835),
            m_body[LeftThigh], Transform(Vec3(0, 0.17, 0)));
    m_mobod[LeftShank] = MobilizedBody::Pin(
            m_mobod[LeftThigh], Vec3(0, -0.226, 0),
            m_body[LeftShank], Transform(Vec3(0, 0.1867, 0)));
    m_mobod[LeftFoot] = MobilizedBody::Pin(
            m_mobod[LeftShank], Transform(Vec3(0, -0.2433, 0)),
            m_body[LeftFoot], Transform(Vec3(-0.05123, 0.01195, 0.00792)));

    // Add right leg.
    m_mobod[RightThigh] = MobilizedBody::Ball(
            m_mobod[Pelvis], Vec3(0, -0.0661, 0.0835),
            m_body[RightThigh], Transform(Vec3(0, 0.17, 0)));
    m_mobod[RightShank] = MobilizedBody::Pin(
            m_mobod[RightThigh], Vec3(0, -0.226, 0),
            m_body[RightShank], Transform(Vec3(0, 0.1867, 0)));
    m_mobod[RightFoot] = MobilizedBody::Pin(
            m_mobod[RightShank], Transform(Vec3(0, -0.2433, 0)),
            m_body[RightFoot], Transform(Vec3(-0.05123, 0.01195, -0.00792)));

    DecorativeLine baseLine;
    baseLine.setColor(Red).setLineThickness(4).setOpacity(.2);

    // Muscles
    // -------
    PointPathMuscle* muscle;

    // glut_med_r
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[Pelvis],     Vec3(-0.0148, 0.0445, 0.0766));
    muscle->addPoint(m_mobod[RightThigh], Vec3(-0.0258, 0.1642, 0.0527));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[GlutMed_R] = muscle;
    Force::Custom(m_forces, muscle);

    // add_mag_r
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[Pelvis],     Vec3(-0.0025, -0.1174, 0.0255));
    muscle->addPoint(m_mobod[RightThigh], Vec3(-0.0045, 0.0489, 0.0339));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[AddMag_R] = muscle;
    Force::Custom(m_forces, muscle);

    // hamstrings_r
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[Pelvis],     Vec3(-0.05526, -0.10257, 0.06944));
    muscle->addPoint(m_mobod[RightShank], Vec3(-0.028, 0.1667, 0.02943));
    muscle->addPoint(m_mobod[RightShank], Vec3(-0.021, 0.1467, 0.0343));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[Hamstrings_R] = muscle;
    Force::Custom(m_forces, muscle);

    // bifemsh_r
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[RightThigh], Vec3(0.005, -0.0411, 0.0234));
    muscle->addPoint(m_mobod[RightShank], Vec3(-0.028, 0.1667, 0.02943));
    muscle->addPoint(m_mobod[RightShank], Vec3(-0.021, 0.1467, 0.0343));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[Bifemsh_R] = muscle;
    Force::Custom(m_forces, muscle);

    // glut_max_r
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[Pelvis],     Vec3(-0.0642, 0.0176, 0.0563));
    muscle->addPoint(m_mobod[Pelvis],     Vec3(-0.0669, -0.052, 0.0914));
    muscle->addPoint(m_mobod[RightThigh], Vec3(-0.0426, 0.117, 0.0293));
    muscle->addPoint(m_mobod[RightThigh], Vec3(-0.0156, 0.0684, 0.0419));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[GlutMax_R] = muscle;
    Force::Custom(m_forces, muscle);

    // iliopsoas_r
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[Pelvis],     Vec3(0.006, 0.0887, 0.0289));
    muscle->addPoint(m_mobod[Pelvis],     Vec3(0.0407, -0.01, 0.076));
    muscle->addPoint(m_mobod[RightThigh], Vec3(0.033, 0.135, 0.0038));
    muscle->addPoint(m_mobod[RightThigh], Vec3(-0.0188, 0.1103, 0.0104));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[Iliopsoas_R] = muscle;
    Force::Custom(m_forces, muscle);

    // rect_fem_r
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[Pelvis],     Vec3(0.0412, -0.0311, 0.0968));
    muscle->addPoint(m_mobod[RightThigh], Vec3(0.038, -0.17, 0.004));
    muscle->addPoint(m_mobod[RightShank], Vec3(0.038, 0.2117, 0.0018));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[RectFem_R] = muscle;
    Force::Custom(m_forces, muscle);

    // vasti_r
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[RightThigh], Vec3(0.029, -0.0224, 0.031));
    muscle->addPoint(m_mobod[RightThigh], Vec3(0.038, -0.17, 0.007));
    muscle->addPoint(m_mobod[RightShank], Vec3(0.038, 0.2117, 0.0018));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[Vasti_R] = muscle;
    Force::Custom(m_forces, muscle);

    // gastroc_r
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[RightThigh], Vec3(-0.02, -0.218, -0.024));
    muscle->addPoint(m_mobod[RightFoot],  Vec3(-0.095, 0.001, -0.0053));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[Gastroc_R] = muscle;
    Force::Custom(m_forces, muscle);

    // soleus_r
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[RightShank], Vec3(-0.0024, 0.0334, 0.0071));
    muscle->addPoint(m_mobod[RightFoot],  Vec3(-0.095, 0.001, -0.0053));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[Soleus_R] = muscle;
    Force::Custom(m_forces, muscle);

    // tib_ant_r
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[RightShank], Vec3(0.0179, 0.0243, 0.0115));
    muscle->addPoint(m_mobod[RightShank], Vec3(0.0329, -0.2084, -0.0177));
    muscle->addPoint(m_mobod[RightFoot],  Vec3(0.0166, -0.0122, -0.0305));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[TibAnt_R] = muscle;
    Force::Custom(m_forces, muscle);

    // glut_med_l
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[Pelvis],    Vec3(-0.0148, 0.0445, -0.0766));
    muscle->addPoint(m_mobod[LeftThigh], Vec3(-0.0258, 0.1642, -0.0527));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[GlutMed_L] = muscle;
    Force::Custom(m_forces, muscle);

    // add_mag_l
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[Pelvis],    Vec3(-0.0025, -0.1174, -0.0255));
    muscle->addPoint(m_mobod[LeftThigh], Vec3(-0.0045, 0.0489, -0.0339));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[AddMag_L] = muscle;
    Force::Custom(m_forces, muscle);

    // hamstrings_l
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[Pelvis],   Vec3(-0.05526, -0.10257, -0.06944));
    muscle->addPoint(m_mobod[LeftShank], Vec3(-0.028, 0.1667, -0.02943));
    muscle->addPoint(m_mobod[LeftShank], Vec3(-0.021, 0.1467, -0.0343));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[Hamstrings_L] = muscle;
    Force::Custom(m_forces, muscle);

    // bifemsh_l
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[LeftThigh], Vec3(0.005, -0.0411, -0.0234));
    muscle->addPoint(m_mobod[LeftShank], Vec3(-0.028, 0.1667, -0.02943));
    muscle->addPoint(m_mobod[LeftShank], Vec3(-0.021, 0.1467, -0.0343));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[Bifemsh_L] = muscle;
    Force::Custom(m_forces, muscle);

    // glut_max_l
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[Pelvis],    Vec3(-0.0642, 0.0176, -0.0563));
    muscle->addPoint(m_mobod[Pelvis],    Vec3(-0.0669, -0.052, -0.0914));
    muscle->addPoint(m_mobod[LeftThigh], Vec3(-0.0426, 0.117, -0.0293));
    muscle->addPoint(m_mobod[LeftThigh], Vec3(-0.0156, 0.0684, -0.0419));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[GlutMax_L] = muscle;
    Force::Custom(m_forces, muscle);

    // iliopsoas_l
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[Pelvis],    Vec3(0.006, 0.0887, -0.0289));
    muscle->addPoint(m_mobod[Pelvis],    Vec3(0.0407, -0.01, -0.076));
    muscle->addPoint(m_mobod[LeftThigh], Vec3(0.033, 0.135, -0.0038));
    muscle->addPoint(m_mobod[LeftThigh], Vec3(-0.0188, 0.1103, -0.0104));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[Iliopsoas_L] = muscle;
    Force::Custom(m_forces, muscle);

    // rect_fem_l
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[Pelvis],    Vec3(0.0412, -0.0311, -0.0968));
    muscle->addPoint(m_mobod[LeftThigh], Vec3(0.038, -0.17, -0.004));
    muscle->addPoint(m_mobod[LeftShank], Vec3(0.038, 0.2117, -0.0018));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[RectFem_L] = muscle;
    Force::Custom(m_forces, muscle);

    // vasti_l
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[LeftThigh], Vec3(0.029, -0.0224, -0.031));
    muscle->addPoint(m_mobod[LeftThigh], Vec3(0.038, -0.17, -0.007));
    muscle->addPoint(m_mobod[LeftShank], Vec3(0.038, 0.2117, -0.0018));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[Vasti_L] = muscle;
    Force::Custom(m_forces, muscle);

    // gastroc_l
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[LeftThigh], Vec3(-0.02, -0.218, 0.024));
    muscle->addPoint(m_mobod[LeftFoot],  Vec3(-0.095, 0.001, 0.0053));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[Gastroc_L] = muscle;
    Force::Custom(m_forces, muscle);

    // soleus_l
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[LeftShank], Vec3(-0.0024, 0.0334, -0.0071));
    muscle->addPoint(m_mobod[LeftFoot],  Vec3(-0.095, 0.001, 0.0053));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[Soleus_L] = muscle;
    Force::Custom(m_forces, muscle);

    // tib_ant_l
    muscle = new PointPathMuscle(m_matter);
    muscle->addPoint(m_mobod[LeftShank], Vec3(0.0179, 0.0243, -0.0115));
    muscle->addPoint(m_mobod[LeftShank], Vec3(0.0329, -0.2084, 0.0177));
    muscle->addPoint(m_mobod[LeftFoot],  Vec3(0.0166, -0.0122, 0.0305));
    muscle->addDecorativeLines(m_viz, baseLine);
    m_muscles[TibAnt_L] = muscle;
    Force::Custom(m_forces, muscle);
}

void Gait3D::loadDefaultState(State& state) {
    const static Real hipAngle = -15*Pi/180;
    const static Real kneeAngle = -40*Pi/180;
    const static Real ankleAngle = 20*Pi/180;
    const static Real hipVelocity = .125;
    const static Real kneeVelocity = 0;

    m_mobod[Pelvis].setQToFitTranslation(state, Vec3(0,1.0,0));
    m_mobod[Pelvis].setOneU(state, 2, -hipVelocity);

    m_mobod[LeftThigh].setOneQ(state, 0, -hipAngle);
    m_mobod[LeftShank].setOneQ(state, 0, kneeAngle);
    m_mobod[LeftFoot].setOneQ(state, 0, ankleAngle);
    m_mobod[LeftThigh].setOneU(state, 0, hipVelocity);
    m_mobod[LeftShank].setOneU(state, 0, kneeVelocity);

    m_mobod[RightThigh].setOneQ(state, 0, hipAngle);
    m_mobod[RightShank].setOneQ(state, 0, kneeAngle);
    m_mobod[RightFoot].setOneQ(state, 0, ankleAngle);
    m_mobod[RightThigh].setOneU(state, 0, hipVelocity);
    m_mobod[RightShank].setOneU(state, 0, -kneeVelocity);
}


//////////////////////////////////////////////////////////////////////////

PointPathMuscle::PointPathMuscle(const SimbodyMatterSubsystem& matter) : 
        m_matter(matter) {
    m_bodies.resize(0);
    m_stations.resize(0);
}

void PointPathMuscle::addPoint(MobilizedBodyIndex body, const Vec3& station) {
    m_bodies.push_back(body);
    m_stations.push_back(station);
}

void PointPathMuscle::calcForce(const State& state,
                                Vector_<SpatialVec>& bodyForces,
                                Vector_<Vec3>& particleForces,
                                Vector& mobilityForces) const {
    Real length = 0;
    Real lengthDot = 0;
    Array_<UnitVec3> dirs(m_bodies.size()-1);
    Array_<Vec3> s_G(m_bodies.size());

    const MobilizedBody& body1 = getMobilizedBody(m_bodies[0]);
    const Transform& X_GB1 = body1.getBodyTransform(state);
    Vec3 s1_G = X_GB1.R() * m_stations[0];
    s_G[0] = s1_G;
    Vec3 p1_G = X_GB1.p() + s1_G;
    Vec3 v1_G = body1.findStationVelocityInGround(state, m_stations[0]);
    for (int i = 1; i < m_bodies.size(); ++i) {
        const MobilizedBody& body2 = getMobilizedBody(m_bodies[i]);
        const Transform& X_GB2 = body2.getBodyTransform(state);
        const Vec3 s2_G = X_GB2.R() * m_stations[i];
        s_G[i] = s2_G;
        const Vec3 p2_G = X_GB2.p() + s2_G;
        const Vec3 v2_G = body2.findStationVelocityInGround(state, m_stations[i]);
        const Vec3 r_G = p2_G - p1_G; // vector from point1 to point2
        const UnitVec3 dir(r_G);
        dirs[i-1] = dir; 

        const Real dist = r_G.norm();  // distance between the points
        if( dist < SignificantReal ) return;
        length += dist;
        lengthDot += dot(v2_G - v1_G, dir); // relative velocity    

        s1_G = s2_G;
        p1_G = p2_G;
        v1_G = v2_G;
    }

    const Real stretch   = length - m_x0;  // + -> tension, - -> compression
    const Real frcStretch = m_k*stretch;  // k(x-x0)
    const Real frcDamp = m_d*lengthDot; // c*v

    for (int i = 1; i < m_bodies.size(); ++i) {
        const Vec3 f_G = (frcStretch + frcDamp) * dirs[i-1];

        bodyForces[m_bodies[i-1]] +=  SpatialVec(s_G[i-1] % f_G, f_G);
        bodyForces[m_bodies[i]]   -=  SpatialVec(s_G[i] % f_G, f_G);
    }
}

Real PointPathMuscle::calcPotentialEnergy(const State& state) const {

    Real length = 0;
    const MobilizedBody& body1 = getMobilizedBody(m_bodies[0]);
    const Transform& X_GB1 = body1.getBodyTransform(state);
    Vec3 s1_G = X_GB1.R() * m_stations[0];
    Vec3 p1_G = X_GB1.p() + s1_G;
    for (int i = 1; i < m_bodies.size(); ++i) {
        const MobilizedBody& body2 = getMobilizedBody(m_bodies[i]);
        const Transform& X_GB2 = body2.getBodyTransform(state);
        const Vec3 s2_G = X_GB2.R() * m_stations[i];
        const Vec3 p2_G = X_GB2.p() + s2_G;
        const Vec3 r_G = p2_G - p1_G; // vector from point1 to point2

        const Real dist = r_G.norm(); 
        length += dist; 

        s1_G = s2_G;
        p1_G = p2_G;
    }

    const Real stretch = length - m_x0;  // + -> tension, - -> compression
    return 0.5*m_k*stretch*stretch;      // 1/2 k (x-x0)^2
}



