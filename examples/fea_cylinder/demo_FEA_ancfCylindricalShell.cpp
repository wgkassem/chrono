// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Antonio Recuero
// =============================================================================
//
// Demo on using 8-node ANCF shell elements. These demo reproduces the example
// 3.3 of the paper: 'Analysis of higher-order quadrilateral plate elements based
// on the absolute nodal coordinate formulation for three-dimensional elasticity'
// H.C.J. Ebel, M.K.Matikainen, V.V.T. Hurskainen, A.M.Mikkola, Advances in
// Mechanical Engineering, 2017
//
// =============================================================================

#include "chrono/physics/ChSystem.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/fea/ChElementShellANCF.h"
#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChMeshExporter.h"
#include "chrono/fea/ChContactSurfaceMesh.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/solver/ChIterativeSolverLS.h"

using namespace chrono;
using namespace chrono::fea;

float cell_height = 20;
float cell_rad = 5;
float cell_diam = 2. * cell_rad;

std::string demo_dir = ".";
std::string MESH_CONNECTIVITY = "Flex_Mesh.vtk";

double dT = 5e-3; // time step
double out_fps = 100; // output fps
double sphere_swept_thickness = 0.008;


// *********************************************************************************************
void SaveParaViewFiles(ChSystemSMC& mphysicalSystem,
                                        std::shared_ptr<fea::ChMesh> my_mesh,
                                        int next_frame,
                                        double mTime,
                                        const ChVector<>&);
                        
// ********************************************************************************************


int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    double time_step = 1e-3;

    ChSystemSMC my_system;
    my_system.Set_G_acc(ChVector<>(0, 0, 0.0));

    // Create a mesh, that is a container for groups of elements and their referenced nodes.
    auto my_mesh = chrono_types::make_shared<ChMesh>();

    // Create the Nodes
    unsigned int nrots = 60;
    unsigned int nstacks = 5;
    float dtheta = 2 * M_PI / nrots; //degrees
    float dh = cell_height / (float) nstacks;
    float tile_base = M_PI * (cell_diam * 1.00) / (float) nrots;// / (float) nrots;
    float tile_height = cell_height / (float) nstacks;// / (float) nstacks; // should be a multiple of cell_hgt
    unsigned int ntiles = nrots * nstacks;
    
    unsigned int nnodes = 2 * nrots + (nstacks  - 1) + (nstacks-1) * (nrots - 1); // number of nodes in an open cylindrical with 4 nodes per tile

    // Create nodes stack by stack
    for (unsigned int istack = 0; istack < nstacks+1; istack++){

        float loc_z = -0.5 * cell_height + (float) (istack) * dh; 

        for (unsigned int irot = 0; irot < nrots; irot++){ // dont create nodes for the last element since they will be the first
            // Calculate angle of rotation
            float rot_angle_cw = -(float) irot * dtheta; // rotate clockwise
            float loc_x = cell_rad * cos(rot_angle_cw);
            float loc_y = cell_rad * sin(rot_angle_cw);
            float norm = sqrt(loc_x*loc_x + loc_y*loc_y);
            auto node = chrono_types::make_shared<ChNodeFEAxyzD>(ChVector<>(loc_x, loc_y, loc_z), ChVector<>(-loc_x/norm, -loc_y/norm, 0));

            // fix nodes along the z = -H/2 and z = H/2
            if (istack == 0 or istack == nstacks-1){node->SetFixed(true);}
            node->SetMass(0);

            my_mesh->AddNode(node);
        }
    }

    std::cout << "\nCreated " << my_mesh->GetNnodes() << " nodes\n";

    // Create an orthotropic material
    double rho = 1; // density of rubber
    float E = 200e6;
    float nu = 0.;
    auto core_mat = chrono_types::make_shared<ChMaterialShellANCF>(rho, E, nu);
    
    auto rubber_surf_mat = chrono_types::make_shared<ChMaterialSurfaceSMC>();
    rubber_surf_mat->SetYoungModulus(E);
    rubber_surf_mat->SetFriction(0.5f);
    rubber_surf_mat->SetRestitution(0.05f);
    rubber_surf_mat->SetAdhesion(0.7);

    auto steel_surf_mat = chrono_types::make_shared<ChMaterialSurfaceSMC>();
    rubber_surf_mat->SetYoungModulus(2000e9f);
    rubber_surf_mat->SetFriction(0.1f);
    rubber_surf_mat->SetRestitution(0.95f);
    rubber_surf_mat->SetAdhesion(0.01);

    unsigned int nelemnts = nstacks * nrots;
    for (unsigned int ielem=0; ielem<nelemnts; ielem++){
        int istack = ielem / nrots; // integer division
        int node1 = ielem;
        int node2 = (ielem + 1) % nrots + istack * nrots;
        int node3 = (ielem + nrots + 1) % nrots + (istack+1) * nrots;
        int node4 = (istack +1 ) * nrots + ielem % nrots;

    // Element node debugging info
    //    std::cout << node1 << " "<< node2 << " "<< node3 << " "<< node4 << "\n";

        auto element = chrono_types::make_shared<ChElementShellANCF>();
        element->SetNodes(std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node1)),
                          std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node2)),
                          std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node3)),
                          std::dynamic_pointer_cast<ChNodeFEAxyzD>(my_mesh->GetNode(node4)));
        
        element->SetDimensions(tile_base, tile_height);
        element->AddLayer(0.1, 0.5 * M_PI, core_mat); // add material layer, thicknes = 0.01, fiber angle = pi/2
        element->SetAlphaDamp(0.08);   // Structural damping for this element
        element->SetGravityOn(false);  // gravitational forces

        my_mesh->AddElement(element);
    }

    // Add a contact surface to interact with other particles converting FEA::mesh to a triangular mesh
    auto contact_surf = chrono_types::make_shared<ChContactSurfaceMesh>(rubber_surf_mat);
    my_mesh->AddContactSurface(contact_surf);
    contact_surf->AddFacesFromBoundary(sphere_swept_thickness);
    
    // Add a ball
    auto mySphere = chrono_types::make_shared<ChBodyEasySphere>(1,      // radius
                                                   8,   // density
                                                   true,   // collision enabled
                                                   true,
                                                   steel_surf_mat);  // visualization enabled
    mySphere->SetPos( ChVector<>(3.95,0,0) );
    mySphere->SetPos_dt( ChVector<>(0,0,0) );

    my_system.Add(mySphere);

    my_system.Add(my_mesh);
    fea::ChMeshExporter::writeMesh(my_mesh, MESH_CONNECTIVITY);
    
    //////////////////////////////////////////////////////////////////////////////////

    auto solver = chrono_types::make_shared<ChSolverMINRES>();
    my_system.Set_G_acc(ChVector<>(980,0,0));
    my_system.SetSolver(solver);
    solver->SetMaxIterations(1000);
    solver->SetTolerance(1e-10);
    solver->EnableDiagonalPreconditioner(true);
    solver->SetVerbose(false);

    my_system.SetSolverForceTolerance(1e-10);

    //    mphysicalSystem.SetTimestepperType(ChTimestepper::Type::HHT);
    //    auto mystepper = std::static_pointer_cast<ChTimestepperHHT>(mphysicalSystem.GetTimestepper());
    //    mystepper->SetAlpha(-0.2);
    //    mystepper->SetMaxiters(1000);
    //    mystepper->SetAbsTolerances(1e-5);
    //    mystepper->SetMode(ChTimestepperHHT::POSITION);
    //    mystepper->SetScaling(true);

    //    mphysicalSystem.SetTimestepperType(ChTimestepper::Type::EULER_IMPLICIT);

    int stepEnd = 1000000;
    float time = 0;
    bool isAdaptive = false;
    
    for (int tStep = 0; tStep < stepEnd + 1; tStep++) {
        printf("\nstep : %d, time= %f (s), pos ball= %6f cm, force on ball = %6f", tStep, time, mySphere->GetPos().x(), mySphere->GetContactForce().x());
        
        my_system.DoStepDynamics(dT);
        time += dT;

        SaveParaViewFiles(my_system, my_mesh, tStep, time, mySphere->GetPos());
    }

   //my_system.SetupInitial() ;
    
    return 0;
}

//------------------------------------------------------------------
// Function to save the paraview files
//------------------------------------------------------------------

void SaveParaViewFiles(ChSystemSMC& mphysicalSystem,
                                        std::shared_ptr<fea::ChMesh> my_mesh,
                                        int next_frame,
                                        double mTime,
                                        const ChVector<>& ball) 
{
    static double exec_time;
    int out_steps = (int)ceil((1.0 / dT) / out_fps);
    exec_time += mphysicalSystem.GetTimerStep();
    int num_contacts = mphysicalSystem.GetNcontacts();
    double frame_time = 1.0 / out_fps;
    static int out_frame = 0;

        char SaveAsBuffer[256];  // The filename buffer.
        snprintf(SaveAsBuffer, sizeof(char) * 256, (demo_dir + "/flex_body.%d.vtk").c_str(), next_frame);
        char MeshFileBuffer[256];  // The filename buffer.
        snprintf(MeshFileBuffer, sizeof(char) * 256, ("%s"), MESH_CONNECTIVITY.c_str());
        fea::ChMeshExporter::writeFrame(my_mesh, SaveAsBuffer, MeshFileBuffer);
        out_frame++;

    static std::string header = "# vtk DataFile Version 2.0\nUnstructured Grid Example\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS 1 float";
        char fballname[200];
        sprintf(fballname, "sphere.%d.vtk", next_frame);
        char content[512];
        sprintf(content, "%s\n%6f %6f %6f", header.c_str(), ball.x(), ball.y(), ball.z());
        std::ofstream fball(fballname, std::ios::out);
        fball << content;
    
}
