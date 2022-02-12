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

#include <random>

#include "chrono/physics/ChSystem.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/fea/ChElementShellBST.h"
#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChMeshExporter.h"
#include "chrono/fea/ChContactSurfaceMesh.h"
#include "chrono/fea/ChLoadContactSurfaceMesh.h" 
#include "chrono/geometry/ChTriangleMeshConnected.h"
#include "chrono/fea/ChContactSurfaceNodeCloud.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/solver/ChIterativeSolverLS.h"

#include "chrono/physics/ChLoadContainer.h"

#include <cmath>
#include <iostream>
#include <string>

#include "chrono/core/ChGlobal.h"
#include "chrono/core/ChQuaternion.h"     
#include "chrono/utils/ChUtilsSamplers.h"
#include "chrono/assets/ChTriangleMeshShape.h"

#include "chrono_gpu/ChGpuData.h"
#include "chrono_gpu/physics/ChSystemGpu.h"
#include "chrono_gpu/utils/ChGpuJsonParser.h"
#include "chrono_gpu/utils/ChGpuVisualization.h"

#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::fea;
using namespace chrono::gpu;
using namespace chrono::geometry;

bool testing = false;

std::string demo_dir = ".";
std::string MESH_CONNECTIVITY = "Flex_Mesh.vtk";

double dT = 1e-3; // time step
double sphere_swept_thickness = 0.008;

// Output frequency
float out_fps = 100;

// Enable/disable run-time visualization (if Chrono::OpenGL is available)
float render_fps = 2000;

// unit conversion from cgs to si
float F_CGS_TO_SI = 1e-5f;
float KE_CGS_TO_SI = 1e-7f;
float L_CGS_TO_SI = 1e-2f;
float P_CGS_TO_SI = F_CGS_TO_SI / L_CGS_TO_SI / L_CGS_TO_SI;

// sample information
ChVector<float> sample_center(0.f, 0.f, 0.f); //cm
float sample_hgt = 10.;  //cm
float sample_diam = 5.; //cm
float sample_rad = sample_diam / 2.f;

// triaxial cell information
ChVector<float> cyl_center(0.0f, 0.0f, 0.0f);
float cell_height = 8.5f;  //cm
float cell_diam = 5.f;  //cm
float cell_rad = cell_diam / 2.f;

float water_ratio = 0.633; // m_w / m_s
float sample_volume = M_PI * sample_diam * sample_diam / 4.f * sample_hgt; 
float sample_mass = 278.1; //g m_s + m_w
float sample_solid_mass = sample_mass / (1.f + water_ratio);

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}


void find_neighbors(int i0, int nrots, int nstacks, std::vector<int>& nghbrs){
    int istack = i0 / nrots;
    int irot = i0 % nrots;

    int i1 = istack * nrots + (irot+1) % nrots; // right
    int i2 = istack < nstacks - 1 ? (istack+1)*nrots + irot : -1; //top
    int i3 = istack < nstacks - 1 ? (istack+1)*nrots + (irot+1)%nrots : -1; //top-right
    int i4 = istack < nstacks -1 ? (istack+1)*nrots + (irot-1)%nrots:-1; //top-left
    int i5 = istack > 0 ? (istack-1)*nrots + (irot+1)%nrots : -1; //bottom-right
    int i6 = istack > 0 ? (istack-1)*nrots + (irot-1)%nrots : -1; //bottom-left
    int i7 = istack > 0 ? (istack-1)*nrots : -1; //bottom
    int i8 = istack * nrots + (irot-1)%nrots; //left

    nghbrs.clear();
    nghbrs.insert( nghbrs.end(), {i0,i1,i2,i3,i4,i5,i6,i7,i8});
}

ChVector<> cart2cyl_vector(ChVector<>& pos, ChVector<>& v){
    ChVector<> vcyl;

    double norm_pos = pos.x();
    double theta_pos = pos.y();

    double norm_v = sqrt( v.x() * v.x() + v.y() * v.y());
    double theta_v = acos( v.x() / norm_v);
    if (v.y() < 0) {theta_v = 2.f * M_PI - theta_v;}
    if (norm_v < 0.0000001) {theta_v = 0.f;}
    
    double cst = cos(theta_pos - theta_v);
    double snt = sin(theta_pos - theta_v);
    vcyl.Set( norm_v * cst, 
                norm_v * snt,
                v.z() );

    return vcyl;
}

int sign(float x){
    if (x<0){
        return -1;
    }
    else{
        return +1;
    }
}

void cart2cyl_vector(std::vector<ChVector<>>& pos, std::vector<ChVector<>>& v){
    unsigned int N = pos.size();
    for (unsigned int i = 0; i < N; i++){

        double norm_pos = pos[i].x();
        double theta_pos = pos[i].y();

        double norm_v = sqrt( v[i].x() * v[i].x() + v[i].y() * v[i].y());
        double theta_v = acos( v[i].x() / norm_v);
        if (v[i].y() < 0) {theta_v = 2.f * M_PI - theta_v;}
        if (norm_v < 0.0000001) {theta_v = 0.f;}
    
        double cst = cos(theta_pos - theta_v);
        double snt = sin(theta_pos - theta_v);
        v[i].Set( norm_v * cst, 
                    norm_v * snt,
                    v[i].z() );
    }
}

int get_contacting_meshes(const std::vector<ChVector<>>& pos, std::vector<unsigned int>& contacts){

    contacts.clear();
    unsigned int nmesh = pos.size();
    float maxz = pos[nmesh-1].z();
    for (unsigned int imesh=1; imesh<nmesh-1; imesh++){
        if (pos[imesh].z() < maxz){
            contacts.push_back(imesh);
        }
    }
    return contacts.size();
}

void get_radius_metrics(const std::vector<ChVector<>>& pos, float radii[3], const std::vector<unsigned int>& contacts){
    unsigned int nmesh = pos.size();
    float zmax = pos[nmesh-1].z();
    radii[0] = 1000.;
    radii[1] = -1000.;
    radii[2] = 0.;
    float tmp = 0.;
    for (unsigned int imesh : contacts){
        if (pos[imesh].x() < radii[0]){radii[0] = pos[imesh].x();}
        else {
            if (pos[imesh].x() > radii[1]){radii[1] = pos[imesh].x();}
        }
        radii[2] += pos[imesh].x();
    }
    if (contacts.size() == 0){std::cout << "Error!\n";}
    radii[2] /= (float) contacts.size();
}

void get_axial_radial_pressure(const std::vector<ChVector<>>& pos, const std::vector<ChVector<>>& forces, float radii[3], float p[2], const std::vector<unsigned int>& contacts){
    unsigned int nmesh = pos.size();
    float zmax = pos[nmesh-1].z();
    float zmin = pos[0].z();
    float h = zmax-zmin;
    float avg_face_P = 2. * M_PI * radii[2];
    float avg_face_A = M_PI * pow(radii[2],2);
    float sumFr = 0.;  
    for (unsigned int imesh : contacts){
        sumFr += forces[imesh].x();
    }
     p[0] = forces[nmesh-1].z()/avg_face_A;
     p[1] = sumFr / h / avg_face_P;
}

// *********************************************************************************************
void SaveParaViewFiles(ChSystemSMC& mphysicalSystem,
                                        std::shared_ptr<fea::ChMesh> my_mesh,
                                        int next_frame,
                                        double mTime,
                                        const ChVector<>&);

void myWriteMesh(std::shared_ptr<ChMesh>, int, int, std::string);
// ********************************************************************************************
class radialpressureloader : public ChLoaderForceOnSurface {
        public:
            radialpressureloader(std::shared_ptr<ChLoadableUV> mloadable) : ChLoaderForceOnSurface(mloadable){};

            virtual void ComputeF(const double U, const double V, ChVectorDynamic<>& F, 
                ChVectorDynamic<> *  	state_x,
		        ChVectorDynamic<> *  	state_w ){
                
                //std::cout << "\nIN" << std::endl;
                
                auto shell = std::dynamic_pointer_cast<ChElementShellBST>(this->loadable);
                    
                    ChVector<> shell_pos;
                    ChVector<> shell_vel;
                     if (state_x->size()>2) {
                        shell_pos = state_x->segment(0, 3);
                        shell_vel = state_w->segment(0, 3);
                    } else {
                        shell_pos = shell->GetNodeTriangleN(0)->GetPos();
                        shell_vel =  shell->GetNodeTriangleN(0)->GetPos_dt();
                    }
                    double norm = sqrt(shell_pos.x()*shell_pos.x() + shell_pos.y()*shell_pos.y());
                    double cs = shell_pos.x()/norm;
                    double sn = shell_pos.y()/norm;
                    
                    double area = shell->area;
                    double f_ext = this->p * area;
                    //std::cout << "\n " << norm << ", " << acos(cs)*180./M_PI << " area = " << area << std::endl;
                    //std::cout << "force = " << this->GetForce() << std::endl;
                    //std::cout << "F.sizer = " << F.size() << std::endl; 
                    F.segment(0,3) = (this->GetForce()  + ChVector<double>(f_ext * cs, f_ext * sn, 0.)).eigen();
                    //F.segment(3,6).setZero();
                    //std::cout << "force = " << F.segment(0,3) << std::endl;
                    //std::cout << "nelem = " << shell->GetNodeTriangleN(0)->GetPos().z() << std::endl;
                }
                // float GetPressure() {return this->p;}
                void setUniformRadialPressure(double ptarget){this->p = ptarget;}
            virtual bool isStiff() {return true;}

            private:
                double p = 0;
    };

int main(int argc, char* argv[]) {

    // ===============================================
    // 1. Read json paramater files
    // 2. Create ChSystemGpuMesh object
    // 3. Set simulation parameters
    // ===============================================

    ChGpuSimulationParameters params;

    if (argc != 2 || ParseJSON( argv[1], params) == false) {
        std::cout << "Usage:\n./demo_triaxial.json <json_file>" << std::endl;
        return 1;
    }

    const float Bx = params.box_X;
    const float By = Bx;
    const float Bz = params.box_Z;

    std::cout << "Box Dims: " << Bx << " " << By << " " << Bz << std::endl;

    float iteration_step = params.step_size;

    ChSystemGpuMesh gpu_sys(params.sphere_radius, 
                            params.sphere_density, 
                            make_float3(Bx, By, Bz), 
                            make_float3((float)0., (float)0., (float)0.));
    
    gpu_sys.SetKn_SPH2SPH(params.normalStiffS2S);
    gpu_sys.SetKn_SPH2WALL(params.normalStiffS2W);
    gpu_sys.SetKn_SPH2MESH(params.normalStiffS2M);

    gpu_sys.SetKt_SPH2SPH(params.tangentStiffS2S);
    gpu_sys.SetKt_SPH2WALL(params.tangentStiffS2W);
    gpu_sys.SetKt_SPH2MESH(params.tangentStiffS2M);

    gpu_sys.SetGn_SPH2SPH(params.normalDampS2S);
    gpu_sys.SetGn_SPH2WALL(params.normalDampS2W);
    gpu_sys.SetGn_SPH2MESH(params.normalDampS2M);

    gpu_sys.SetGt_SPH2SPH(params.tangentDampS2S);
    gpu_sys.SetGt_SPH2WALL(params.tangentDampS2W);
    gpu_sys.SetGt_SPH2MESH(params.tangentDampS2M);

    gpu_sys.SetCohesionRatio(params.cohesion_ratio);
    gpu_sys.SetAdhesionRatio_SPH2MESH(params.adhesion_ratio_s2m);
    gpu_sys.SetAdhesionRatio_SPH2WALL(params.adhesion_ratio_s2w);
    gpu_sys.SetFrictionMode(chrono::gpu::CHGPU_FRICTION_MODE::MULTI_STEP);

    gpu_sys.SetStaticFrictionCoeff_SPH2SPH(params.static_friction_coeffS2S);
    gpu_sys.SetStaticFrictionCoeff_SPH2WALL(params.static_friction_coeffS2W);
    gpu_sys.SetStaticFrictionCoeff_SPH2MESH(params.static_friction_coeffS2M);

    gpu_sys.SetOutputMode(params.write_mode);

    std::string out_dir = "./";
    filesystem::create_directory(filesystem::path(out_dir));
    out_dir = out_dir + params.output_dir;
    filesystem::create_directory(filesystem::path(out_dir));

    gpu_sys.SetTimeIntegrator(CHGPU_TIME_INTEGRATOR::CENTERED_DIFFERENCE);
    gpu_sys.SetFixedStepSize(params.step_size);
    gpu_sys.SetBDFixed(true);

    float volume_grain = pow(params.sphere_radius,3) * M_PI * 4.f/3.f;
    float mass_grain = params.sphere_density * volume_grain; 
    unsigned int num_create_spheres = round( sample_solid_mass / mass_grain );
    
    cell_diam = cell_diam + params.sphere_radius;
    cell_rad = cell_diam / 2.;

    // ***************************************************
    //
    // Create the FEA System as well as the Collision model
    //
    // ***************************************************
    
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    ChSystemSMC my_system;
    my_system.SetNumThreads(ChOMP::GetNumThreads(), 0, 1);
    my_system.Set_G_acc(ChVector<>(0, 0, 0.0));
    // Set default effective radius of curvature for all SCM contacts.
    collision::ChCollisionInfo::SetDefaultEffectiveCurvatureRadius(1);
    // collision::ChCollisionModel::SetDefaultSuggestedEnvelope(0.0); // not needed, already 0 when using ChSystemSMC
    collision::ChCollisionModel::SetDefaultSuggestedMargin(0.006);  // max inside penetration - if not enough stiffness in material: troubles
 
    // Create a mesh, that is a container for groups of elements and their referenced nodes.
    auto my_mesh = chrono_types::make_shared<ChMesh>();
    my_system.Add(my_mesh);


    auto surfmaterial = chrono_types::make_shared<ChMaterialSurfaceSMC>();
    surfmaterial->SetYoungModulus(6e4);
    surfmaterial->SetFriction(0.3f);
    surfmaterial->SetRestitution(0.2f);
    surfmaterial->SetAdhesion(0);
    
    auto surfmaterialhard = chrono_types::make_shared<ChMaterialSurfaceSMC>();
    surfmaterial->SetYoungModulus(6e7);
    surfmaterial->SetFriction(0.3f);
    surfmaterial->SetRestitution(0.2f);
    surfmaterial->SetAdhesion(0);
    // *****************************************************
    //
    // Create the top and bottom platens
    //
    // *****************************************************
    
    std::vector<std::shared_ptr<ChTriangleMeshConnected>> feameshes(0);
    std::vector<float> feameshmass(0);

    float scale_xy = cell_rad - std::min(0.02, (double) params.sphere_radius); // make sure this is smaller than the diameter of one particle
    float scale_z = cell_height * 0.01; 
    float3 scaling = make_float3(scale_xy, scale_xy, scale_z);
    ChMatrix33<double> platen_scale(ChVector<float>(scaling.x, scaling.y, scaling.z));

    ChVector<> platen_position(0,0,(-cell_height + scale_z)*0.5);

    auto botrigidmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
    auto toprigidmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
    botrigidmesh->LoadWavefrontMesh("unit_cylinder.obj", true, true); // make sure to include the normals
    toprigidmesh->LoadWavefrontMesh("unit_cylinder.obj", true, true);
    botrigidmesh->Transform(platen_position, platen_scale);
    toprigidmesh->Transform(platen_position+ChVector<>(cell_diam,0,cell_height), platen_scale);

    auto botPlatenBody = chrono_types::make_shared<ChBodyEasyMesh>(botrigidmesh, 8000, true, true, true, surfmaterialhard);
    auto topPlatenBody = chrono_types::make_shared<ChBodyEasyMesh>(toprigidmesh, 8000, true, true, true, surfmaterialhard);

    botPlatenBody->SetBodyFixed(true);

    feameshes.push_back( botrigidmesh);
    feameshes.push_back( toprigidmesh);
    feameshmass.push_back( 3000. );
    feameshmass.push_back( 3000.);

    std::cout << "Test 1" << std::endl;
    my_system.Add(botPlatenBody);
    my_system.Add(topPlatenBody);
    
    // ***************************************************
    //
    // Create nodes stack by stack
    //
    // ***************************************************

    unsigned int nrots = 120;
    unsigned int nstacks = 10;
    float dtheta = 2 * M_PI / nrots; //degrees
    float dh = cell_height / (float) nstacks;
    float tile_base = M_PI * (cell_diam * 1.00) / (float) nrots;// / (float) nrots;
    float tile_height = cell_height / (float) nstacks;// / (float) nstacks; // should be a multiple of cell_hgt
    unsigned int ntiles = nrots * nstacks;
    
    unsigned int nnodes = 2 * nrots + (nstacks  - 1) + (nstacks-1) * (nrots - 1); // number of nodes in an open cylindrical with 4 nodes per tile

    for (unsigned int istack = 0; istack < nstacks+1; istack++){

        float loc_z = -0.5 * cell_height + (float) (istack) * dh; 

        for (unsigned int irot = 0; irot < nrots; irot++){ // dont create nodes for the last element since they will be the first
            // Calculate angle of rotation
            float rot_angle_cw = -(float) irot * dtheta; // rotate clockwise
            float loc_x = cell_rad * cos(rot_angle_cw);
            float loc_y = cell_rad * sin(rot_angle_cw);
            float norm = sqrt(loc_x*loc_x + loc_y*loc_y);
            ChVector<> pos(loc_x, loc_y, loc_z);
            auto node = chrono_types::make_shared<ChNodeFEAxyz>(pos);

            // fix nodes along the z = -H/2 and z = H/2
            if (istack == 0 or istack == nstacks){node->SetFixed(true);}
            node->SetMass(0);

            my_mesh->AddNode(node);

        }
    }
    std::cout << "\nCreated " << my_mesh->GetNnodes() << " nodes\n";

    // *******************************************************
    //
    // Create an orthotropic material
    // Create a material.
    // This will be the material that the Kirchhoff cell is made of
    //
    // *******************************************************

        double density = 200;
        double E = 6e4;
        double nu = 0.0;
        double thickness = 0.01;

        auto melasticity = chrono_types::make_shared<ChElasticityKirchhoffIsothropic>(E,nu);
        auto material = chrono_types::make_shared<ChMaterialShellKirchhoff>(melasticity);
        material->SetDensity(density);
    
    // *******************************************************
    // 
    // Create the Elements Kirchhoff cells and assign the material
    // and nodes to them
    //
    // *******************************************************

    // std::vector<ChTriangleMeshConnected> radialmesh(1);
    // We need to create the surface normals and their indices manually 
    unsigned int nelemnts = 2 * nstacks * nrots; // 2 triangles per tile
    std::vector<int> nghbr;
    for (unsigned int itile=0; itile<ntiles; itile++){
        
        find_neighbors(itile, nrots, nstacks+1, nghbr); // add 1 because eventhough the element does not exist, the node still exists
        auto node0_lt = std::dynamic_pointer_cast<ChNodeFEAxyz>( my_mesh->GetNode(nghbr[0]) ); // all these are guaranteed to be valid
        auto node1_lt = std::dynamic_pointer_cast<ChNodeFEAxyz>( my_mesh->GetNode(nghbr[1]) );
        auto node2_lt = std::dynamic_pointer_cast<ChNodeFEAxyz>( my_mesh->GetNode(nghbr[2]) );
        auto node3_lt = std::dynamic_pointer_cast<ChNodeFEAxyz>( my_mesh->GetNode(nghbr[3]) );  
        auto node4_lt = std::dynamic_pointer_cast<ChNodeFEAxyz>( my_mesh->GetNode(nghbr[4]) );
        std::shared_ptr<ChNodeFEAxyz> node5_lt = nullptr;
        if (itile/nrots != 0)
            node5_lt =  std::dynamic_pointer_cast<ChNodeFEAxyz>( my_mesh->GetNode(nghbr[5]) );
        
        if (itile/nrots == 0){
            node0_lt->SetFixed(true);
            node1_lt->SetFixed(true);
        }
        if (itile/nrots == nstacks -1){
            node2_lt->SetFixed(true);
        }

        // printf("tile %d, triangle_lt, %d %d %d, %d %d %d", itile, node0_lt->GetIndex()-1, node1_lt->GetIndex()-1, node2_lt->GetIndex()-1,
        // node3_lt->GetIndex()-1, node4_lt->GetIndex()-1,  node5_lt != nullptr ? node5_lt->GetIndex()-1: -1);
        // std::cout << "\n---------------------------------------------------------------------\n";
        
        auto element_lt = chrono_types::make_shared<ChElementShellBST>();
        element_lt->SetNodes( node0_lt, node1_lt, node2_lt, node3_lt, node4_lt, node5_lt);
        element_lt->AddLayer(thickness, 0 * CH_C_DEG_TO_RAD, material); // add material layer, thicknes = 0.01, fiber angle = pi/2
        my_mesh->AddElement(element_lt);
        
        auto radialFEAMesh = chrono_types::make_shared<ChTriangleMeshConnected>();
        
        // push 4 nodes and 2 vectors for each GPU mesh because GPU calculates forces on meshes not mesh elements !
        // assume each fea mesh is 2 BST triangles -- > this affects how force is to be calculated in the FEA system
        radialFEAMesh->m_vertices.push_back( node0_lt->GetPos() );
        radialFEAMesh->m_vertices.push_back( node1_lt->GetPos() );
        radialFEAMesh->m_vertices.push_back( node2_lt->GetPos() );
        radialFEAMesh->m_vertices.push_back( node3_lt->GetPos() );
        radialFEAMesh->m_normals.push_back(-node0_lt->GetPos().GetNormalized());
        radialFEAMesh->m_normals.push_back(-node3_lt->GetPos().GetNormalized());
        
        radialFEAMesh->m_face_v_indices.push_back(ChVector<int>(0, 1, 2));
        radialFEAMesh->m_face_v_indices.push_back(ChVector<int>(3, 2, 1));
        radialFEAMesh->m_face_n_indices.push_back(ChVector<int>(0, 1, 0));
        radialFEAMesh->m_face_n_indices.push_back(ChVector<int>(1, 0, 1));
        feameshes.push_back(radialFEAMesh);
        feameshmass.push_back(1000.); // doesnt really matter
    }
    
    my_mesh->SetAutomaticGravity(false);
    std::cout << "Done adding meshes to vector  " << feameshes.back()->getNumTriangles() << " triangles" << std::endl;
    std::cout << "vector sizes  " << feameshes.size() << " " << feameshmass.size() << std::endl;
    std::cout << "Indices normals " << feameshes[1]->m_face_n_indices.size() << " "<<feameshes[1]->m_face_n_indices.size() << std::endl;
    // return 0;
    std::vector<ChTriangleMeshConnected> meshobjs;
    for (auto it = feameshes.begin(); it < feameshes.end(); it++){meshobjs.push_back(**it);} // so wasteful !
    gpu_sys.SetMeshes(meshobjs, feameshmass);
    //std::cout << "Test 2 " << std::endl;
    // gpu_sys.InitializeMeshes();
    // gpu_sys.WriteMeshes("testfeamesh");
    
    // *****************************************************************
    // 
    // Create the contact surface. This will handle collisions
    // We can only use a contact surface node cloud sine contact sirface mesh
    // does not support BST shells
    //
    // *****************************************************************

    // Add a contact surface to interact with other particles converting FEA::mesh to a triangular mesh
    auto surfmesh = chrono_types::make_shared<ChMeshSurface>();

    auto contact_surf = chrono_types::make_shared<ChContactSurfaceNodeCloud>(surfmaterial);
    my_mesh->AddContactSurface(contact_surf);
    contact_surf->AddAllNodes(0.005);
    printf("\n Added %d contact nodes to mesh \n", contact_surf->GetNnodes());

    // Create a mesh load for cosimulation, acting on the contact surface above
    // (forces on nodes will be computed by an external procedure)
    my_mesh->AddMeshSurface(surfmesh);
    myWriteMesh(my_mesh, nrots, nstacks, MESH_CONNECTIVITY);

    // *****************************************************************
    // 
    // Add a ball for testing
    //
    // *****************************************************************
    
    if (testing){
        auto steel_mat = chrono_types::make_shared<ChMaterialSurfaceSMC>();
        steel_mat->SetYoungModulus(200e9);
        steel_mat->SetPoissonRatio(0.3);
        steel_mat->SetRestitution(0.5);
        steel_mat->SetSfriction(0.4);
        steel_mat->SetKfriction(0.01);

        auto msphere = chrono_types::make_shared<ChBodyEasySphere>(2.5, 2, true, true, surfmaterial);
        msphere->SetPos(ChVector<>(7.8, 0.5, 0.6));
        my_system.Add(msphere);
    }

    // *******************************************************
    //
    // Add non-uniform random (gaussian) inner pressure
    //
    // *******************************************************

    if (testing){

        auto mloadcontainer = chrono_types::make_shared<ChLoadContainer>();
        my_system.Add(mloadcontainer);

        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.05e7,0.01e7);    //pressure centered at 10MPa with 2MPa standard deviation

        for (unsigned int ielem = nrots; ielem < ntiles-nrots; ielem++){
            double number = distribution(generator);        
            std::shared_ptr<ChLoad<radialpressureloader>> mpress(new ChLoad<radialpressureloader>(
            std::dynamic_pointer_cast<ChElementShellBST>(my_mesh->GetElement(ielem) ) ) );
            mpress->loader.setUniformRadialPressure(-0.2e4);
            mloadcontainer->Add(mpress);
         }
    }

    // ***************************************************************
    //
    // SOLVER
    //
    // ***************************************************************

        my_system.Setup();
        my_system.Update();
        auto solver = chrono_types::make_shared<ChSolverMINRES>();
        my_system.Set_G_acc(ChVector<>(0,0,0));
        my_system.SetSolver(solver);
        solver->SetMaxIterations(500);
        solver->SetTolerance(1e-5);
        solver->EnableDiagonalPreconditioner(true);
        solver->SetVerbose(false);
        my_system.SetSolverForceTolerance(1e-5);

if (testing){    
        int stepEnd = 200;
        float time = 0;
        bool isAdaptive = false;
        ChVector<> x0(0,0,0);
        for (int tStep = 0; tStep < stepEnd + 1; tStep++) {
            float norm = 0; //sqrt(pow(msphere->GetContactForce().x(),2) +  pow(msphere->GetContactForce().y(),2) +  pow(msphere->GetContactForce().z(),2));
        
            printf("\nstep : %d, time= %f (s), pos ball= %6f cm, force on ball = %6e", tStep, time, 0., norm);
        
            my_system.DoStepDynamics(dT);
            time += dT;

            SaveParaViewFiles(my_system, my_mesh, tStep, time, x0);
        }
   }

    // ======================================================
    //
    // Add the particles to the sim
    //
    // ======================================================    

    // initialize sampler, set distance between center of spheres as 2.1r
    gpu_sys.EnableMeshCollision(true);
    utils::PDSampler<float> sampler(2.1f * params.sphere_radius);
    std::vector<ChVector<float>> initialPos, initialVelo;

    float z_top = 0; 
    // randomize by layer
    ChVector<float> center(0.0f, 0.0f, z_top);
    // fill up each layer
    // particles start from 0 (middle) to cylinder_height/2 (top)
    size_t numSpheres = initialPos.size();
    
    while (numSpheres < num_create_spheres)  {
        auto points = sampler.SampleCylinderZ(center, sample_rad - 0.1*sample_rad, 0);
        initialPos.insert(initialPos.end(), points.begin(), points.end());
        center.z() += 2.1f * params.sphere_radius;
        numSpheres = initialPos.size();

    }
    numSpheres = initialPos.size();
    
    for (size_t i = 0; i < numSpheres; i++) {
        ChVector<float> velo(0,0,0);//-initialPos.at(i).x() / cell_rad, -initialPos.at(i).x() / cell_rad, 0.0f);
        initialVelo.push_back(velo);
    }

    gpu_sys.SetParticlePositions(initialPos, initialVelo);
    gpu_sys.SetGravitationalAcceleration(ChVector<float>(0, 0, -980));

    auto mloadcontainer = chrono_types::make_shared<ChLoadContainer>();
    //my_system.Add(mloadcontainer);

    for (unsigned int ielem = 0; ielem < ntiles; ielem++){
        std::shared_ptr<ChLoad<radialpressureloader>> mpress(new ChLoad<radialpressureloader>(
        std::dynamic_pointer_cast<ChElementShellBST>(my_mesh->GetElement(ielem) ) ) );
        mpress->loader.setUniformRadialPressure (0);
        mloadcontainer->Add(mpress);
    }
    // ===================================================
    //
    // Initialize
    //
    // ====================================================

    gpu_sys.Initialize();
    unsigned int nummeshes = gpu_sys.GetNumMeshes();
    std::cout << nummeshes << " meshes generated!" << std::endl;
    std::cout << "Created " << initialPos.size() << " spheres" << std::endl;
    
    // ===================================================
    //
    // Prepare main loop parameters
    //
    // ===================================================

    std::vector<ChVector<>> meshPostions, meshForces, meshTorques; 

    unsigned int nmeshes = gpu_sys.GetNumMeshes(); // 122 meshes (bottom + 120 side + top) TODO: consider changing side to also just one mesh 
    unsigned int out_steps = (unsigned int)(1.0f / (out_fps * iteration_step));
    unsigned int fps = (unsigned int)(1.0f / iteration_step);
    unsigned int render_steps = (unsigned int)(1.0 / (render_fps * iteration_step));
    unsigned int total_frames = (unsigned int)(params.time_end * fps);
    std::cout << "out_steps " << out_steps << std::endl;

    unsigned int step = 0;
    float curr_time = 0;
    float consolidation_time = 0.5;
    bool closed = false;

    // let system run for 0.5 second so the particles can settle
    while (curr_time < 0.6) {
        

        gpu_sys.CollectMeshContactForces(meshForces, meshTorques);
        //for (int imesh = nrots; imesh < nmeshes-2-nrots; imesh++) { 
        //    std::dynamic_pointer_cast<ChLoad<radialpressureloader>> (mloadcontainer->GetLoadList()[imesh])->loader.SetForce(meshForces[2+imesh]); // two fiirst GPU meshes are the platens
            //std::dynamic_pointer_cast<ChNodeFEAxyz>(my_mesh->GetNode(imesh))->SetForce(meshForces[2+imesh] - 10000.);
        //}

        if (curr_time > consolidation_time){
            if (not closed){
                ChVector<> close_shift( cell_diam, 0., 0.5*cell_height - gpu_sys.GetMaxParticleZ() + 1.5 * params.sphere_radius);
                ChMatrix33<> fill_platen(ChVector<>(1., 1., 1.)); // close_shift.z() ));
                feameshes[1]->Transform(-close_shift, fill_platen);
                closed = true;
            }
            else{
                //ChVector<> close_shift( 0., 0., 0.1);
                //ChMatrix33<> fill_platen(ChVector<>(1., 1., 1.)); // close_shift.z() ));
                //feameshes[1]->Transform(-close_shift, fill_platen);
            }
            
            my_system.DoStepDynamics(iteration_step);
            // update the geometry of the mesh
            for (int imesh = 2; imesh < feameshes.size()-2; imesh++){
                find_neighbors(imesh-2, nrots, nstacks+1, nghbr); // get the neighbors i.e. 4 corners of the tile (double triangle) 
                feameshes[imesh]->m_vertices[0] = std::dynamic_pointer_cast<ChNodeFEAxyz>(my_mesh->GetNode(nghbr[0]))->GetPos();
                feameshes[imesh]->m_vertices[1] = std::dynamic_pointer_cast<ChNodeFEAxyz>(my_mesh->GetNode(nghbr[1]))->GetPos();
                feameshes[imesh]->m_vertices[2] = std::dynamic_pointer_cast<ChNodeFEAxyz>(my_mesh->GetNode(nghbr[2]))->GetPos();
                feameshes[imesh]->m_vertices[3] = std::dynamic_pointer_cast<ChNodeFEAxyz>(my_mesh->GetNode(nghbr[3]))->GetPos();
                // no need to update the normals since they're going to be calculated by SetMeshes
                // no need to update connectivity
            }
            // SaveParaViewFiles(my_system, my_mesh, step, curr_time, VNULL);           
            meshobjs.clear();
            for (auto it = feameshes.begin(); it < feameshes.end(); it++){meshobjs.push_back(**it);} // so wasteful !
            gpu_sys.SetMeshes(meshobjs, feameshmass);
            gpu_sys.InitializeMeshes();
        }        
        if (step % out_steps == 0){

            // filenames for mesh, particles, force-per-mesh
            char filename[100], filenamemesh[100], filenameforce[100];;
            sprintf(filename, "%s/step%06d", out_dir.c_str(), step);
            sprintf(filenamemesh, "%s/main%06d", out_dir.c_str(), step);
            sprintf(filenameforce, "%s/meshforce%06d.csv", out_dir.c_str(), step);

            
            gpu_sys.WriteFile(std::string(filename));
            gpu_sys.WriteMeshes(filenamemesh);

            // force-per-mesh files
            // std::ofstream meshfrcFile(filenameforce, std::ios::out);
            // meshfrcFile << "#imesh, r, theta, z, f_x, f_y, f_z, f_r, f_theta\n";

            // Pull individual mesh forces
            // for (unsigned int imesh = 0; imesh < nmeshes; imesh++) {
            //     ChVector<> imeshforce;  // forces for each mesh
            //     ChVector<> imeshtorque; //torques for each mesh
            //     ChVector<> imeshforcecyl;
            //     ChVector<> imeshposition;

            //     // get the force on the ith-mesh
            //     gpu_sys.GetMeshPosition(imesh, imeshposition, 1);
            //     imeshforce *= F_CGS_TO_SI;                
                
            //     // change to cylinderical coordinates
            //     double normF = sqrt( imeshforce.x() * imeshforce.x() + imeshforce.y() * imeshforce.y());
            //     double thetaF = acos( imeshforce.x() / normF);
            //     if (imeshforce.y() < 0) {thetaF = 2.f * M_PI - thetaF;}
            //     if (normF < 0.0000001) {thetaF = 0.f;}
            //     double cst = cos(imeshposition.y() - thetaF);
            //     double snt = sin(imeshposition.y() - thetaF);
            //     imeshforcecyl.Set( normF * cst, 
            //                         normF * snt,
            //                         imeshforce.z() );

            //     // output to mesh file(s)
            //     char meshfforces[100];
            //     sprintf(meshfforces, "%d, %6f, %6f, %6f, %6f, %6f, %6f, %6f, %6f \n", imesh, 
            //         imeshposition.x(), imeshposition.y(), imeshposition.z(),
            //         imeshforce.x(), imeshforce.y(), imeshforce.z(),
            //         imeshforcecyl.x(), imeshforcecyl.y());
            //     meshfrcFile << meshfforces; 
            // }

            printf("time = %.4f\n", curr_time);
        }

        gpu_sys.AdvanceSimulation(iteration_step);
        curr_time += iteration_step;
        step++;

    };
    
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
        char SaveAsBuffer[256];  // The filename buffer.
        snprintf(SaveAsBuffer, sizeof(char) * 256, (demo_dir + "/flex_body.%d.vtk").c_str(), next_frame);
        char MeshFileBuffer[256];  // The filename buffer.
        snprintf(MeshFileBuffer, sizeof(char) * 256, ("%s"), MESH_CONNECTIVITY.c_str());
        fea::ChMeshExporter::writeFrame(my_mesh, SaveAsBuffer, MeshFileBuffer);

    //static std::string header = "# vtk DataFile Version 2.0\nUnstructured Grid Example\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS 1 float";
    //    char fballname[200];
    //    sprintf(fballname, "sphere.%d.vtk", next_frame);
    //    char content[512];
    //    sprintf(content, "%s\n%6f %6f %6f", header.c_str(), ball.x(), ball.y(), ball.z());
    //    std::ofstream fball(fballname, std::ios::out);
    //    fball << content;
    
}

void myWriteMesh( std::shared_ptr<ChMesh> mesh, int nrots, int nstacks,  std::string fname){
    std::ofstream fconnect(fname);
    
    int ntriangles = mesh->GetNelements();
    int ntiles = ntriangles * 2 ;
    fconnect << "\nCELLS " << ntiles << " " << ntiles*4;
    
    for (unsigned int itile=0; itile<ntriangles; itile++){
        char line[200];
        int node0 = itile;
        int node1 = (itile+1) % nrots == 0 ? itile/nrots*nrots : itile + 1;
        int node2 = (itile+1) % nrots ==0 ? itile + 1 : itile + 1 + nrots;
        int node3 = itile + nrots; 
        sprintf(line, "\n3 %d %d %d", node0, node1, node3 );
        fconnect << line;
        sprintf(line, "\n3 %d %d %d", node2, node3, node1 );
        fconnect << line;

    }
    fconnect << "\nCELL_TYPES " << ntiles;
    for (unsigned int itile = 0; itile<ntiles; itile++){
        fconnect << "\n9";
    }
}
