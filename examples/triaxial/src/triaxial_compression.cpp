// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2019 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Nic Olsen
// =============================================================================
// Chrono::Gpu evaluation of several simple mixer designs. Material consisting
// of spherical particles is let to aggitate in a rotating mixer. Metrics on the
// performance of each mixer can be determined in post-processing.
// =============================================================================

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

#include "pid.h"

using namespace chrono;
using namespace chrono::gpu;

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
float cell_hgt = 8.5f;  //cm
float cell_diam = 5.f;  //cm
float cell_rad = cell_diam / 2.f;

float water_ratio = 0.633; // m_w / m_s
float sample_volume = M_PI * sample_diam * sample_diam / 4.f * sample_hgt; 
float sample_mass = 278.1; //g m_s + m_w
float sample_solid_mass = sample_mass / (1.f + water_ratio);

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
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

    // ================================================
    //
    // Read and add the mesh(es) to the simulation
    //
    // ================================================
    
    
    float scale_xy = 2.f*cell_rad;
    float scale_z = cell_hgt; 
    float3 scaling = make_float3(scale_xy, scale_xy, scale_z);
    std::vector<float> mesh_masses;
    float mixer_mass = 1000;

    std::vector<string> mesh_filenames;
    // std::vector<string> mesh_side_filenames;
    std::vector<ChMatrix33<float>> mesh_rotscales;
    ChMatrix33<float> mesh_scale(ChVector<float>(scaling.x, scaling.y, scaling.z));
    ChMatrix33<float> hopper_scale(ChVector<float>(1.5 * scaling.x, 1.5 * scaling.y, 0.5 * scaling.z));
    std::vector<float3> mesh_translations;


    // add hopper
    //mesh_filenames.push_back("./models/unit_cone_10to1.obj");
    //mesh_rotscales.push_back(hopper_scale); // hopper has same radius as cell
    //mesh_translations.push_back(make_float3(cyl_center.x(), cyl_center.y(), cell_hgt/2.f + 5.f)); // move the hopper 5cm above the cell
    //mesh_masses.push_back(mixer_mass);

    // add bottom
    //ChVector<float> topWallPos(0.0f, 0.0f, -0.5 * scaling.z);
    //ChVector<float> topWallNrm(0.0f, 0.0f, 1.0f);
    //size_t bottomWall = gpu_sys.CreateBCPlane(topWallPos, topWallNrm, true);
    
    mesh_filenames.push_back("./models/unit_circle_+z.obj"); // add bottom slice
    mesh_rotscales.push_back(mesh_scale); // push scaling - no rotation
    mesh_translations.push_back(make_float3(cyl_center.x(), cyl_center.y(), -0.5 * scaling.z)); // push translation
    mesh_masses.push_back(mixer_mass); // push mass

    // add sides
    // for (int i=0; i<120; ++i){
    //     mesh_filenames.push_back("./models/open_unit_cylinder_side_slab_120.obj"); 
    //     ChQuaternion<> quat = Q_from_AngAxis(i*3.f * CH_C_DEG_TO_RAD, VECT_Z); // rotate by 3°*i around z-axis 
    //     mesh_rotscales.push_back(mesh_scale * ChMatrix33<float>(quat)); // create rotation-scaling matrix
    //     mesh_translations.push_back(make_float3(cyl_center.x(), cyl_center.y(), cyl_center.z())); // no translation for side slab
    //     mesh_masses.push_back(mixer_mass); // push standard mass
    // }
    unsigned int nrots = 60;
    unsigned int nstacks = 20;
    float dtheta = 360. / nrots; //degrees
    float dz = cell_hgt / nstacks;
    float tile_base = M_PI * (cell_diam * 1.00) / (float) nrots;// / (float) nrots;
    float tile_height = cell_hgt / (float) nstacks;// / (float) nstacks; // should be a multiple of cell_hgt
    unsigned int ntiles = nrots * nstacks;
    ChMatrix33<float> tile_scale(ChVector<float>(1., tile_base, tile_height));

    for (unsigned int i = 0; i < ntiles; i++){
        float rot_ang = (float) (i/nstacks) * dtheta * CH_C_DEG_TO_RAD;
        ChQuaternion<> quatRot = Q_from_AngAxis( rot_ang, VECT_Z); // stacked ntriangles
        mesh_filenames.push_back("./models/unit_tritile_-y.obj");
        mesh_rotscales.push_back(ChMatrix33<float>(quatRot) * tile_scale);
        mesh_translations.push_back(make_float3(cell_rad*cos(rot_ang), cell_rad*sin(rot_ang), 0.5 * (tile_height - cell_hgt) + (float) (i%nstacks) * dz));
        mesh_masses.push_back(mixer_mass);
    }
    //mesh_filenames.push_back("./models/double_open_unit_cylinder.obj");
    //mesh_rotscales.push_back(mesh_scale);
    //mesh_translations.push_back(make_float3(0,0,0));
    //mesh_masses.push_back(mixer_mass);

    // add top
    mesh_filenames.push_back("./models/unit_circle_-z.obj"); // add bottom slice
    mesh_rotscales.push_back(mesh_scale); // push scaling - no rotation
    mesh_translations.push_back(make_float3(cyl_center.x(), cyl_center.y(), params.box_Z/2.f - 5.)); // push translation top top of box
    mesh_masses.push_back(mixer_mass); // push mass

    gpu_sys.LoadMeshes(mesh_filenames, mesh_rotscales, mesh_translations, mesh_masses); 
    std::cout << gpu_sys.GetNumMeshes() << " meshes" << std::endl;

    // ======================================================
    //
    // Add the particles to the sim
    //
    // ======================================================    

    // initialize sampler, set distance between center of spheres as 2.1r
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

    // ===================================================
    //
    // Initialize
    //
    // ====================================================

    gpu_sys.EnableMeshCollision(true);    
    gpu_sys.Initialize();
    unsigned int nummeshes = gpu_sys.GetNumMeshes();
    std::cout << nummeshes << " meshes generated!" << std::endl;
    std::cout << "Created " << initialPos.size() << " spheres" << std::endl;
    
    gpu_sys.WriteMeshes(out_dir + "/init.vtk");
    // ===================================================
    //
    // Prepare main loop parameters
    //
    // ===================================================
    
    unsigned int nmeshes = gpu_sys.GetNumMeshes(); // 122 meshes (bottom + 120 side + top) TODO: consider changing side to also just one mesh 
    unsigned int out_steps = (unsigned int)(1.0f / (out_fps * iteration_step));
    unsigned int fps = (unsigned int)(1.0f / iteration_step);
    unsigned int render_steps = (unsigned int)(1.0 / (render_fps * iteration_step));
    unsigned int total_frames = (unsigned int)(params.time_end * fps);
    std::cout << "out_steps " << out_steps << std::endl;

    unsigned int step = 0;
    float curr_time = 0;

    // let system run for 0.5 second so the particles can settle
    while (curr_time < 0.5) {
        
        if (step % out_steps == 0){

            // filenames for mesh, particles, force-per-mesh
            char filename[100], filenamemesh[100], filenameforce[100];;
            sprintf(filename, "%s/step%06d", out_dir.c_str(), step);
            sprintf(filenamemesh, "%s/main%06d", out_dir.c_str(), step);
            sprintf(filenameforce, "%s/meshforce%06d.csv", out_dir.c_str(), step);

            gpu_sys.WriteFile(std::string(filename));
            gpu_sys.WriteMeshes(filenamemesh);

            // force-per-mesh files
            std::ofstream meshfrcFile(filenameforce, std::ios::out);
            meshfrcFile << "#imesh, r, theta, z, f_x, f_y, f_z, f_r, f_theta\n";

            // Pull individual mesh forces
            for (unsigned int imesh = 0; imesh < nmeshes; imesh++) {
                ChVector<> imeshforce;  // forces for each mesh
                ChVector<> imeshtorque; //torques for each mesh
                ChVector<> imeshforcecyl;
                ChVector<> imeshposition;

                // get the force on the ith-mesh
                gpu_sys.CollectMeshContactForces(imesh, imeshforce, imeshtorque);
                gpu_sys.GetMeshPosition(imesh, imeshposition, 1);
                imeshforce *= F_CGS_TO_SI;                
                
                // change to cylinderical coordinates
                double normF = sqrt( imeshforce.x() * imeshforce.x() + imeshforce.y() * imeshforce.y());
                double thetaF = acos( imeshforce.x() / normF);
                if (imeshforce.y() < 0) {thetaF = 2.f * M_PI - thetaF;}
                if (normF < 0.0000001) {thetaF = 0.f;}
                double cst = cos(imeshposition.y() - thetaF);
                double snt = sin(imeshposition.y() - thetaF);
                imeshforcecyl.Set( normF * cst, 
                                    normF * snt,
                                    imeshforce.z() );

                // output to mesh file(s)
                char meshfforces[100];
                sprintf(meshfforces, "%d, %6f, %6f, %6f, %6f, %6f, %6f, %6f, %6f \n", imesh, 
                    imeshposition.x(), imeshposition.y(), imeshposition.z(),
                    imeshforce.x(), imeshforce.y(), imeshforce.z(),
                    imeshforcecyl.x(), imeshforcecyl.y());
                meshfrcFile << meshfforces; 
            }

            printf("time = %.4f\n", curr_time);
        }

        gpu_sys.AdvanceSimulation(iteration_step);
        curr_time += iteration_step;
        step++;

    };
    // ============================================
    //
    // Compression
    //
    //=============================================
    // Useful information
    unsigned int nc=0; // number of contacts
    float step_size = params.step_size;
    float topPlate_moveTime = curr_time;
    
    Eigen::MatrixXf mesh_ticks(total_frames - step, 2*nmeshes);
    std::ofstream fticks(out_dir+"/fticks.csv", std::ios::out); 
    
    ChVector<> topPlate_offset(0.0f, 0.0f, -(params.box_Z/2.f - 5.f + cell_hgt/2.f) + (gpu_sys.GetMaxParticleZ() + cell_hgt/2.f) + params.sphere_radius); // initial top plate position
    mesh_ticks(0,2*nmeshes-2) = topPlate_offset.z();
    for (unsigned int i = 0; i < 2*nmeshes-2; i++){mesh_ticks(0,i)=0.;}
    ChQuaternion<float> q0(1,0,0,0);
    
    // top plate move downward with velocity 1cm/s
    ChVector<> topPlate_vel(0.f, 0.f, -.75f);
    ChVector<> topPlate_ang(0.f, 0.f, 0.f);
    std::function<ChVector<>(unsigned int, float)> topPlate_posFunc = [&topPlate_vel, &topPlate_moveTime, &step_size, &mesh_ticks](unsigned int istep, float gamma){
        ChVector<> shift(0, 0, 0);
        shift.Set(0, 0, mesh_ticks(istep, mesh_ticks.cols()-1) + gamma * topPlate_vel.z() * step_size);
        mesh_ticks(istep+1, mesh_ticks.cols()-1) = shift.z();
        return shift;
    };

    // side plate move inward with velocity 1cm/s
    float sidePlate_moveTime = curr_time;
    float tile_radial_vel = -.75; // max speed is cm.s-1
    std::function<ChVector<>(ChVector<>&, unsigned int, unsigned int, float)> tile_advancePosDr = 
    [&tile_radial_vel, &sidePlate_moveTime, &step_size, &mesh_ticks](ChVector<>& pos, unsigned int istep, unsigned int imesh, float gamma){ 
        ChVector<> delta(0.f, 0.f, 0.f);
        float x = pos.x();
        float y = pos.y();
        float z = pos.z();
        float r = sqrt(x*x + y*y);
        if (r==0) { return delta; }
        float cstheta = x / r;
        float sntheta = y / r;
        float dx = mesh_ticks(istep, 2*imesh) + gamma * step_size * tile_radial_vel * cstheta;
        float dy = mesh_ticks(istep, 2*imesh+1) + gamma * step_size * tile_radial_vel * sntheta;
        mesh_ticks(istep+1, 2*imesh) = dx;
        mesh_ticks(istep+1, 2*imesh+1) = dy;
        delta.Set(dx,dy,0.f);
        return delta;
    };
     
    // create vectors to hold useful information on meshes
    ChVector<> myv, shift, v0, w0; v0.Set(0,0,0); w0.Set(0,0,0);
    std::vector<ChVector<>> meshForces(nmeshes), meshTorques(nmeshes), meshPositions(nmeshes);
    std::vector<unsigned int> contacting_meshes;

    gpu_sys.GetMeshPositions(meshPositions, 1);
    float new_cell_radii[3]; 
    float top_cell_new_rad;
    get_contacting_meshes(meshPositions, contacting_meshes);
    get_radius_metrics(meshPositions, new_cell_radii, contacting_meshes);

    // pressure information
    gpu_sys.CollectMeshContactForces(meshForces, meshTorques);
    float sigma3 = 500.f; //consolidating pressure // Pa, consolidation stress
    float average_xr_press[2];
    get_axial_radial_pressure(meshPositions, meshForces, new_cell_radii, average_xr_press, contacting_meshes);
    cart2cyl_vector(meshPositions, meshForces);

    float top_press_diff = sigma3 - average_xr_press[0] * P_CGS_TO_SI;
    float radial_press_diff = sigma3 - average_xr_press[1] * P_CGS_TO_SI;
    float max_tick, avg_tick, min_tick;
    char tickout[500];

    float sphere_vol = 4./3.*M_PI*pow(params.sphere_radius,3);
    unsigned int step0 = step;
    float solid_ratio = numSpheres*sphere_vol / cell_hgt / M_PI / pow(cell_rad,2.);

//    float press_rate = 1.; // pressure speed Pa/s
//    float press_accl = .1; // pressure acceleration Pa/s^2
    float Kp_r = tile_radial_vel*params.step_size/sigma3; //tile_radial_vel / press_rate; // cm/Pa
    float Kp_x = topPlate_vel.z()*params.step_size/sigma3; //topPlate_vel.z() / press_rate;
    float Kd_r = Kp_r/5.; //tile_radial_vel / press_accl;
    float Kd_x = Kp_x/5.; //topPlate_vel.z() / press_accl;
    
    float max_radial_step = -10. * params.step_size * tile_radial_vel;
    float min_radial_step =  10. * params.step_size * tile_radial_vel;
    float max_axial_step =  -10. * params.step_size * topPlate_vel.z();
    float min_axial_step =   10. * params.step_size * topPlate_vel.z(); 
    std::vector<PID> pid_controllers;
    
    pid_controllers.emplace_back(params.step_size, max_axial_step, min_axial_step, Kp_x, Kd_x, 0.) ;
    for (unsigned int i =1; i < nmeshes-1; i++){
        pid_controllers.emplace_back(params.step_size, max_radial_step, min_radial_step, Kp_r, Kd_r, 0.);
    }
    pid_controllers.emplace_back(params.step_size, max_axial_step, min_axial_step, Kp_x, Kd_x, 0. ); 
    
    
    /*
     * Main loop thermo infor
     */
    int n;
    fticks << "step, curr_time, contacts, top_ticks, axial_ticks top_press, axial_press";
    
    printf("Main loop starting:\n");
    n = printf("\n%-10s | %-10s | %-10s | %-11s | %-11s | %-11s | %-10s | %-40s", 
    "step", "curr_time", "contacts", "solid_ratio","av. pzz", "av. prr", "pos_z", "radius (min,max,avg,top)");
    printf("\n(/%-7d) | %-10s | %-10s | %-11s | %-11s | %-11s | %-10s | %-30s", 
    total_frames, "   (s)   ", "    (#)   ", "   (1)   ","  (kPA)  ", "  (kPa)  ", "  (cm) ", "  (cm)");
    string tmps = "\n";
    for (unsigned int i=0; i<n; i++){tmps += "-";}
    std::cout << tmps;; 
    
    /* 
    *  ------------------------------
    *         Main loop starts
    *  ------------------------------
    */    
    
    while (curr_time < params.time_end) {
        // Collect mesh positions and forces
        unsigned int dstep = step - step0;
        contacting_meshes.clear();

        gpu_sys.GetMeshPositions(meshPositions, 1);
        gpu_sys.CollectMeshContactForces(meshForces, meshTorques);  // get forces
        cart2cyl_vector(meshPositions, meshForces);
        
        get_contacting_meshes(meshPositions, contacting_meshes);
        get_radius_metrics(meshPositions, new_cell_radii, contacting_meshes);
        get_axial_radial_pressure(meshPositions, meshForces, new_cell_radii, average_xr_press,contacting_meshes);

        float move_r = 1.;
        float move_x = 1.;
        int sgr = sign(sigma3 - abs(average_xr_press[1])*P_CGS_TO_SI);
        int sgx = sign(sigma3 - abs(average_xr_press[0])*P_CGS_TO_SI);
        if (abs(average_xr_press[0] / average_xr_press[1]) < 0.95 and sgx == sgr ){
            move_r = 0.;
        }
        if (abs(average_xr_press[1] / average_xr_press[0]) < 0.95 and sgx == sgr){
            move_x = 0.;
        }
        float tile_press_diff = 0.;
        min_tick =  1000.;
        max_tick = -1000.;
        avg_tick = 0.;
        float dr = 0;
        float dz = 0;
        float dx = 0;
        float dy = 0;
        float min_tile_press_diff = 1e9;
        float max_tile_press_diff = -1e9;
        float avg_tile_press_diff = 0;
        float tile_press = 0.;
        for (unsigned int imesh : contacting_meshes){

            tile_press = abs( meshForces[imesh].x() )/tile_base/tile_height*P_CGS_TO_SI;
            tile_press_diff = sigma3 - tile_press;
            
            dr = pid_controllers[imesh].calculate(sigma3, tile_press );
            dx = dr * cos(meshPositions[imesh].y());
            dy = dr * sin(meshPositions[imesh].y());
            shift.Set( mesh_ticks(dstep, 2*imesh)+dx, mesh_ticks(dstep, 2*imesh+1)+dy, 0. );
            //std::cout << "\ni = " << imesh << ", shift = (" << shift.x() << "," << shift.y() << ")\n"; 
            gpu_sys.ApplyMeshMotion(imesh, shift, q0, v0, w0);
            mesh_ticks(dstep+1,2*imesh) = shift.x();
            mesh_ticks(dstep+1, 2*imesh+1) = shift.y();
                
            if (max_tick < dr){max_tick = dr;}
            else{if (min_tick > dr){min_tick = dr;}}
            if (tile_press_diff < min_tile_press_diff){min_tile_press_diff = tile_press_diff;}
            else{if(tile_press_diff > max_tile_press_diff){max_tile_press_diff = tile_press_diff;}}

            avg_tick += dr;
            avg_tile_press_diff += tile_press_diff;
        }
        avg_tick /= contacting_meshes.size();
        avg_tile_press_diff /= contacting_meshes.size();




        top_press_diff = sigma3 - abs(average_xr_press[0]) * P_CGS_TO_SI;
        dz = pid_controllers[nmeshes-1].calculate(sigma3, abs(average_xr_press[0] * P_CGS_TO_SI));
        shift.Set(0., 0., mesh_ticks(dstep, 2*nmeshes-2)+dz);
        gpu_sys.ApplyMeshMotion(nmeshes-1, shift, q0, v0, w0);
        mesh_ticks(dstep+1, 2*nmeshes-2) = shift.z();   

        solid_ratio = numSpheres * sphere_vol / (meshPositions[nmeshes-1].z()+cell_hgt/2.) / M_PI / (pow(new_cell_radii[2],2));
        // write position
        nc = gpu_sys.GetNumContacts();

        sprintf(tickout, "\n%d, %6f, %d, %6f, %6f, %6f, %6f, %6f, %6f, %6f, %6f, %6f", 
        step-step0, curr_time, nc,
        meshPositions[nmeshes-1].z(), new_cell_radii[0], new_cell_radii[1], new_cell_radii[2], 
        min_tick, max_tick, avg_tick,
        average_xr_press[0], average_xr_press[1]);
        fticks << tickout;
        
        
        if (step % out_steps == 0){

            printf("\n%-10d | %-10.6f | %-10d | %-11.9f | %-6.5e | %-6.5e | %-10.8f | %-10.8f, %-10.8f, %-10.8f, %-10.8f | %-5.4f; %-5.4f | %-8.7f, %-8.7f, %8.7f %-8.7f", 
            step, curr_time, nc, solid_ratio, 
            average_xr_press[0]*P_CGS_TO_SI/1000., average_xr_press[1]*P_CGS_TO_SI/1000.,
            meshPositions[nmeshes-1].z(),
            new_cell_radii[0], new_cell_radii[1], new_cell_radii[2], 0.0,
            top_press_diff, avg_tile_press_diff,
            dz, dr, move_x, move_r);


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

            // // Pull individual mesh forces
            // for (unsigned int imesh = 0; imesh < nmeshes; imesh++) {
            //     ChVector<> imeshforce;  // forces for each mesh
            //     ChVector<> imeshtorque; //torques for each mesh
            //     ChVector<> imeshforcecyl;
            //     ChVector<> imeshposition;

            //     // get the force on the ith-mesh
            //     gpu_sys.CollectMeshContactForces(imesh, imeshforce, imeshtorque);
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
            if (step % 10*out_fps == 0){fticks.flush();}
        }

        gpu_sys.AdvanceSimulation(iteration_step);
        step++;
        curr_time += iteration_step;
    }
    
    return 0;
    
}
