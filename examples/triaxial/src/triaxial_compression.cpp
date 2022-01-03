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

// sample information
ChVector<float> sample_center(0.f, 0.f, 0.f); //cm
float sample_hgt = 23.;  //cm
float sample_diam = 20.; //cm
float sample_rad = sample_diam / 2.f;

// triaxial cell information
ChVector<float> cyl_center(0.0f, 0.0f, 0.0f);
float cell_hgt = 25.f;  //cm
float cell_diam = 20.f;  //cm
float cell_rad = cell_diam / 2.f;

ChVector<> cart2cyl_vector(ChVector<>& pos, ChVector<>& v){
    ChVector<> vcyl;

    double norm_pos = sqrt( pos.x() * pos.x() + pos.y() * pos.y() );
    double theta_pos = acos( pos.x() / norm_pos );
    if (pos.y() < 0) {theta_pos = 2.f * M_PI - theta_pos;}
    if (norm_pos  == 0) { theta_pos = 0; }

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

    // ================================================
    //
    // Read and add the mesh(es) to the simulation
    //
    // ================================================
    
    
    float scale_xy = 2.f*cell_rad;
    float scale_z = cell_hgt; 
    float3 scaling = make_float3(scale_xy, scale_xy, scale_z);
    std::vector<float> mesh_masses;
    float mixer_mass = 10;

    std::vector<string> mesh_filenames;
    // std::vector<string> mesh_side_filenames;
    std::vector<ChMatrix33<float>> mesh_rotscales;
    ChMatrix33<float> mesh_scale(ChVector<float>(scaling.x, scaling.y, scaling.z));
    std::vector<float3> mesh_translations;


    // add bottom
    mesh_filenames.push_back("./models/unit_circle_+z.obj"); // add bottom slice
    mesh_rotscales.push_back(mesh_scale); // push scaling - no rotation
    mesh_translations.push_back(make_float3(cyl_center.x(), cyl_center.y(), -0.5f * scaling.z)); // push translation
    mesh_masses.push_back(mixer_mass); // push mass

    // add sides
    for (int i=0; i<120; ++i){
        mesh_filenames.push_back("./models/open_unit_cylinder_side_slab_120.obj"); 
        ChQuaternion<> quat = Q_from_AngAxis(i*3.f * CH_C_DEG_TO_RAD, VECT_Z); // rotate by 3°*i around z-axis 
        mesh_rotscales.push_back(mesh_scale * ChMatrix33<float>(quat)); // create rotation-scaling matrix
        mesh_translations.push_back(make_float3(cyl_center.x(), cyl_center.y(), cyl_center.z())); // no translation for side slab
        mesh_masses.push_back(mixer_mass); // push standard mass
    }

    // add top
    mesh_filenames.push_back("./models/unit_circle_-z.obj"); // add bottom slice
    mesh_rotscales.push_back(mesh_scale); // push scaling - no rotation
    mesh_translations.push_back(make_float3(cyl_center.x(), cyl_center.y(), +0.5f * scaling.z)); // push translation
    mesh_masses.push_back(mixer_mass); // push mass
    gpu_sys.LoadMeshes(mesh_filenames, mesh_rotscales, mesh_translations, mesh_masses);
    // gpu_sys.LoadMeshes(mesh_side_filenames, mesh_rotscales, mesh_translations, mesh_masses);
        
    std::cout << gpu_sys.GetNumMeshes() << " meshes" << std::endl;

    // ======================================================
    //
    // Add the particles to the sim
    //
    // ======================================================    

    // initialize sampler, set distance between center of spheres as 2.1r
    utils::PDSampler<float> sampler(2.1f * params.sphere_radius);
    std::vector<ChVector<float>> initialPos;

    // randomize by layer
    ChVector<float> center(0.0f, 0.0f, -sample_hgt/2.f);
    // fill up each layer
    // particles start from 0 (middle) to cylinder_height/2 (top)
    while (center.z() + params.sphere_radius < sample_hgt / 2.0f )  {
        auto points = sampler.SampleCylinderZ(center, sample_rad - params.sphere_radius, 0);
        initialPos.insert(initialPos.end(), points.begin(), points.end());
        center.z() += 2.1f * params.sphere_radius;
    }

    size_t numSpheres = initialPos.size();
    
    // create initial velocity vector
    std::vector<ChVector<float>> initialVelo;
    for (size_t i = 0; i < numSpheres; i++) {
        ChVector<float> velo(-initialPos.at(i).x() / cell_rad, -initialPos.at(i).x() / cell_rad, 0.0f);
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
    ChVector<> topPlate_forces; // forces on the top plate
    ChVector<> topPlate_torques; // forces on the top plate
    ChVector<> topPlate_offset(0.0f, 0.0f, - cell_hgt + abs( gpu_sys.GetMaxParticleZ() ) + 5.f * params.sphere_radius); // initial top plate position
    float topPlate_moveTime = curr_time;
    ChQuaternion<float> q0(1,0,0,0);
    
    // top plate move downward with velocity 1cm/s
    ChVector<> topPlate_vel(0.f, 0.f, -10.f);
    ChVector<> topPlate_ang(0.f, 0.f, 0.f);

    std::function<ChVector<float>(float)> topPlate_posFunc = [&topPlate_offset, &topPlate_vel, &topPlate_moveTime](float t){
        ChVector<> pos(topPlate_offset);
        if (t > topPlate_moveTime){
            pos.Set(topPlate_offset.x() + topPlate_vel.x() * (t - topPlate_moveTime),  
                    topPlate_offset.y() + topPlate_vel.y() * (t - topPlate_moveTime),  
                    topPlate_offset.z() + topPlate_vel.z() * (t - topPlate_moveTime) );
        }
        return pos;
    };

    // side plate move inward with velocity 1cm/s
    float sidePlate_radial_vel = -2.f;  // cm.s-1
    float sidePlate_moveTime = curr_time;
    ChVector<> v0(0.f, 0.f, 0.f);  // place-holder
    ChVector<> w0(0.f, 0.f, 0.f);  // place-holder

    std::function<ChVector<float>(float, ChVector<>&)> sidePlate_advancePos = [&sidePlate_radial_vel, &sidePlate_moveTime](float t, ChVector<>& pos){ 
        ChVector<float> delta(0.f,0.f, 0.f);
        float x = pos.x();
        float y = pos.y();
        float z = pos.z();
        float r = sqrt(x*x + y*y);
        if (r==0) { return delta; }
        float cstheta = x / r;
        float sntheta = y / r;
        float dx = (t - sidePlate_moveTime) * sidePlate_radial_vel * cstheta;
        float dy = (t - sidePlate_moveTime) * sidePlate_radial_vel * sntheta;
        delta.Set(dx,dy,0.f);
        return delta;
    };
     
    // continue simulation until the end
    ChVector<> myv, shift;
    std::vector<ChVector<>> meshForces, meshTorques, meshPositions;
    for (unsigned int i=0; i < nmeshes; i++){
        meshForces.push_back(ChVector<float>(0.,0.,0.));
        meshTorques.push_back(ChVector<float>(0.,0.,0.));
        meshPositions.push_back(ChVector<float>(0.,0.,0.));
    }

    while (curr_time < params.time_end) {
        printf("rendering frame: %u of %u, curr_time: %.4f, ", step + 1, total_frames, curr_time);
        // Move side plates
        for (unsigned int i=1; i<nmeshes-1; i++){
            gpu_sys.GetMeshPosition(i, meshPositions[i], 0);
            shift.Set(sidePlate_advancePos( curr_time, meshPositions[i] ));
            gpu_sys.ApplyMeshMotion(i,shift,q0, v0, w0); 
        }
        
        // Move top plate
        ChVector<> topPlate_pos(topPlate_posFunc(curr_time));
//        gpu_sys.ApplyMeshMotion(nmeshes-1, topPlate_pos, q0, topPlate_vel, topPlate_ang);

        // write position
        gpu_sys.AdvanceSimulation(iteration_step);

//        std::cout << "top plate pos_z: " << topPlate_pos.z() << " cm";

        nc = gpu_sys.GetNumContacts();
        std::cout << ", numContacts: " << nc;

        for (unsigned int i = 0; i < nmeshes; i++){
            gpu_sys.CollectMeshContactForces(i, meshForces[i], meshTorques[i]);  // get forces
            meshForces[i].Set(cart2cyl_vector(meshPositions[i], meshForces[i])); // change to cylindrical
            meshForces[i] *= F_CGS_TO_SI;                                        // change to SI
        }
        //std::cout << ", top plate force: " << topPlate_forces.z() * F_CGS_TO_SI << " Newton";
        //std::cout << "\n";

        float radial_press = 0.f;
        for (unsigned int i=1; i < nmeshes-1; i++){
            radial_press += meshForces[i].x(); // r-component
        }
        radial_press /= gpu_sys.GetMaxParticleZ() * M_PI * (cell_diam - 2.f*(curr_time-sidePlate_moveTime)*sidePlate_radial_vel)*1e-2; // N.m-2=Pa
        std::cout << "\nradial pressure = " << radial_press / 1000.f << "kPa\n";

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

        step++;
        curr_time += iteration_step;
    }
    
    return 0;
    
}
