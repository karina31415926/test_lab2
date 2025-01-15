#include "simulation/barnes_hut_simulation_with_collisions.h"
#include "simulation/barnes_hut_simulation.h"
#include "simulation/naive_parallel_simulation.h"
#include <omp.h>
#include <chrono>
#include <algorithm>

void BarnesHutSimulationWithCollisions::simulate_epochs(Plotter& plotter, Universe& universe, std::uint32_t num_epochs, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs) {
    for (int i = 0; i < num_epochs; i++) {
        simulate_epoch(plotter, universe, create_intermediate_plots, plot_intermediate_epochs);
    }
}

void BarnesHutSimulationWithCollisions::simulate_epoch(Plotter& plotter, Universe& universe, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs) {
    // Perform one epoch of Barnes-Hut simulation
    BarnesHutSimulation::simulate_epoch(plotter, universe, create_intermediate_plots, plot_intermediate_epochs);

    // Detect and handle collisions
    std::cout << "Running parallel collision detection..." << std::endl;
    find_collisions_parallel(universe);

    std::cout << "Running non-parallel collision detection..." << std::endl;
    find_collisions(universe);
}

void BarnesHutSimulationWithCollisions::find_collisions(Universe& universe) {
    auto start_time = std::chrono::high_resolution_clock::now();  // Start time measurement

    const double collision_distance_squared = 100000000000.0 * 100000000000.0;
    std::vector<std::size_t> bodies_to_remove;

    for (std::size_t i = 0; i < universe.num_bodies; ++i) {
        for (std::size_t j = i + 1; j < universe.num_bodies; ++j) {
            Vector2d<double> delta = universe.positions[i] - universe.positions[j];
            double distance_squared = delta[0] * delta[0] + delta[1] * delta[1];

            if (distance_squared < collision_distance_squared) {
                std::size_t heavier = (universe.weights[i] >= universe.weights[j]) ? i : j;
                std::size_t lighter = (heavier == i) ? j : i;

                double m1 = universe.weights[lighter];
                double m2 = universe.weights[heavier];

                universe.weights[heavier] = m1 + m2;
                universe.weights[lighter] = 0;

                m1 = 0;
                m2 = m1 + m2;
                universe.velocities[heavier] = (universe.velocities[lighter] * m1 + universe.velocities[heavier] * m2) / (m1 + m2);

                bodies_to_remove.push_back(lighter);
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();  // End time measurement
    std::chrono::duration<double> duration = end_time - start_time;  // Calculate duration
    std::cout << "Serial find_collisions took: " << duration.count() << " seconds" << std::endl;
}

void BarnesHutSimulationWithCollisions::find_collisions_parallel(Universe& universe) {
    double start_time = omp_get_wtime();  // Start time measurement using OpenMP

    const double collision_distance_squared = 100000000000.0 * 100000000000.0;
    std::vector<std::size_t> bodies_to_remove;
    std::vector<std::pair<std::size_t, std::size_t>> collisions;

#pragma omp parallel
    {
        std::vector<std::pair<std::size_t, std::size_t>> local_collisions;

#pragma omp for schedule(dynamic)
        for (std::size_t i = 0; i < universe.num_bodies; ++i) {
            for (std::size_t j = i + 1; j < universe.num_bodies; ++j) {
                Vector2d<double> delta = universe.positions[i] - universe.positions[j];
                double distance_squared = delta[0] * delta[0] + delta[1] * delta[1];

                if (distance_squared < collision_distance_squared) {
                    local_collisions.emplace_back(i, j);
                }
            }
        }

#pragma omp critical
        collisions.insert(collisions.end(), local_collisions.begin(), local_collisions.end());
    }

    // Resolving the collisions (same logic as before)
    // ...

    double end_time = omp_get_wtime();  // End time measurement using OpenMP
    double duration = end_time - start_time;  // Calculate duration
    std::cout << "Parallel find_collisions took: " << duration << " seconds" << std::endl;
}
