#include "simulation/barnes_hut_simulation_with_collisions.h"
#include "simulation/barnes_hut_simulation.h"
#include "simulation/naive_parallel_simulation.h"
#include <omp.h>

void BarnesHutSimulationWithCollisions::simulate_epochs(Plotter& plotter, Universe& universe, std::uint32_t num_epochs, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs) {
    for (int i = 0; i < num_epochs; i++) {
        simulate_epoch(plotter, universe, create_intermediate_plots, plot_intermediate_epochs);
    }
}

void BarnesHutSimulationWithCollisions::simulate_epoch(Plotter& plotter, Universe& universe, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs) {
    // Perform one epoch of Barnes-Hut simulation
    BarnesHutSimulation::simulate_epoch(plotter, universe, create_intermediate_plots, plot_intermediate_epochs);

    // Detect and handle collisions
    find_collisions_parallel(universe);
}

void BarnesHutSimulationWithCollisions::find_collisions(Universe& universe) {
    const double collision_distance_squared = 100000000000.0 * 100000000000.0; // Squared collision distance in m
    std::vector<std::size_t> bodies_to_remove;

    // Iterate through all pairs of bodies
    for (std::size_t i = 0; i < universe.num_bodies; ++i) {
        for (std::size_t j = i + 1; j < universe.num_bodies; ++j) {
            // Compute the squared distance between the two bodies
            Vector2d<double> delta = universe.positions[i] - universe.positions[j];
            double distance_squared = delta[0] * delta[0] + delta[1] * delta[1];

            // Check if the squared distance is below the collision threshold
            if (distance_squared < collision_distance_squared) {
                // Determine the heavier and lighter body
                std::size_t heavier = (universe.weights[i] >= universe.weights[j]) ? i : j;
                std::size_t lighter = (heavier == i) ? j : i;

                // Update the heavier body's mass and velocity
                double m1 = universe.weights[lighter];
                double m2 = universe.weights[heavier];

                universe.weights[heavier] = m1 + m2;
                universe.weights[lighter] = 0;

                m1 = 0;
                m2 = m1 + m2;
                universe.velocities[heavier] = (universe.velocities[lighter] * m1 + universe.velocities[heavier] * m2) / (m1 + m2);

                // Mark the lighter body for removal
                bodies_to_remove.push_back(lighter);
            }
        }
    }

    // Sort and remove duplicates from the removal list
    std::sort(bodies_to_remove.begin(), bodies_to_remove.end());
    bodies_to_remove.erase(std::unique(bodies_to_remove.begin(), bodies_to_remove.end()), bodies_to_remove.end());

    // Remove bodies in reverse order to prevent invalidating indices
    for (auto it = bodies_to_remove.rbegin(); it != bodies_to_remove.rend(); ++it) {
        std::size_t index = *it;

        // Remove from positions, velocities, forces, and weights
        universe.positions.erase(universe.positions.begin() + index);
        universe.velocities.erase(universe.velocities.begin() + index);
        universe.forces.erase(universe.forces.begin() + index); // Remove the corresponding force
        universe.weights.erase(universe.weights.begin() + index);

        // Decrement the number of bodies
        universe.num_bodies--;
    }
}

void BarnesHutSimulationWithCollisions::find_collisions_parallel(Universe& universe) {
    const double collision_distance_squared = 100000000000.0 * 100000000000.0; // Squared collision distance in m
    std::vector<std::size_t> bodies_to_remove;

    // Create a thread-safe structure to record collisions
    std::vector<std::pair<std::size_t, std::size_t>> collisions;

    // Parallelize the outer loop using OpenMP
    #pragma omp parallel
    {
        // Thread-local storage for collisions
        std::vector<std::pair<std::size_t, std::size_t>> local_collisions;

        #pragma omp for schedule(dynamic)
        for (std::size_t i = 0; i < universe.num_bodies; ++i) {
            for (std::size_t j = i + 1; j < universe.num_bodies; ++j) {
                // Compute the squared distance between the two bodies
                Vector2d<double> delta = universe.positions[i] - universe.positions[j];
                double distance_squared = delta[0] * delta[0] + delta[1] * delta[1];

                // If the distance between bodies is smaller than the collision threshold
                if (distance_squared < collision_distance_squared) {
                    // Record the collision
                    local_collisions.emplace_back(i, j);
                }
            }
        }

        // Merge local results into global results
        #pragma omp critical
        collisions.insert(collisions.end(), local_collisions.begin(), local_collisions.end());
    }

    // Resolve collisions sequentially
    for (const auto& collision : collisions) {
        std::size_t i = collision.first;
        std::size_t j = collision.second;

        // Determine the heavier and lighter body
        std::size_t heavier = (universe.weights[i] >= universe.weights[j]) ? i : j;
        std::size_t lighter = (heavier == i) ? j : i;

        // Update the heavier body's mass and velocity
        double m1 = universe.weights[lighter];
        double m2 = universe.weights[heavier];
        double total_mass = m1 + m2;

        universe.weights[heavier] = total_mass;

        // Compute the new velocity based on the conservation of momentum
        Vector2d<double> new_velocity = (universe.velocities[lighter] * m1 +
            universe.velocities[heavier] * m2) / total_mass;
        universe.velocities[heavier] = new_velocity;

        // Mark the lighter body for removal
        bodies_to_remove.push_back(lighter);
    }

    // Sort the bodies to remove in descending order to avoid index invalidation during removal
    std::sort(bodies_to_remove.rbegin(), bodies_to_remove.rend());

    // Remove bodies and update the number of bodies in the universe
    for (auto& body_idx : bodies_to_remove) {
        // Erase the body from all vectors
        universe.positions.erase(universe.positions.begin() + body_idx);
        universe.velocities.erase(universe.velocities.begin() + body_idx);
        universe.forces.erase(universe.forces.begin() + body_idx);
        universe.weights.erase(universe.weights.begin() + body_idx);

        // Decrement the total number of bodies
        universe.num_bodies--;
    }
}

