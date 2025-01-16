#include "simulation/barnes_hut_simulation.h"
#include "simulation/naive_parallel_simulation.h"
#include "physics/gravitation.h"
#include "physics/mechanics.h"

#include <cmath>

void BarnesHutSimulation::simulate_epochs(Plotter& plotter, Universe& universe, std::uint32_t num_epochs, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs){
    for(int i = 0; i < num_epochs; i++){
        simulate_epoch(plotter, universe, create_intermediate_plots, plot_intermediate_epochs);
    }
}

void BarnesHutSimulation::simulate_epoch(Plotter& plotter, Universe& universe, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs) {
    Quadtree quadtree(universe, universe.get_bounding_box(), 2); // Mode 2: Parallel construction with cut-off

    quadtree.calculate_cumulative_masses();
    quadtree.calculate_center_of_mass();

    calculate_forces(universe, quadtree);

    NaiveParallelSimulation::calculate_velocities(universe);
    NaiveParallelSimulation::calculate_positions(universe);

    universe.current_simulation_epoch++;

    if (create_intermediate_plots && (universe.current_simulation_epoch % plot_intermediate_epochs == 0)) {
        for (std::uint32_t i = 0; i < universe.num_bodies; i++) {
            const Vector2d<double>& position = universe.positions[i];
            plotter.mark_position(position, 255, 0, 0); // Red color for the position
        }
        plotter.write_and_clear();
    }
}

void BarnesHutSimulation::get_relevant_nodes(Universe& universe, Quadtree& quadtree, std::vector<QuadtreeNode*>& relevant_nodes, Vector2d<double>& body_position, std::int32_t body_index, double threshold_theta) {
    // Stack for depth-first search starting from the root of the quadtree
    std::vector<QuadtreeNode*> stack;
    stack.push_back(quadtree.root);

    while (!stack.empty()) {
        // Take the next node from the stack
        QuadtreeNode* node = stack.back();
        stack.pop_back();

        // Calculate the diagonal of the quadrant and the distance from body K to the center of mass
        double d = node->bounding_box.get_diagonal();
        Vector2d<double> delta = body_position - node->center_of_mass;
        double r_squared = delta[0] * delta[0] + delta[1] * delta[1];
        double r = std::sqrt(r_squared);

        // Prevent division by zero
        if (r == 0.0) {
            r = std::numeric_limits<double>::epsilon();
        }

        // Compute Theta (d / r)
        double theta = d / r;

        // Debug output to check values of theta and the threshold

        // Case 1: The node is a leaf (no children)
        if (node->children.empty()) {
            // If the node contains body K, skip it
            if (node->body_identifier == body_index) {
                continue;
            }

            // If the node contains at least one body, it's relevant for force calculation
            if (node->body_identifier != -1) {
                relevant_nodes.push_back(node);
            }
        }
        // Case 2: Theta < threshold_theta, the node is relevant for force calculation
        else if (theta < threshold_theta) {
            relevant_nodes.push_back(node);
        }
        else {
            // Case 3: Theta >= threshold_theta, the node must be further subdivided
            // Add all children of the current node to the stack
            for (QuadtreeNode* child : node->children) {
                // Only add children that are not the same as body K
                if (child != nullptr && child->body_identifier != body_index) {
                    stack.push_back(child);
                }
            }
        }
    }
}



void BarnesHutSimulation::calculate_forces(Universe& universe, Quadtree& quadtree) {
    const double threshold_theta = 0.2;

#pragma omp parallel for
    for (std::uint32_t i = 0; i < universe.num_bodies; ++i) {
        Vector2d<double> total_force(0.0, 0.0);

        std::vector<QuadtreeNode*> relevant_nodes;
        get_relevant_nodes(universe, quadtree, relevant_nodes, universe.positions[i], i, threshold_theta);

        for (auto* node : relevant_nodes) {
            if (node->body_identifier == -1) {
                Vector2d<double> delta = universe.positions[i] - node->center_of_mass;
                double r_squared = delta[0] * delta[0] + delta[1] * delta[1];

                // Avoid division by zero
                if (r_squared > 0) {
                    double force_magnitude = gravitational_force(universe.weights[i], node->cumulative_mass, std::sqrt(r_squared));
                    Vector2d<double> force_direction = delta / std::sqrt(r_squared);
                    total_force = total_force + force_direction * force_magnitude;
                }
            }
            else if (i != node->body_identifier) {
                Vector2d<double> delta = universe.positions[i] - universe.positions[node->body_identifier];
                double r_squared = delta[0] * delta[0] + delta[1] * delta[1];

                // Avoid division by zero
                if (r_squared > 0) {
                    double force_magnitude = gravitational_force(universe.weights[i], universe.weights[node->body_identifier], std::sqrt(r_squared));
                    Vector2d<double> force_direction = delta / std::sqrt(r_squared);
                    total_force = total_force + force_direction * force_magnitude;
                }
            }
        }

        universe.forces[i] = total_force;
    }
}