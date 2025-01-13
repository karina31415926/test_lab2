#include "quadtreeNode.h"
#include "structures/universe.h"
#include <iostream>


double QuadtreeNode::calculate_node_cumulative_mass() {
    // If cumulative mass has already been calculated, return it.
    if (cumulative_mass_ready) {
        return cumulative_mass;
    }

    cumulative_mass = 0.0;

    // Get the global Universe instance
    Universe& universe = Universe::get_instance();  

    // Debug: Print bounding box for this node
    std::cout << "Checking node's bounding box: x_min = " << bounding_box.x_min 
              << ", x_max = " << bounding_box.x_max 
              << ", y_min = " << bounding_box.y_min 
              << ", y_max = " << bounding_box.y_max << std::endl;

    // Iterate through the bodies in the universe
    for (int i = 0; i < universe.num_bodies; ++i) {
        const Vector2d<double>& pos = universe.positions[i];

        // Debug: Print body position
        std::cout << "Checking body " << i << ": Position (" << pos[0] << ", " << pos[1] << "), Mass: " << universe.weights[i] << std::endl;

        // Check if the body is inside the current QuadtreeNode's bounding box
        if (pos[0] >= bounding_box.x_min && pos[0] <= bounding_box.x_max &&
            pos[1] >= bounding_box.y_min && pos[1] <= bounding_box.y_max) {
            // If the body is within the bounding box, add its mass to the cumulative mass
            cumulative_mass += universe.weights[i];  // Assuming `weights` holds the mass of each body
            // Debug: Print the cumulative mass after adding a body
            std::cout << "   Inside node, Adding mass: " << universe.weights[i] 
                      << ", Updated cumulative mass: " << cumulative_mass << std::endl;
        }
    }

    // Mark the cumulative mass as calculated
    cumulative_mass_ready = true;

    // Debug: Print final cumulative mass for this node
    std::cout << "Final cumulative mass for this node: " << cumulative_mass << std::endl;

    return cumulative_mass;
}



QuadtreeNode::QuadtreeNode(BoundingBox arg_bounding_box)
    : bounding_box(arg_bounding_box), body_identifier(-1), cumulative_mass(0.0),
    cumulative_mass_ready(false), center_of_mass_ready(false) {}

QuadtreeNode::~QuadtreeNode() {
    for (QuadtreeNode* child : children) {
        delete child;
    }
    children.clear();
}

Vector2d<double> QuadtreeNode::calculate_node_center_of_mass() {
    // If the center of mass has already been calculated, return it
    if (center_of_mass_ready) {
        return center_of_mass;  // Return the calculated center of mass
    }

    // If there is only one body in the node, return its position as the center of mass
    if (cumulative_mass_ready && cumulative_mass > 0.0) {
        // Create a temporary Vector2d to accumulate the weighted sum of positions
        Vector2d<double> node_center_of_mass(0.0, 0.0);
        Universe& universe = Universe::get_instance();

        // Iterate through the bodies in the universe
        for (int i = 0; i < universe.num_bodies; ++i) {
            const Vector2d<double>& pos = universe.positions[i];

            // Debug: Print position and mass of each body being considered
            std::cout << "Checking body " << i << ": Position (" << pos[0] << ", " << pos[1] << "), Mass: " << universe.weights[i] << std::endl;

            // Check if the body is inside the current QuadtreeNode's bounding box
            if (pos[0] >= bounding_box.x_min && pos[0] <= bounding_box.x_max &&
                pos[1] >= bounding_box.y_min && pos[1] <= bounding_box.y_max) {
                // Accumulate the weighted position
                double mass = universe.weights[i];
                node_center_of_mass.set(node_center_of_mass[0] + pos[0] * mass,
                    node_center_of_mass[1] + pos[1] * mass);
                std::cout << "   Inside node, Accumulated center of mass: ("
                    << node_center_of_mass[0] << ", " << node_center_of_mass[1] << ")\n";
            }
        }

        // Normalize by the total mass to get the center of mass
        if (cumulative_mass > 0) {
            node_center_of_mass.set(node_center_of_mass[0] / cumulative_mass,
                node_center_of_mass[1] / cumulative_mass);
        }

        // Mark as calculated
        center_of_mass_ready = true;
        std::cout << "Final center of mass for this node: (" << node_center_of_mass[0] << ", " << node_center_of_mass[1] << ")\n";
        return node_center_of_mass;
    }

    // If there are sub-quadrants, recursively calculate their center of mass
    double total_mass = 0.0;
    Vector2d<double> weighted_center_of_mass(0.0, 0.0);

    // Recursively calculate the center of mass for each sub-quadrant
    for (QuadtreeNode* child : children) {
        Vector2d<double> child_center_of_mass = child->calculate_node_center_of_mass();

        // Debug: Print the center of mass of the child node
        std::cout << "Child node center of mass: (" << child_center_of_mass[0] << ", " << child_center_of_mass[1] << "), Mass: " << child->cumulative_mass << std::endl;

        // Add the weighted center of mass of the child to the total center of mass
        total_mass += child->cumulative_mass;
        weighted_center_of_mass.set(weighted_center_of_mass[0] + child_center_of_mass[0] * child->cumulative_mass,
            weighted_center_of_mass[1] + child_center_of_mass[1] * child->cumulative_mass);
    }

    // Set the current node's center of mass by normalizing with the total mass
    if (total_mass > 0) {
        weighted_center_of_mass.set(weighted_center_of_mass[0] / total_mass,
            weighted_center_of_mass[1] / total_mass);
    }

    // Mark as calculated
    center_of_mass_ready = true;
    std::cout << "Final center of mass for this parent node: (" << weighted_center_of_mass[0] << ", " << weighted_center_of_mass[1] << ")\n";
    return weighted_center_of_mass;
}
