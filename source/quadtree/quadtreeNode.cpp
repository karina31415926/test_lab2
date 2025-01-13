#include "quadtreeNode.h"
#include "structures/universe.h"

double QuadtreeNode::calculate_node_cumulative_mass(Universe& universe) {
    // If cumulative mass has already been calculated, return it
    if (cumulative_mass_ready) {
        return cumulative_mass;
    }

    // Reset cumulative mass to ensure it is calculated fresh
    cumulative_mass = 0.0;

    // If the node is a leaf, sum up the masses of bodies inside the bounding box
    if (children.empty()) {
        for (int i = 0; i < universe.num_bodies; ++i) {
            const Vector2d<double>& pos = universe.positions[i];
            double mass = universe.weights[i];

            // Check if the body is within the bounding box
            if (pos[0] >= bounding_box.x_min && pos[0] <= bounding_box.x_max &&
                pos[1] >= bounding_box.y_min && pos[1] <= bounding_box.y_max) {
                cumulative_mass += mass;
            }
        }
    }
    else {
        // If the node has children, recursively calculate their cumulative masses
        for (QuadtreeNode* child : children) {
            cumulative_mass += child->calculate_node_cumulative_mass(universe);
        }
    }

    // Mark as calculated
    cumulative_mass_ready = true;
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

Vector2d<double> QuadtreeNode::calculate_node_center_of_mass(Universe& universe) {
    // If the center of mass has already been calculated, return it
    if (center_of_mass_ready) {
        return center_of_mass;
    }

    if (children.empty()) {
        // Leaf node: Compute the center of mass directly from bodies in the bounding box
        Vector2d<double> total_position(0.0, 0.0);

        for (int i = 0; i < universe.num_bodies; ++i) {
            const Vector2d<double>& pos = universe.positions[i];
            double mass = universe.weights[i];

            if (pos[0] >= bounding_box.x_min && pos[0] <= bounding_box.x_max &&
                pos[1] >= bounding_box.y_min && pos[1] <= bounding_box.y_max) {
                total_position.set(total_position[0] + pos[0] * mass,
                    total_position[1] + pos[1] * mass);
            }
        }

        if (cumulative_mass > 0) {
            center_of_mass.set(total_position[0] / cumulative_mass,
                total_position[1] / cumulative_mass);
        }
    }
    else {
        // Internal node: Calculate center of mass recursively from children
        Vector2d<double> weighted_center(0.0, 0.0);
        double total_mass = 0.0;

        for (QuadtreeNode* child : children) {
            Vector2d<double> child_center_of_mass = child->calculate_node_center_of_mass(universe);

            // Accumulate weighted center of mass
            weighted_center.set(weighted_center[0] + child_center_of_mass[0] * child->cumulative_mass,
                weighted_center[1] + child_center_of_mass[1] * child->cumulative_mass);
            total_mass += child->cumulative_mass;

            // Mark child node's center of mass as ready
            child->center_of_mass_ready = true;
        }

        if (total_mass > 0) {
            center_of_mass.set(weighted_center[0] / total_mass,
                weighted_center[1] / total_mass);
        }
    }

    // Mark this node's center of mass as ready
    center_of_mass_ready = true;

    return center_of_mass;
}
