#include "quadtreeNode.h"
#include "structures/universe.h"

double QuadtreeNode::calculate_node_cumulative_mass() {
    // Wenn die kumulierte Masse bereits berechnet wurde, gib sie zurück
    if (cumulative_mass_ready) {
        return cumulative_mass;
    }

    // Falls es sich um einen Blattknoten handelt, gib einfach die gespeicherte Masse zurück
    if (children.empty()) {
        // cummulative_mass sollte bereits im Knoten gesetzt worden sein, daher gibt es hier nichts mehr zu tun
        cumulative_mass_ready = true; // Sicherstellen, dass der Wert als berechnet markiert wird
        return cumulative_mass;
    }

    // Wenn es sich um einen inneren Knoten handelt, berechne die kumulierte Masse der Kinder
    cumulative_mass = 0.0;
    for (auto* child : children) {
        // Rekursive Berechnung der kumulierten Masse für jedes Kind
        cumulative_mass += child->calculate_node_cumulative_mass();
    }

    // Nachdem die kumulierte Masse berechnet wurde, markiere den Knoten als "fertig"
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

Vector2d<double> QuadtreeNode::calculate_node_center_of_mass() {
    if (center_of_mass_ready) {
        return center_of_mass;  // Wenn bereits berechnet, gib den gespeicherten Wert zurück
    }

    // Wenn der Knoten ein Blattknoten ist
    if (children.empty()) {
        // Die Position des Körpers ist der Massenschwerpunkt des Blattknotens
        center_of_mass_ready = true;
        return center_of_mass;
    }

    // Andernfalls müssen wir den Massenschwerpunkt der Kinder berechnen
    double total_mass = 0.0;
    Vector2d<double> weighted_position(0.0, 0.0);

    for (auto* child : children) {
        // Rekursiv den Massenschwerpunkt jedes Kindes berechnen
        Vector2d<double> child_center_of_mass = child->calculate_node_center_of_mass();
        double child_mass = child->calculate_node_cumulative_mass();

        weighted_position = weighted_position + child_center_of_mass * child_mass;
        total_mass += child_mass;
    }

    // Berechne den Massenschwerpunkt des aktuellen Knotens
    if (total_mass > 0) {
        center_of_mass = weighted_position / total_mass;
    }
    else {
        center_of_mass = Vector2d<double>(0.0, 0.0); // Falls keine Masse, setze den Standardwert
    }

    center_of_mass_ready = true;
    return center_of_mass;
}
