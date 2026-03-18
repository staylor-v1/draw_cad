// Hydraulic involute cam actuator
// Rebuilt from US 8,047,094 B2 Figure 1A with a focus on the front-view silhouette
// and a plausible involute/roller contact relationship.

$fn = 96;

// ----------------------------
// Presentation controls
// ----------------------------
animate_motion = false;
show_housing_shell = true;
theta_limit_deg = 18;
theta_deg = animate_motion ? sin($t * 360) * theta_limit_deg : 0;
theta_rad = theta_deg * PI / 180;

// ----------------------------
// Global dimensions
// ----------------------------
member_depth = 18;
roller_depth = member_depth + 6;
cam_vertical_offset = -6;

// Arm / upper member (12)
arm_width = 108;
arm_height = 265;
arm_attach_y = 10;
pivot_block_w = 44;
pivot_block_h = 18;
pivot_radius = 6;

// Cam / follower geometry
base_radius = 28;          // Constant moment arm for the involute pitch curve
roller_radius = 10;
lobe_angle_deg = 126;
lobe_step_deg = 2;
bridge_radius = base_radius - roller_radius + 2;
shoulder_radius = 11;

// Actuators (28, 30)
rod_width = 12;
guide_width = 20;
guide_height = 24;
barrel_width = 34;
barrel_height = 62;
base_pad_width = 42;
base_pad_height = 18;
clevis_top_w = 30;
clevis_bottom_w = 18;
clevis_height = 30;
clevis_neck_drop = 18;
barrel_bottom_y = -214;

// Lower member / housing (14)
housing_width = 118;
housing_height = 152;
housing_bottom_y = -248;
housing_wall = 4;

// ----------------------------
// Utility functions
// ----------------------------
function to_rad(deg) = deg * PI / 180;

function involute_point(rb, deg) =
    let(t = to_rad(deg))
    [
        rb * (cos(deg) + t * sin(deg)),
        -rb * (sin(deg) - t * cos(deg))
    ];

function rotate_point(pt, deg) =
    [
        pt[0] * cos(deg) - pt[1] * sin(deg),
        pt[0] * sin(deg) + pt[1] * cos(deg)
    ];

function pitch_curve_point(rb, deg) = rotate_point(involute_point(rb, deg), -90);

neutral_pitch_y = cam_vertical_offset - base_radius * PI / 2;

// ----------------------------
// 2D profiles
// ----------------------------
module upper_arm_profile() {
    translate([-arm_width / 2, arm_attach_y])
        square([arm_width, arm_height]);
}

module pivot_block_profile() {
    translate([-pivot_block_w / 2, -pivot_block_h / 2])
        square([pivot_block_w, pivot_block_h]);
}

module lobe_pitch_profile(side = 1) {
    pts =
        [
            [0, 0],
            for (a = [0 : lobe_step_deg : lobe_angle_deg])
                let(p = pitch_curve_point(base_radius, a))
                [side * p[0], p[1]]
        ];

    polygon(pts);
}

module dual_cam_profile() {
    union() {
        circle(r = bridge_radius);

        offset(r = -roller_radius)
            union() {
                lobe_pitch_profile(1);
                lobe_pitch_profile(-1);
            }

        hull() {
            translate([-base_radius * 0.80, 7]) circle(r = shoulder_radius);
            translate([ base_radius * 0.80, 7]) circle(r = shoulder_radius);
            translate([0, 8]) square([pivot_block_w - 4, 4], center = true);
        }
    }
}

module clevis_profile(roller_y) {
    difference() {
        hull() {
            translate([0, roller_y - clevis_height + 6])
                square([clevis_bottom_w, 12], center = true);
            translate([0, roller_y - 6])
                square([clevis_top_w, 14], center = true);
        }

        translate([0, roller_y])
            circle(r = 4.5);
    }
}

module rod_profile(roller_y) {
    rod_bottom_y = barrel_bottom_y + barrel_height + guide_height;
    rod_top_y = roller_y - clevis_neck_drop;

    union() {
        translate([-rod_width / 2, rod_bottom_y])
            square([rod_width, max(0.1, rod_top_y - rod_bottom_y)]);

        hull() {
            translate([0, rod_top_y - 2])
                square([rod_width, 4], center = true);
            translate([0, roller_y - clevis_height + 8])
                square([clevis_bottom_w, 4], center = true);
        }
    }
}

module actuator_body_profile() {
    union() {
        translate([-base_pad_width / 2, barrel_bottom_y - base_pad_height])
            square([base_pad_width, base_pad_height]);

        translate([-barrel_width / 2, barrel_bottom_y])
            square([barrel_width, barrel_height]);

        translate([-guide_width / 2, barrel_bottom_y + barrel_height])
            square([guide_width, guide_height]);
    }
}

module housing_profile() {
    difference() {
        translate([-housing_width / 2, housing_bottom_y])
            square([housing_width, housing_height]);

        translate([-housing_width / 2 + housing_wall, housing_bottom_y + housing_wall])
            square([housing_width - 2 * housing_wall, housing_height - 2 * housing_wall]);
    }
}

// ----------------------------
// 3D assemblies
// ----------------------------
module top_member() {
    color("gainsboro")
        linear_extrude(height = member_depth, center = true)
            union() {
                upper_arm_profile();
                pivot_block_profile();
                dual_cam_profile();
            }

    color("dimgray")
        cylinder(r = pivot_radius, h = member_depth + 2, center = true);
}

module actuator(side = 1) {
    roller_center_y = neutral_pitch_y + side * base_radius * theta_rad;

    translate([side * base_radius, 0, 0]) {
        color("steelblue")
            linear_extrude(height = member_depth, center = true)
                actuator_body_profile();

        color("lightsteelblue")
            linear_extrude(height = member_depth * 0.78, center = true)
                rod_profile(roller_center_y);

        color("silver")
            linear_extrude(height = member_depth, center = true)
                clevis_profile(roller_center_y);

        color("dimgray")
            translate([0, roller_center_y, 0])
                cylinder(r = roller_radius, h = roller_depth, center = true);

        color("silver")
            translate([0, roller_center_y, 0])
                cylinder(r = 3.4, h = roller_depth + 2, center = true);
    }
}

module lower_housing() {
    if (show_housing_shell)
        color([0.65, 0.65, 0.65, 0.28])
            linear_extrude(height = member_depth + 8, center = true)
                housing_profile();
}

module assembly() {
    lower_housing();

    actuator(-1);
    actuator(1);

    translate([0, cam_vertical_offset, 0])
        rotate([0, 0, theta_deg])
            top_member();
}

assembly();
