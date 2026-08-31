"""
Isaac Gym / Isaac Sim-Style Multi-Robot 3D Stage Viewer for MuJoCo.
-------------------------------------------------------------------
Spawns a 4x4 Grid (16 Piper Robot Arms) or 5x5 Grid (25 Piper Robot Arms)
operating simultaneously on a single 3D tabletop stage in a single GLFW window.

Usage:
    # Launch 16 Parallel Piper Arms on a 4x4 Grid Stage:
    python -m src.view_multi_robot_grid --grid 4
"""

import os
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer

from src.environment.env import MJCF_PATH, HOME_QPOS, HOME_CTRL


def build_grid_xml(grid_size: int = 4) -> str:
    """Generate a multi-robot grid MJCF XML string containing grid_size x grid_size Piper arms."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(MJCF_PATH))
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str(MJCF_PATH.parent / "piper" / "meshes"))

    worldbody = root.find("worldbody")
    actuator = root.find("actuator")
    equality = root.find("equality")
    contact = root.find("contact")

    # Find base robot body in original XML
    robot_template = None
    table_template = None
    cube_template = None

    for child in list(worldbody):
        if child.tag == "body" and child.get("name") == "base_link":
            robot_template = child
            worldbody.remove(child)
        elif child.tag == "body" and child.get("name") == "table":
            table_template = child
            worldbody.remove(child)
        elif child.tag == "body" and "cube" in child.get("name", ""):
            if child.get("name") == "cube_red":
                cube_template = child
            worldbody.remove(child)
        elif child.tag == "body" and child.get("name") in ["bin_blue", "bin_red"]:
            worldbody.remove(child)

    # Clear original actuators, equality, contact
    actuator_templates = list(actuator) if actuator is not None else []
    if actuator is not None:
        actuator.clear()

    equality_templates = list(equality) if equality is not None else []
    if equality is not None:
        equality.clear()

    contact_templates = list(contact) if contact is not None else []
    if contact is not None:
        contact.clear()

    # Recursive renaming helper
    def copy_and_rename(elem, pfx):
        new_elem = ET.Element(elem.tag, attrib=elem.attrib.copy())
        if "name" in new_elem.attrib:
            new_elem.attrib["name"] = f"{pfx}{new_elem.attrib['name']}"
        if "joint" in new_elem.attrib:
            new_elem.attrib["joint"] = f"{pfx}{new_elem.attrib['joint']}"
        if "joint1" in new_elem.attrib:
            new_elem.attrib["joint1"] = f"{pfx}{new_elem.attrib['joint1']}"
        if "joint2" in new_elem.attrib:
            new_elem.attrib["joint2"] = f"{pfx}{new_elem.attrib['joint2']}"
        if "site" in new_elem.attrib:
            new_elem.attrib["site"] = f"{pfx}{new_elem.attrib['site']}"
        if "body1" in new_elem.attrib:
            new_elem.attrib["body1"] = f"{pfx}{new_elem.attrib['body1']}"
        if "body2" in new_elem.attrib:
            new_elem.attrib["body2"] = f"{pfx}{new_elem.attrib['body2']}"
        for sub in elem:
            new_elem.append(copy_and_rename(sub, pfx))
        return new_elem

    spacing_x = 1.2
    spacing_y = 1.2

    for r in range(grid_size):
        for c in range(grid_size):
            prefix = f"r{r}_c{c}_"
            offset_x = (r - (grid_size - 1) / 2.0) * spacing_x
            offset_y = (c - (grid_size - 1) / 2.0) * spacing_y

            # 1. Spawn Table
            if table_template is not None:
                tbl_el = ET.Element("body", attrib={
                    "name": f"{prefix}table",
                    "pos": f"{offset_x + 0.40} {offset_y} 0.0"
                })
                for child in table_template:
                    tbl_el.append(copy_and_rename(child, prefix))
                worldbody.append(tbl_el)

            # 2. Spawn Robot Arm
            if robot_template is not None:
                arm_el = copy_and_rename(robot_template, prefix)
                arm_el.attrib["pos"] = f"{offset_x} {offset_y} 0.16"
                worldbody.append(arm_el)

            # 3. Spawn Cube
            if cube_template is not None:
                cb_el = ET.Element("body", attrib={
                    "name": f"{prefix}cube_red",
                    "pos": f"{offset_x + 0.28} {offset_y + 0.15} 0.18"
                })
                for child in cube_template:
                    cb_el.append(copy_and_rename(child, prefix))
                worldbody.append(cb_el)

            # 4. Spawn Actuators
            for act_elem in actuator_templates:
                new_act = ET.Element(act_elem.tag, attrib=act_elem.attrib.copy())
                if "name" in new_act.attrib:
                    new_act.attrib["name"] = f"{prefix}{new_act.attrib['name']}"
                if "joint" in new_act.attrib:
                    new_act.attrib["joint"] = f"{prefix}{new_act.attrib['joint']}"
                actuator.append(new_act)

            # 5. Spawn Equality Constraints
            if equality is not None:
                for eq_elem in equality_templates:
                    equality.append(copy_and_rename(eq_elem, prefix))

            # 6. Spawn Contact Exclusions
            if contact is not None:
                for ct_elem in contact_templates:
                    contact.append(copy_and_rename(ct_elem, prefix))

    return ET.tostring(root, encoding="unicode")


def launch_grid_stage(grid_size: int = 4):
    print(f"\n========================================================================")
    print(f"   Isaac Sim-Style Multi-Robot Stage ({grid_size}x{grid_size} = {grid_size**2} Piper Arms)  ")
    print(f"========================================================================")
    print(f"  Spawning {grid_size**2} Piper robot arms operating on a single 3D stage...")
    
    xml_str = build_grid_xml(grid_size)
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    print(f"✓ Created stage model with {model.nu} actuators and {model.nq} DOFs!")
    print("Opening interactive GLFW 3D viewer window...\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            t = time.time() - start_time
            # Apply dynamic sinusoidal motor babbling to all parallel robots
            ctrl_targets = np.zeros(model.nu)
            num_robots = grid_size * grid_size

            for i in range(num_robots):
                base_idx = i * 7
                # Unique phase shift for each robot
                phase = i * 0.5 + t * 2.0
                ctrl_targets[base_idx + 0] = 0.3 * np.sin(phase)          # j1
                ctrl_targets[base_idx + 1] = -3.10 + 0.2 * np.cos(phase)  # j2
                ctrl_targets[base_idx + 2] = -0.25 + 0.2 * np.sin(phase)  # j3
                ctrl_targets[base_idx + 3] = 0.4 * np.cos(phase * 1.5)    # j4
                ctrl_targets[base_idx + 4] = 0.3 * np.sin(phase * 1.2)    # j5
                ctrl_targets[base_idx + 5] = 0.5 * np.sin(phase)          # j6
                ctrl_targets[base_idx + 6] = 0.04                         # gripper

            data.ctrl[:] = ctrl_targets
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)


def main():
    parser = argparse.ArgumentParser(description="Isaac Sim-Style Multi-Robot Stage Viewer for MuJoCo")
    parser.add_argument("--grid", type=int, default=4, help="Grid size N x N (e.g. 4 for 16 arms, 5 for 25 arms)")
    args = parser.parse_args()

    launch_grid_stage(args.grid)


if __name__ == "__main__":
    main()
