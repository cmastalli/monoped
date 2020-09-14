import example_robot_data
import pinocchio
import numpy as np

def loadSoloLeg(solo8 = True):
    if solo8:
        URDF_FILENAME = "solo.urdf"
        legMaxId = 4
    else:
        URDF_FILENAME = "solo12.urdf"
        legMaxId = 5
    SRDF_FILENAME = "solo.srdf"
    SRDF_SUBPATH = "/solo_description/srdf/" + SRDF_FILENAME
    URDF_SUBPATH = "/solo_description/robots/" + URDF_FILENAME
    modelPath = example_robot_data.getModelPath(URDF_SUBPATH)

    robot = example_robot_data.loadSolo(solo8)
    m1 = robot.model
    m2 = pinocchio.Model()
    for index, [j, M, name, parent, Y] in enumerate(zip(m1.joints, m1.jointPlacements, m1.names, m1.parents, m1.inertias)):
        if j.id < legMaxId:
            jointType = j.shortname()
            if jointType == 'JointModelFreeFlyer':
                # for the freeflyer we just take reduced mass and zero inertia
                jointType = 'JointModelPZ'
                Y.mass = Y.mass/4
                Y.inertia = np.diag(1e-6*np.ones(3))
                M = pinocchio.SE3.Identity()
            
            if index == 2:
                # start with the prismatic joint on the sliding guide
                M.translation = np.zeros(3)
            
            # 2D model, flatten y axis
            vector2d = M.translation
            vector2d[1] = 0.0
            M.translation = vector2d

            jid = m2.addJoint(parent, getattr(pinocchio, jointType)(), M, name)
            assert (jid == j.id)
            m2.appendBodyToJoint(jid, Y, pinocchio.SE3.Identity())

    for f in m1.frames:
        if f.parent < legMaxId:
            m2.addFrame(f)

    g2 = pinocchio.GeometryModel()
    for g in robot.visual_model.geometryObjects:
        if g.parentJoint < legMaxId:
            g2.addGeometryObject(g)

    robot.model = m2
    robot.data = m2.createData()
    robot.visual_model = g2
    robot.visual_data = pinocchio.GeometryData(g2)

    # Load SRDF file
    #q0 = example_robot_data.readParamsFromSrdf(robot.model, modelPath + SRDF_SUBPATH, False, False, 'standing')
    robot.q0 = np.zeros(m2.nq+1)

    assert ((m2.rotorInertia[:m2.joints[1].nq] == 0.).all())
    return robot