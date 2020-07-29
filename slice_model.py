def loadTalosLegs():
    robot = loadTalos()
    URDF_FILENAME = "talos_reduced.urdf"
    SRDF_FILENAME = "talos.srdf"
    SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
    URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME
    modelPath = getModelPath(URDF_SUBPATH)

    legMaxId = 14
    m1 = robot.model
    m2 = pinocchio.Model()
    for j, M, name, parent, Y in zip(m1.joints, m1.jointPlacements, m1.names, m1.parents, m1.inertias):
        if j.id < legMaxId:
            jid = m2.addJoint(parent, getattr(pinocchio, j.shortname())(), M, name)
            upperPos = m2.upperPositionLimit
            lowerPos = m2.lowerPositionLimit
            effort = m2.effortLimit
            upperPos[m2.joints[jid].idx_q:m2.joints[jid].idx_q + j.nq] = m1.upperPositionLimit[j.idx_q:j.idx_q + j.nq]
            lowerPos[m2.joints[jid].idx_q:m2.joints[jid].idx_q + j.nq] = m1.lowerPositionLimit[j.idx_q:j.idx_q + j.nq]
            effort[m2.joints[jid].idx_v:m2.joints[jid].idx_v + j.nv] = m1.effortLimit[j.idx_v:j.idx_v + j.nv]
            m2.upperPositionLimit = upperPos
            m2.lowerPositionLimit = lowerPos
            m2.effortLimit = effort
            assert (jid == j.id)
            m2.appendBodyToJoint(jid, Y, pinocchio.SE3.Identity())

    upperPos = m2.upperPositionLimit
    upperPos[:7] = 1
    m2.upperPositionLimit = upperPos
    lowerPos = m2.lowerPositionLimit
    lowerPos[:7] = -1
    m2.lowerPositionLimit = lowerPos
    effort = m2.effortLimit
    effort[:6] = np.inf
    m2.effortLimit = effort

    # q2 = robot.q0[:19]
    for f in m1.frames:
        if f.parent < legMaxId:
            m2.addFrame(f)

    g2 = pinocchio.GeometryModel()
    for g in robot.visual_model.geometryObjects:
        if g.parentJoint < 14:
            g2.addGeometryObject(g)

    robot.model = m2
    robot.data = m2.createData()
    robot.visual_model = g2
    # robot.q0=q2
    robot.visual_data = pinocchio.GeometryData(g2)

    # Load SRDF file
    robot.q0 = robot.q0[:robot.model.nq]
    robot.q0 = readParamsFromSrdf(robot.model, modelPath + SRDF_SUBPATH, False)

    assert ((m2.armature[:6] == 0.).all())
    # Add the free-flyer joint limits
    addFreeFlyerJointLimits(robot.model)
    return robot