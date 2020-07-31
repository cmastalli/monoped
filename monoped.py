from pinocchio.utils import *
import pinocchio

def createMonoped(nbJoint, linkLength=1.0, floatingMass=1.0, linkMass=1.0):
    rmodel = pinocchio.Model()
    jointId = 0
    jointPlacement = pinocchio.SE3.Identity()
    baseInertia = pinocchio.Inertia(floatingMass,
                          np.matrix([0.0, 0.0, 0.0]).T,
                          np.diagflat([0.0, 0.0, 0.0]))
    linkInertia = pinocchio.Inertia(linkMass,
                          np.matrix([0.0, 0.0, linkLength/2]).T,
                          linkMass/5*np.diagflat([1e-2, linkLength**2, 1e-2]))
    istr = str(jointId)
    name               = "prismatic_"+istr
    jointName,bodyName = [name+"_joint",name+"_body"]
    jointId = rmodel.addJoint(jointId, pinocchio.JointModelPZ(), jointPlacement, jointName)
    rmodel.addFrame(pinocchio.Frame('base', jointId, 0, jointPlacement, pinocchio.FrameType.OP_FRAME))
    rmodel.appendBodyToJoint(jointId, baseInertia, pinocchio.SE3.Identity())
    jointPlacement     = pinocchio.SE3(eye(3), np.matrix([0.0, 0.0, 0.0]).T)

    name               = "shoulder_"+istr
    jointName,bodyName = [name+"_joint",name+"_body"]
    jointId = rmodel.addJoint(jointId, pinocchio.JointModelRX(), jointPlacement, jointName)
    rmodel.appendBodyToJoint(jointId, baseInertia, pinocchio.SE3.Identity())
    jointPlacement     = pinocchio.SE3(eye(3), np.matrix([0.0, 0.0, 0.0]).T)

    for i in range(0, nbJoint):
        istr = str(i + 1)
        name               = "revolute_" + istr
        jointName,bodyName = [name + "_joint", name+"_body"]
        jointId = rmodel.addJoint(jointId,pinocchio.JointModelRY(),jointPlacement,jointName)
        rmodel.addFrame(pinocchio.Frame(jointName, jointId, 0, jointPlacement, pinocchio.FrameType.OP_FRAME))
        rmodel.appendBodyToJoint(jointId,linkInertia,pinocchio.SE3.Identity())
        jointPlacement     = pinocchio.SE3(eye(3), np.matrix([0.0, 0.0, linkLength]).T)

    rmodel.addFrame( pinocchio.Frame('foot', jointId, 0, jointPlacement, pinocchio.FrameType.OP_FRAME))
    rmodel.upperPositionLimit = np.concatenate((np.array([100]),  2 * np.pi * np.ones(nbJoint)), axis=0)
    rmodel.lowerPositionLimit = np.concatenate((np.array([0.0]), -2 * np.pi * np.ones(nbJoint)), axis=0)
    rmodel.velocityLimit      = np.concatenate((np.array([100]),  5 * np.ones(nbJoint)), axis=0)

    return rmodel

def createMonopedWrapper(nbJoint):
    '''
    Returns a RobotWrapper with a monoped inside.
    '''
    rmodel = createMonoped(nbJoint)
    rw = pinocchio.RobotWrapper(rmodel,visual_model=None,collision_model=None)
    return rw
