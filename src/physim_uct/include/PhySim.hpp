#ifndef PHYSIM
#define PHYSIM

#include <common_io.h>
#include <btBulletDynamicsCommon.h>
#include <Bullet3Common/b3FileUtils.h>
#include <Bullet3Common/b3AlignedObjectArray.h>
#include <btBulletDynamicsCommon.h>
#include <LinearMath/btVector3.h>
#include <LinearMath/btAlignedObjectArray.h> 
#include "../examples/Importers/ImportObjDemo/LoadMeshFromObj.h"
#include "../examples/OpenGLWindow/GLInstanceGraphicsShape.h"
#include "../examples/CommonInterfaces/CommonRigidBodyBase.h"
#include "../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.h"
#include "../examples/OpenGLWindow/GLInstanceGraphicsShape.h"
#include "../examples/Importers/ImportObjDemo/Wavefront2GLInstanceGraphicsShape.h"
#include "../examples/Utils/b3ResourcePath.h"

namespace physim{

	class PhySim{
		public:
			btDiscreteDynamicsWorld* dynamicsWorld;
			std::map<std::string, btRigidBody*> rBodyMap;
			std::map<std::string, btCollisionShape*> cShapes;

			PhySim();
			~PhySim();
			void addTable(float tableHt);
			void initRigidBody(std::string);
			void addObject(std::string objName, Eigen::Isometry3d tform, float mass);
			void removeObject(std::string objName);
			void simulate(int num_steps);
			void getTransform(std::string objName, Eigen::Isometry3d &tform);

	};

} //namespace
#endif
