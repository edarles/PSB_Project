#ifndef _HEIGHT_FIELD_COLLISION_
#define _HEIGHT_FIELD_COLLISION_

#include <ObjectCollision.h>
/*****************************************************************************************************/
/*****************************************************************************************************/
class HeightFieldCollision : public ObjectCollision 
{
/*****************************************************************************************************/
	public:
		HeightFieldCollision();
		HeightFieldCollision(float elast);
		HeightFieldCollision(const HeightFieldCollision&);
		~HeightFieldCollision();
};
/*****************************************************************************************************/
/*****************************************************************************************************/
#endif
/*****************************************************************************************************/
/*****************************************************************************************************/
