/*****************************************************************************************************/
/*****************************************************************************************************/
#ifndef _FORCE_EXT_GRAVITY_
#define _FORCE_EXT_GRAVITY_

#include <ForceExt_Constante.h>

/*****************************************************************************************************/
/*****************************************************************************************************/
class ForceExt_Gravity : public ForceExt_Constante {

	public:

/*****************************************************************************************************/
/*****************************************************************************************************/
		ForceExt_Gravity();
		ForceExt_Gravity(Vector3 direction, float amplitude);
		ForceExt_Gravity(const ForceExt_Gravity& F);
		~ForceExt_Gravity();
};
/*****************************************************************************************************/

#endif
/*****************************************************************************************************/
/*****************************************************************************************************/
