#include <ForceExt_Gravity.h>

/*****************************************************************************************************/
/*****************************************************************************************************/
ForceExt_Gravity::ForceExt_Gravity():ForceExt_Constante(Vector3(0,-1,0),9.81)
{
}
/*****************************************************************************************************/
ForceExt_Gravity::ForceExt_Gravity(Vector3 G, float A):ForceExt_Constante(G,A)
{
}
/*****************************************************************************************************/
ForceExt_Gravity::ForceExt_Gravity(const ForceExt_Gravity &F):ForceExt_Constante(F)
{
}
/*****************************************************************************************************/
ForceExt_Gravity::~ForceExt_Gravity()
{
}
/*****************************************************************************************************/
/*****************************************************************************************************/