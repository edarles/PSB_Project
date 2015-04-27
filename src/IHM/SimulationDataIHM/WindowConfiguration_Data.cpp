#include <WindowConfiguration_Data.h>

WindowConfiguration_Data::WindowConfiguration_Data(QWidget* widget)
{
}
WindowConfiguration_Data::~WindowConfiguration_Data()
{
	delete(data);
}
SimulationData* WindowConfiguration_Data::getData()
{
	return data;
}
