#include <WindowConfiguration_Data_SimpleSystem.h>

/*********************************************************************************************/
/*********************************************************************************************/
WindowConfiguration_Data_SimpleSystem::WindowConfiguration_Data_SimpleSystem(QWidget* widget):WindowConfiguration_Data(widget)
{
	particleRadiusLabel = new QLabel("Particle Radius",widget);
 	particleRadius = new QDoubleSpinBox(widget);
	particleRadius->setMinimum(0.01);
	particleRadius->setMaximum(1.0);
	particleRadius->setValue(0.02);
	connect(particleRadius, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	particleMassLabel = new QLabel("Particle Mass",this);
 	particleMass = new QDoubleSpinBox(this);
	particleMass->setMinimum(0.01);
	particleMass->setMaximum(100.0);
	particleMass->setValue(1.0);
	connect(particleMass, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	colorButton = new QPushButton("Particles color",this);
	connect(colorButton, SIGNAL(pressed()),this, SLOT(setColor()));

	QVBoxLayout *layout1 = new QVBoxLayout();

	QGridLayout *grid1 = new QGridLayout();
	grid1->addWidget(particleRadiusLabel,0,0);
	grid1->addWidget(particleRadius,0,1); 
	layout1->addLayout(grid1);

	QGridLayout *grid2 = new QGridLayout();
	grid2->addWidget(particleMassLabel,0,0);
	grid2->addWidget(particleMass,0,1); 
	layout1->addLayout(grid2);

	QGridLayout *grid3 = new QGridLayout();
	grid3->addWidget(colorButton,0,0); 
	layout1->addLayout(grid3);

	widget->setLayout(layout1);
    
        color = QColor(255,0,255);
	data = new SimulationData_SimpleSystem(particleRadius->value(),particleMass->value(),
					       Vector3(color.red()/255,color.green()/255,color.blue()/255));
}
/*********************************************************************************************/
WindowConfiguration_Data_SimpleSystem::~WindowConfiguration_Data_SimpleSystem()
{
}
/*********************************************************************************************/
/*********************************************************************************************/
void WindowConfiguration_Data_SimpleSystem::changeData(double newValue)
{
	if(data!=NULL)	delete(data);
	data = new SimulationData_SimpleSystem(particleRadius->value(),particleMass->value(),Vector3(0,0,1));
}
/*********************************************************************************************/
/*********************************************************************************************/
void WindowConfiguration_Data_SimpleSystem::changeData(QColor newValue)
{
	if(data!=NULL)	delete(data);
	data = new SimulationData_SimpleSystem(particleRadius->value(),particleMass->value(),
				               Vector3((double)(newValue.red()/255),(double)(newValue.green()/255),(double)(newValue.blue()/255)));
}
/*********************************************************************************************/
/*********************************************************************************************/
void WindowConfiguration_Data_SimpleSystem::setColor()
{
	QColorDialog *colorDialog = new QColorDialog();
	connect(colorDialog,SIGNAL(currentColorChanged(QColor)),this, SLOT(changeData(QColor)));
	colorDialog->open();
}
/*********************************************************************************************/
/*********************************************************************************************/
