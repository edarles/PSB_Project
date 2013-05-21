#include <WindowConfiguration_Data_SPHSystem.h>

/*********************************************************************************************/
/*********************************************************************************************/
WindowConfiguration_Data_SPHSystem::WindowConfiguration_Data_SPHSystem(QWidget* widget):WindowConfiguration_Data(widget)
{
	particleRadiusLabel = new QLabel("Particle Radius",widget);
 	particleRadius = new QDoubleSpinBox(widget);
	particleRadius->setMinimum(0.01);
	particleRadius->setMaximum(1.0);
	particleRadius->setValue(0.02);
	connect(particleRadius, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	particleMassLabel = new QLabel("Particle Mass",widget);
 	particleMass = new QDoubleSpinBox(widget);
	particleMass->setMinimum(0.01);
	particleMass->setMaximum(100.0);
	particleMass->setValue(0.02);
	connect(particleMass, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	restDensityLabel = new QLabel("Rest Density",widget);
	restDensity = new QDoubleSpinBox(widget);
	restDensity->setMinimum(1.0);
	restDensity->setMaximum(10000);
	restDensity->setValue(998.29);
        connect(restDensity, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	viscosityLabel = new QLabel("Viscosity",widget);
	viscosity = new QDoubleSpinBox(widget);
	viscosity->setMinimum(0.0);
	viscosity->setMaximum(100);
	viscosity->setValue(3.5);
	connect(viscosity, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	surfaceTensionLabel = new QLabel("Surface tension",widget);
	surfaceTension = new QDoubleSpinBox(widget);
	surfaceTension->setDecimals(4);
	surfaceTension->setMinimum(0.0);
	surfaceTension->setMaximum(100);
	surfaceTension->setValue(0.0728);
	connect(surfaceTension, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	gasStiffnessLabel = new QLabel("Gas Stiffness",widget);
	gasStiffness = new QDoubleSpinBox(widget);
	gasStiffness->setMinimum(0.0);
	gasStiffness->setMaximum(100);
	gasStiffness->setValue(3);
	connect(gasStiffness, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

	kernelParticlesLabel = new QLabel("Kernel Particles",widget);
	kernelParticles = new QDoubleSpinBox(widget);
	kernelParticles->setMinimum(0.0);
	kernelParticles->setMaximum(100);
	kernelParticles->setValue(20);
	connect(kernelParticles, SIGNAL(valueChanged(double)), this, SLOT(changeData(double))); 

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
	grid3->addWidget(restDensityLabel,0,0);
	grid3->addWidget(restDensity,0,1); 
	layout1->addLayout(grid3);

	QGridLayout *grid4 = new QGridLayout();
	grid4->addWidget(viscosityLabel,0,0);
	grid4->addWidget(viscosity,0,1); 
	layout1->addLayout(grid4);

	QGridLayout *grid5 = new QGridLayout();
	grid5->addWidget(surfaceTensionLabel,0,0);
	grid5->addWidget(surfaceTension,0,1); 
	layout1->addLayout(grid5);

	QGridLayout *grid6 = new QGridLayout();
	grid6->addWidget(gasStiffnessLabel,0,0);
	grid6->addWidget(gasStiffness,0,1); 
	layout1->addLayout(grid6);

	QGridLayout *grid7 = new QGridLayout();
	grid7->addWidget(kernelParticlesLabel,0,0);
	grid7->addWidget(kernelParticles,0,1); 
	layout1->addLayout(grid7);

	QGridLayout *grid8 = new QGridLayout();
	grid8->addWidget(colorButton,0,0); 
	layout1->addLayout(grid8);

	widget->setLayout(layout1);
    
        color = QColor(255,0,255);
	
	data = new SimulationData_SPHSystem(particleRadius->value(),particleMass->value(),Vector3(color.red()/255,color.green()/255,
		   color.blue()/255),restDensity->value(),viscosity->value(),surfaceTension->value(),gasStiffness->value(),
		   kernelParticles->value());
}
/*********************************************************************************************/
WindowConfiguration_Data_SPHSystem::~WindowConfiguration_Data_SPHSystem()
{
}
/*********************************************************************************************/
/*********************************************************************************************/
void WindowConfiguration_Data_SPHSystem::changeData(double newValue)
{
	if(data!=NULL)	delete(data);
	data = new SimulationData_SPHSystem(particleRadius->value(),particleMass->value(),Vector3(color.red()/255,color.green()/255,
		   color.blue()/255),restDensity->value(),viscosity->value(),surfaceTension->value(),gasStiffness->value(),
		   kernelParticles->value());
}
/*********************************************************************************************/
/*********************************************************************************************/
void WindowConfiguration_Data_SPHSystem::changeData(QColor newValue)
{
	if(data!=NULL)	delete(data);
	data = new SimulationData_SPHSystem(particleRadius->value(),particleMass->value(),
	       Vector3((double)(newValue.red()/255),(double)(newValue.green()/255),(double)(newValue.blue()/255)),
	       restDensity->value(),viscosity->value(),surfaceTension->value(),gasStiffness->value(),kernelParticles->value());
}
/*********************************************************************************************/
/*********************************************************************************************/
void WindowConfiguration_Data_SPHSystem::setColor()
{
	QColorDialog *colorDialog = new QColorDialog();
	connect(colorDialog,SIGNAL(currentColorChanged(QColor)),this, SLOT(changeData(QColor)));
	colorDialog->open();
}
/*********************************************************************************************/
/*********************************************************************************************/
