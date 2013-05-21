#include <windowConfiguration_Mesh.h>

/*********************************************************************************************/
/*********************************************************************************************/
WindowConfiguration_Mesh::WindowConfiguration_Mesh(GLWidget *widget)
{
	onglets = new QTabWidget();
    	onglets->setGeometry(300, 200, 240, 160);

	QWidget *page1 = new QWidget();
	QWidget *page2 = new QWidget();
	onglets->addTab(page1,"Mesh");
	onglets->addTab(page2,"Parameters");

	buttonLoad  = new QPushButton(tr("File"),this);
	connect(buttonLoad, SIGNAL(clicked()), this , SLOT(loadFile()));
	QVBoxLayout *layout1 = new QVBoxLayout();
	layout1->addWidget(buttonLoad);
	page1->setLayout(layout1);
	
	elastLabel = new QLabel("Damping",page2);
	elast = new QDoubleSpinBox(page2);
	elast->setValue(0.8);
	elast->setMinimum(0.0);
	elast->setMaximum(1.0);
	frictionLabel = new QLabel("Friction",page2);
	friction = new QDoubleSpinBox(page2);
	friction->setValue(0.2);
	friction->setMinimum(0.0);
	friction->setMaximum(1.0);

	container = new QCheckBox(tr("container"),page2);
	container->setChecked(true);
	connect(container,SIGNAL(stateChanged(int)), this, SLOT(changedContainer(int)));

	QVBoxLayout *layout2 = new QVBoxLayout();
	QGridLayout *grid5 = new QGridLayout();
	grid5->addWidget(elastLabel,0,0);
	grid5->addWidget(elast,0,1);
	layout2->addLayout(grid5);
	QGridLayout *grid6 = new QGridLayout();
	grid6->addWidget(frictionLabel,0,0);
	grid6->addWidget(friction,0,1);
	layout2->addLayout(grid6);
	layout2->addWidget(container);
	page2->setLayout(layout2);

	QGridLayout *grid7 = new QGridLayout();
	buttonOK = new QPushButton(tr("OK"),this);
	connect(buttonOK, SIGNAL(clicked()), this , SLOT(accept()));
	buttonCancel = new QPushButton(tr("Cancel"),this);
        connect(buttonCancel, SIGNAL(clicked()), this , SLOT(cancel()));
	grid7->addWidget(buttonOK,0,0);
	grid7->addWidget(buttonCancel,0,1);

	layout = new QVBoxLayout();
	layout->addWidget(onglets);
	layout->addLayout(grid7);

	setLayout(layout);
	setWindowTitle("Configuration");

	M = new MeshCollision();
	loader = NULL;
	this->glWidget = widget;
}
/*********************************************************************************************/
/*********************************************************************************************/
WindowConfiguration_Mesh::~WindowConfiguration_Mesh()
{
	delete(M);
}
/*********************************************************************************************/
/*********************************************************************************************/
void WindowConfiguration_Mesh::accept()
{
	System *system = glWidget->getSystem();
	system->addObjectCollision(M);
	glWidget->getDisplay()->setObjectIsCreate();
	close();
}
/*********************************************************************************************/
void WindowConfiguration_Mesh::cancel()
{
	delete(M);
	close();
}
/*********************************************************************************************/
/*********************************************************************************************/
void WindowConfiguration_Mesh::loadFile()
{
	QStringList type_load;
 	current_dir = "Ressources/OBJ/";
 	type_load << "OBJ file (*.obj)";
 	std::string filename = getOpenFileName("Specify a OBJ file", type_load,0);
 	if (filename != ""){
		if(loader!=NULL) delete(loader);
		loader = new ObjLoader();
		Mesh mesh = loader->load(filename.c_str());
		M = new MeshCollision(elast->value(),friction->value(),container->isChecked());
		M->setFaces(mesh.getFaces());
		M->storeFacesOnGPU();
		glWidget->getDisplay()->displayObjectCollision(M);
  	}	
}
/*********************************************************************************************/
/*********************************************************************************************/
MeshCollision* WindowConfiguration_Mesh::getMesh()
{
	return M;
}
/*********************************************************************************************/
/*********************************************************************************************/
void WindowConfiguration_Mesh::displayMesh(double d)
{
	if(M==NULL) delete(M);
	if(loader!=NULL){
		//M = new MeshCollision(loader->getFaces(),elast->value(),friction->value(),container->isChecked());
		glWidget->getDisplay()->displayObjectCollision(M);
	}
}
/*********************************************************************************************/
void WindowConfiguration_Mesh::changedIsContainer(int state)
{
	if(state==0 && M!=NULL){
		if(M->getIsContainer()){
			M->setIsContainer(false);
			M->inverseNormales();
		}
	}
  	if(state==1 && M!=NULL){
		if(!M->getIsContainer()){
			M->setIsContainer(true);
			M->inverseNormales();
		}
	}
}
/*********************************************************************************************/
/*********************************************************************************************/
string WindowConfiguration_Mesh::getOpenFileName(const QString & caption,
                                    const QStringList & filters,
                                    int * ind_filter)
{
   QString filename;
   QFileDialog open_dialog(0, caption, current_dir.path());

   if (ind_filter != NULL) *ind_filter = -1;
   open_dialog.setFilters(filters);
   open_dialog.setAcceptMode(QFileDialog::AcceptOpen);
   open_dialog.setFileMode(QFileDialog::ExistingFile);

   if (open_dialog.exec())
   {
      filename = open_dialog.selectedFiles().at(0);
      current_dir = open_dialog.directory();

      if (ind_filter != NULL)
      {
         for ((*ind_filter) = 0;
               filters.at(*ind_filter) != open_dialog.selectedFilter();
               (*ind_filter)++);
      }
      return filename.toStdString();
   }
   return std::string("");
}
/*********************************************************************************************/
/*********************************************************************************************/
