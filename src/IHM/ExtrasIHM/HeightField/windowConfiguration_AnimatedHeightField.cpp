#include <windowConfiguration_AnimatedHeightField.h>
#include <windowConfiguration_CombinedHeightField.h>
/****************************************************************************************************/
/****************************************************************************************************/
WindowConfiguration_AnimatedHeightField::WindowConfiguration_AnimatedHeightField(WindowConfiguration_CombinedHeightField* parent, 
									        GLWidget *widget)
{
   this->parent = parent;

   if(parent == NULL)
   {
            std::cout<<"TROLOLO"<<std::endl;
         onglets = new QTabWidget();
             onglets->setGeometry(300, 200, 240, 160);

         QWidget *page1 = new QWidget();
         QWidget *page2 = new QWidget();
         QWidget *page3 = new QWidget();
         onglets->addTab(page1,"HeightField");
         onglets->addTab(page2,"Animated");
         onglets->addTab(page3,"Parameters");

         originLabel = new QLabel("Origin",page1);
         OX = new QDoubleSpinBox(page1);
         OY = new QDoubleSpinBox(page1);
         OZ = new QDoubleSpinBox(page1);
         OX->setMinimum(-1000); OX->setMaximum(1000);
         OY->setMinimum(-1000); OY->setMaximum(1000);
         OZ->setMinimum(-1000); OZ->setMaximum(1000);
         OX->setValue(0.0); OY->setValue(0.0); OZ->setValue(0.0);
         connect(OX, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 
         connect(OY, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 
         connect(OZ, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

         widthLabel = new QLabel("width",page1);
         width = new QDoubleSpinBox(page1);
         width->setMinimum(0.01);
         width->setValue(1.0);
         connect(width, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

         lengthLabel = new QLabel("length",page1);
         length = new QDoubleSpinBox(page1);
         length->setMinimum(0.01);
         length->setValue(1.0);
         connect(length, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

         dxLabel = new QLabel("dx",page1);
         dx = new QDoubleSpinBox(page1);
         dx->setMinimum(0.01);
         dx->setValue(0.05);
         connect(dx, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

         dzLabel = new QLabel("dz",page1);
         dz = new QDoubleSpinBox(page1);
         dz->setMinimum(0.01);
         dz->setValue(0.05);
         connect(dz, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

         QVBoxLayout *layout1 = new QVBoxLayout();

         QGridLayout *grid1 = new QGridLayout();
         grid1->addWidget(originLabel,0,0);
         grid1->addWidget(OX,0,1); 
         grid1->addWidget(OY,0,2);
         grid1->addWidget(OZ,0,3);
         layout1->addLayout(grid1);

         QGridLayout *grid2 = new QGridLayout();
         grid2->addWidget(lengthLabel,0,0);
         grid2->addWidget(length,0,1);
         layout1->addLayout(grid2);

         QGridLayout *grid3 = new QGridLayout();
         grid3->addWidget(widthLabel,0,0);
         grid3->addWidget(width,0,1);
         layout1->addLayout(grid3);

         QGridLayout *grid4 = new QGridLayout();
         grid4->addWidget(dxLabel,0,0);
         grid4->addWidget(dx,0,1);
         layout1->addLayout(grid4);

         QGridLayout *grid5 = new QGridLayout();
         grid5->addWidget(dzLabel,0,0);
         grid5->addWidget(dz,0,1);
         layout1->addLayout(grid5);

         page1->setLayout(layout1);

         nbFuncLabel = new QLabel("Nb functions",page2);
         nbFunc = new QSpinBox(page2);
         nbFunc->setValue(20);
         nbFunc->setMaximum(100);
         connect(nbFunc, SIGNAL(valueChanged(int)), this, SLOT(displayHeightField(int))); 

         AMinLabel = new QLabel("AMin",page2);
         AMin = new QDoubleSpinBox(page2);
         AMin->setDecimals(4);
         AMin->setValue(0.005);
         AMin->setMinimum(0.0);
         AMin->setMaximum(10.0);
         connect(AMin, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

         AMaxLabel = new QLabel("AMax",page2);
         AMax = new QDoubleSpinBox(page2);
         AMax->setDecimals(4);
         AMax->setValue(0.008);
         AMax->setMinimum(0.0);
         AMax->setMaximum(1.0);
         connect(AMax, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

         kMinLabel = new QLabel("kMin",page2);
         kMin = new QDoubleSpinBox(page2);
         kMin->setDecimals(2);
         kMin->setValue(15.0);
         kMin->setMinimum(0.0);
         kMin->setMaximum(100.0);
         connect(kMin, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

         kMaxLabel = new QLabel("kMax",page2);
         kMax = new QDoubleSpinBox(page2);
         kMax->setDecimals(2);
         kMax->setValue(20.0);
         kMax->setMinimum(0.0);
         kMax->setMaximum(100.0);
         connect(kMax, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

         thetaMinLabel = new QLabel("thetaMin",page2);
         thetaMin = new QDoubleSpinBox(page2);
         thetaMin->setDecimals(2);
         thetaMin->setValue(-10);
         thetaMin->setMinimum(-100);
         thetaMin->setMaximum(100);
         connect(thetaMin, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

         thetaMaxLabel = new QLabel("thetaMax",page2);
         thetaMax = new QDoubleSpinBox(page2);
         thetaMax->setDecimals(2);
         thetaMax->setValue(10);
         thetaMax->setMinimum(-100);
         thetaMax->setMaximum(100);
         connect(thetaMax, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

         phiMinLabel = new QLabel("phiMin",page2);
         phiMin = new QDoubleSpinBox(page2);
         phiMin->setDecimals(2);
         phiMin->setValue(-10);
         phiMin->setMinimum(-100);
         phiMin->setMaximum(100);
         connect(phiMin, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

         phiMaxLabel = new QLabel("phiMax",page2);
         phiMax = new QDoubleSpinBox(page2);
         phiMax->setDecimals(2);
         phiMax->setValue(10);
         phiMax->setMinimum(-100);
         phiMax->setMaximum(100);
         connect(phiMax, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

         omegaMinLabel = new QLabel("omegaMinx",page2);
         omegaMin = new QDoubleSpinBox(page2);
         omegaMin->setDecimals(2);
         omegaMin->setValue(10);
         omegaMin->setMinimum(-100);
         omegaMin->setMaximum(100);
         connect(omegaMin, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

             omegaMaxLabel = new QLabel("omegaMax",page2);
         omegaMax = new QDoubleSpinBox(page2);
         omegaMax->setDecimals(2);
         omegaMax->setValue(10);
         omegaMax->setMinimum(-100);
         omegaMax->setMaximum(100);
         connect(omegaMax, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

             tLabel = new QLabel("time step",page2);
         t = new QDoubleSpinBox(page2);
         t->setDecimals(2);
         t->setValue(0.01);
         t->setMinimum(0);
         t->setMaximum(1);
         connect(t, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 


         buttonLoad  = new QPushButton(tr("Load spectrum"),this);
         connect(buttonLoad, SIGNAL(clicked()), this , SLOT(loadSpectrum()));

         buttonSave  = new QPushButton(tr("Save spectrum"),this);
         connect(buttonSave, SIGNAL(clicked()), this , SLOT(saveSpectrum()));

         QVBoxLayout *layout2 = new QVBoxLayout();

         QGridLayout *grid6 = new QGridLayout();
         grid6->addWidget(nbFuncLabel,0,0);
         grid6->addWidget(nbFunc,0,1);
         layout2->addLayout(grid6);

         QGridLayout *grid7 = new QGridLayout();
         grid7->addWidget(AMinLabel,0,0);
         grid7->addWidget(AMin,0,1);
         layout2->addLayout(grid7);

         QGridLayout *grid8 = new QGridLayout();
         grid8->addWidget(AMaxLabel,0,0);
         grid8->addWidget(AMax,0,1);
         layout2->addLayout(grid8);

         QGridLayout *grid9 = new QGridLayout();
         grid9->addWidget(kMinLabel,0,0);
         grid9->addWidget(kMin,0,1);
         layout2->addLayout(grid9);

         QGridLayout *grid10 = new QGridLayout();
         grid10->addWidget(kMaxLabel,0,0);
         grid10->addWidget(kMax,0,1);
         layout2->addLayout(grid10);

         QGridLayout *grid11 = new QGridLayout();
         grid11->addWidget(thetaMinLabel,0,0);
         grid11->addWidget(thetaMin,0,1);
         layout2->addLayout(grid11);

         QGridLayout *grid12 = new QGridLayout();
         grid12->addWidget(thetaMaxLabel,0,0);
         grid12->addWidget(thetaMax,0,1);
         layout2->addLayout(grid12);

         QGridLayout *grid15 = new QGridLayout();
         grid15->addWidget(phiMinLabel,0,0);
         grid15->addWidget(phiMin,0,1);
         layout2->addLayout(grid15);

         QGridLayout *grid16 = new QGridLayout();
         grid16->addWidget(phiMaxLabel,0,0);
         grid16->addWidget(phiMax,0,1);
         layout2->addLayout(grid16);

             QGridLayout *grid17 = new QGridLayout(); //MATHIAS
         grid17->addWidget(omegaMinLabel,0,0);
         grid17->addWidget(omegaMin,0,1);
         layout2->addLayout(grid17);

             QGridLayout *grid18 = new QGridLayout(); //MATHIAS
         grid18->addWidget(omegaMaxLabel,0,0);
         grid18->addWidget(omegaMax,0,1);
         layout2->addLayout(grid18);

             QGridLayout *grid19 = new QGridLayout(); //MATHIAS
         grid19->addWidget(tLabel,0,0);
         grid19->addWidget(t,0,1);
         layout2->addLayout(grid19);

         layout2->addWidget(buttonLoad);
         layout2->addWidget(buttonSave);

         page2->setLayout(layout2);

         elastLabel = new QLabel("Restitution",page3);
         elast = new QDoubleSpinBox(page3);
         elast->setValue(0.45);
         elast->setMinimum(0.0);
         elast->setMaximum(1.0);
         connect(elast, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

         QVBoxLayout *layout3 = new QVBoxLayout();

         QGridLayout *grid13 = new QGridLayout();
         grid13->addWidget(elastLabel,0,0);
         grid13->addWidget(elast,0,1);
         layout3->addLayout(grid13);

         page3->setLayout(layout3);

         QGridLayout *grid14 = new QGridLayout();
         buttonOK = new QPushButton(tr("OK"),this);
         connect(buttonOK, SIGNAL(clicked()), this , SLOT(accept()));
         buttonCancel = new QPushButton(tr("Cancel"),this);
             connect(buttonCancel, SIGNAL(clicked()), this , SLOT(cancel()));
         grid14->addWidget(buttonOK,0,0);
         grid14->addWidget(buttonCancel,0,1);

         layout = new QVBoxLayout();
         layout->addWidget(onglets);
         layout->addLayout(grid14);

         setLayout(layout);
         setWindowTitle("Configuration");

        H = new AnimatedPeriodicHeightField();
     	H->create(Vector3(OX->value(),OY->value(),OZ->value()),(float)length->value(),(float)width->value(),dx->value(),
            dz->value(), nbFunc->value(),AMin->value(),AMax->value(),kMin->value(),kMax->value(),thetaMin->value(),
            thetaMax->value(),phiMin->value(), phiMax->value(),omegaMin->value(),omegaMax->value(),t->value(),elast->value());
        this->glWidget = widget;
        H->display(Vector3(255,255,255));
    }
    else {
        onglets = new QTabWidget();
            onglets->setGeometry(300, 200, 240, 160);

        QWidget *page2 = new QWidget();
        onglets->addTab(page2,"Animated");

        nbFuncLabel = new QLabel("Nb functions",page2);
        nbFunc = new QSpinBox(page2);
        nbFunc->setValue(20);
        nbFunc->setMaximum(100);
        connect(nbFunc, SIGNAL(valueChanged(int)), this, SLOT(displayHeightField(int))); 

        AMinLabel = new QLabel("AMin",page2);
        AMin = new QDoubleSpinBox(page2);
        AMin->setDecimals(4);
        AMin->setValue(0.005);
        AMin->setMinimum(0.0);
        AMin->setMaximum(10.0);
        connect(AMin, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

        AMaxLabel = new QLabel("AMax",page2);
        AMax = new QDoubleSpinBox(page2);
        AMax->setDecimals(4);
        AMax->setValue(0.008);
        AMax->setMinimum(0.0);
        AMax->setMaximum(1.0);
        connect(AMax, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

        kMinLabel = new QLabel("kMin",page2);
        kMin = new QDoubleSpinBox(page2);
        kMin->setDecimals(2);
        kMin->setValue(15.0);
        kMin->setMinimum(0.0);
        kMin->setMaximum(100.0);
        connect(kMin, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

        kMaxLabel = new QLabel("kMax",page2);
        kMax = new QDoubleSpinBox(page2);
        kMax->setDecimals(2);
        kMax->setValue(20.0);
        kMax->setMinimum(0.0);
        kMax->setMaximum(100.0);
        connect(kMax, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

        thetaMinLabel = new QLabel("thetaMin",page2);
        thetaMin = new QDoubleSpinBox(page2);
        thetaMin->setDecimals(2);
        thetaMin->setValue(-10);
        thetaMin->setMinimum(-100);
        thetaMin->setMaximum(100);
        connect(thetaMin, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

        thetaMaxLabel = new QLabel("thetaMax",page2);
        thetaMax = new QDoubleSpinBox(page2);
        thetaMax->setDecimals(2);
        thetaMax->setValue(10);
        thetaMax->setMinimum(-100);
        thetaMax->setMaximum(100);
        connect(thetaMax, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

        phiMinLabel = new QLabel("phiMin",page2);
        phiMin = new QDoubleSpinBox(page2);
        phiMin->setDecimals(2);
        phiMin->setValue(-10);
        phiMin->setMinimum(-100);
        phiMin->setMaximum(100);
        connect(phiMin, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

        phiMaxLabel = new QLabel("phiMax",page2);
        phiMax = new QDoubleSpinBox(page2);
        phiMax->setDecimals(2);
        phiMax->setValue(10);
        phiMax->setMinimum(-100);
        phiMax->setMaximum(100);
        connect(phiMax, SIGNAL(valueChanged(double)), this, SLOT(displayHeightField(double))); 

        buttonLoad  = new QPushButton(tr("Load spectrum"),this);
        connect(buttonLoad, SIGNAL(clicked()), this , SLOT(loadSpectrum()));

        buttonSave  = new QPushButton(tr("Save spectrum"),this);
        connect(buttonSave, SIGNAL(clicked()), this , SLOT(saveSpectrum()));

        QVBoxLayout *layout2 = new QVBoxLayout();

        QGridLayout *grid6 = new QGridLayout();
        grid6->addWidget(nbFuncLabel,0,0);
        grid6->addWidget(nbFunc,0,1);
        layout2->addLayout(grid6);

        QGridLayout *grid7 = new QGridLayout();
        grid7->addWidget(AMinLabel,0,0);
        grid7->addWidget(AMin,0,1);
        layout2->addLayout(grid7);

        QGridLayout *grid8 = new QGridLayout();
        grid8->addWidget(AMaxLabel,0,0);
        grid8->addWidget(AMax,0,1);
        layout2->addLayout(grid8);

        QGridLayout *grid9 = new QGridLayout();
        grid9->addWidget(kMinLabel,0,0);
        grid9->addWidget(kMin,0,1);
        layout2->addLayout(grid9);

        QGridLayout *grid10 = new QGridLayout();
        grid10->addWidget(kMaxLabel,0,0);
        grid10->addWidget(kMax,0,1);
        layout2->addLayout(grid10);

        QGridLayout *grid11 = new QGridLayout();
        grid11->addWidget(thetaMinLabel,0,0);
        grid11->addWidget(thetaMin,0,1);
        layout2->addLayout(grid11);

        QGridLayout *grid12 = new QGridLayout();
        grid12->addWidget(thetaMaxLabel,0,0);
        grid12->addWidget(thetaMax,0,1);
        layout2->addLayout(grid12);

        QGridLayout *grid15 = new QGridLayout();
        grid15->addWidget(phiMinLabel,0,0);
        grid15->addWidget(phiMin,0,1);
        layout2->addLayout(grid15);

        QGridLayout *grid16 = new QGridLayout();
        grid16->addWidget(phiMaxLabel,0,0);
        grid16->addWidget(phiMax,0,1);
        layout2->addLayout(grid16);

        layout2->addWidget(buttonLoad);
        layout2->addWidget(buttonSave);

        page2->setLayout(layout2);

        QGridLayout *grid14 = new QGridLayout();
        buttonOK = new QPushButton(tr("add"),this);
        connect(buttonOK, SIGNAL(clicked()), this , SLOT(add()));
        buttonCancel = new QPushButton(tr("Cancel"),this);
            connect(buttonCancel, SIGNAL(clicked()), this , SLOT(cancel()));
        grid14->addWidget(buttonOK,0,0);
        grid14->addWidget(buttonCancel,0,1);

        layout = new QVBoxLayout();
        layout->addWidget(onglets);
        layout->addLayout(grid14);

        setLayout(layout);
        setWindowTitle("Configuration");

        H = new AnimatedPeriodicHeightField();
     	H->create(Vector3(OX->value(),OY->value(),OZ->value()),(float)length->value(),(float)width->value(),dx->value(),
            dz->value(), nbFunc->value(),AMin->value(),AMax->value(),kMin->value(),kMax->value(),thetaMin->value(),
            thetaMax->value(),phiMin->value(), phiMax->value(),omegaMin->value(),omegaMax->value(),t->value(),elast->value());
        this->glWidget = widget;
        H->display(Vector3(255,255,255));
    }
}
/****************************************************************************************************/
WindowConfiguration_AnimatedHeightField::~WindowConfiguration_AnimatedHeightField()
{
}
/****************************************************************************************************/
/****************************************************************************************************/
void WindowConfiguration_AnimatedHeightField::accept()
{
	System *system = glWidget->getSystem();
    //  MATHIAS -> peut être ajouter une méthode addAnimatedHeightField ???
//    	system->addObjectCollision(H);
	glWidget->getDisplay()->setObjectIsCreate();
	close();
}
/****************************************************************************************************/
void WindowConfiguration_AnimatedHeightField::cancel()
{
	close();
}
/****************************************************************************************************/
/****************************************************************************************************/
void WindowConfiguration_AnimatedHeightField::add()
{
	if(parent!=NULL){
		H = new AnimatedPeriodicHeightField();
     	H->create(Vector3(OX->value(),OY->value(),OZ->value()),(float)length->value(),(float)width->value(),dx->value(),
            dz->value(), nbFunc->value(),AMin->value(),AMax->value(),kMin->value(),kMax->value(),thetaMin->value(),
            thetaMax->value(),phiMin->value(), phiMax->value(),omegaMin->value(),omegaMax->value(),t->value(),elast->value());
        H->display(Vector3(255,255,255));
		close();
	}
}
/****************************************************************************************************/
/****************************************************************************************************/
AnimatedPeriodicHeightField* WindowConfiguration_AnimatedHeightField::getHeightField()
{
	return H;
}
/****************************************************************************************************/
/****************************************************************************************************/
void WindowConfiguration_AnimatedHeightField::displayHeightField(double d)
{
	if(H==NULL) delete(H);
	H = new AnimatedPeriodicHeightField();
        if(parent==NULL){
     	H->create(Vector3(OX->value(),OY->value(),OZ->value()),(float)length->value(),(float)width->value(),dx->value(),
            dz->value(), nbFunc->value(),AMin->value(),AMax->value(),kMin->value(),kMax->value(),thetaMin->value(),
            thetaMax->value(),phiMin->value(), phiMax->value(),omegaMin->value(),omegaMax->value(),t->value(),elast->value());
        H->display(Vector3(255,255,255));
	}
	else {
     	H->create(Vector3(OX->value(),OY->value(),OZ->value()),(float)length->value(),(float)width->value(),dx->value(),
            dz->value(), nbFunc->value(),AMin->value(),AMax->value(),kMin->value(),kMax->value(),thetaMin->value(),
            thetaMax->value(),phiMin->value(), phiMax->value(),omegaMin->value(),omegaMax->value(),t->value(),elast->value());
        H->display(Vector3(255,255,255));
	}
}
/****************************************************************************************************/
/****************************************************************************************************/
void WindowConfiguration_AnimatedHeightField::displayHeightField(int d)
{
	if(H==NULL) delete(H);
	H = new AnimatedPeriodicHeightField();
	if(parent==NULL){
	
     	H->create(Vector3(OX->value(),OY->value(),OZ->value()),(float)length->value(),(float)width->value(),dx->value(),
            dz->value(), nbFunc->value(),AMin->value(),AMax->value(),kMin->value(),kMax->value(),thetaMin->value(),
            thetaMax->value(),phiMin->value(), phiMax->value(),omegaMin->value(),omegaMax->value(),t->value(),elast->value());
      
        H->display(Vector3(255,255,255));
	}
	else {
	
     	H->create(Vector3(OX->value(),OY->value(),OZ->value()),(float)length->value(),(float)width->value(),dx->value(),
            dz->value(), nbFunc->value(),AMin->value(),AMax->value(),kMin->value(),kMax->value(),thetaMin->value(),
            thetaMax->value(),phiMin->value(), phiMax->value(),omegaMin->value(),omegaMax->value(),t->value(),elast->value());
       
        H->display(Vector3(255,255,255));
	}
}
/****************************************************************************************************/
/****************************************************************************************************/
void WindowConfiguration_AnimatedHeightField::loadSpectrum()
{
	QStringList type_load;
 	current_dir = "../Ressources/Spectrum";
 	type_load << "spc file (*.spc)";
 	std::string filename = getOpenFileName("Specify a SPC file", type_load,0);
 	if (filename != ""){
//		H->loadSpectrum(filename.c_str());
//		glWidget->getDisplay()->displayObjectCollision(H);
  	}	
}
/****************************************************************************************************/
/****************************************************************************************************/
void WindowConfiguration_AnimatedHeightField::saveSpectrum()
{
	current_dir = "../Ressources/Spectrum";
 	QString filename = QFileDialog::getSaveFileName(this, tr("Save spectrum"), "*.spc");
	if (filename != ""){
//		H->saveSpectrum(filename.toStdString().c_str());
	}
}
/****************************************************************************************************/
/****************************************************************************************************/
/*********************************************************************************************/
/*********************************************************************************************/
string WindowConfiguration_AnimatedHeightField::getOpenFileName(const QString & caption,
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
