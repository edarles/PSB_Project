/****************************************************************************
** Meta object code from reading C++ file 'WindowConfiguration_Data_CudaSystem.h'
**
** Created: Mon Jun 3 01:19:13 2013
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "WindowConfiguration_Data_CudaSystem.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'WindowConfiguration_Data_CudaSystem.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_WindowConfiguration_Data_CudaSystem[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      37,   36,   36,   36, 0x0a,
      65,   56,   36,   36, 0x0a,
      84,   36,   36,   36, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_WindowConfiguration_Data_CudaSystem[] = {
    "WindowConfiguration_Data_CudaSystem\0"
    "\0changeData(double)\0newValue\0"
    "changeData(QColor)\0setColor()\0"
};

void WindowConfiguration_Data_CudaSystem::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        WindowConfiguration_Data_CudaSystem *_t = static_cast<WindowConfiguration_Data_CudaSystem *>(_o);
        switch (_id) {
        case 0: _t->changeData((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 1: _t->changeData((*reinterpret_cast< QColor(*)>(_a[1]))); break;
        case 2: _t->setColor(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData WindowConfiguration_Data_CudaSystem::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject WindowConfiguration_Data_CudaSystem::staticMetaObject = {
    { &WindowConfiguration_Data::staticMetaObject, qt_meta_stringdata_WindowConfiguration_Data_CudaSystem,
      qt_meta_data_WindowConfiguration_Data_CudaSystem, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &WindowConfiguration_Data_CudaSystem::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *WindowConfiguration_Data_CudaSystem::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *WindowConfiguration_Data_CudaSystem::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_WindowConfiguration_Data_CudaSystem))
        return static_cast<void*>(const_cast< WindowConfiguration_Data_CudaSystem*>(this));
    return WindowConfiguration_Data::qt_metacast(_clname);
}

int WindowConfiguration_Data_CudaSystem::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = WindowConfiguration_Data::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
