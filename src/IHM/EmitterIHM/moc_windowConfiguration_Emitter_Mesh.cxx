/****************************************************************************
** Meta object code from reading C++ file 'windowConfiguration_Emitter_Mesh.h'
**
** Created: Sat May 18 15:55:04 2013
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "windowConfiguration_Emitter_Mesh.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'windowConfiguration_Emitter_Mesh.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_WindowConfiguration_Emitter_Mesh[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      34,   33,   33,   33, 0x0a,
      43,   33,   33,   33, 0x0a,
      52,   33,   33,   33, 0x0a,
      63,   33,   33,   33, 0x0a,
      83,   33,   33,   33, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_WindowConfiguration_Emitter_Mesh[] = {
    "WindowConfiguration_Emitter_Mesh\0\0"
    "accept()\0cancel()\0loadFile()\0"
    "displayMesh(double)\0displayMesh(int)\0"
};

void WindowConfiguration_Emitter_Mesh::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        WindowConfiguration_Emitter_Mesh *_t = static_cast<WindowConfiguration_Emitter_Mesh *>(_o);
        switch (_id) {
        case 0: _t->accept(); break;
        case 1: _t->cancel(); break;
        case 2: _t->loadFile(); break;
        case 3: _t->displayMesh((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 4: _t->displayMesh((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData WindowConfiguration_Emitter_Mesh::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject WindowConfiguration_Emitter_Mesh::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_WindowConfiguration_Emitter_Mesh,
      qt_meta_data_WindowConfiguration_Emitter_Mesh, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &WindowConfiguration_Emitter_Mesh::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *WindowConfiguration_Emitter_Mesh::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *WindowConfiguration_Emitter_Mesh::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_WindowConfiguration_Emitter_Mesh))
        return static_cast<void*>(const_cast< WindowConfiguration_Emitter_Mesh*>(this));
    return QWidget::qt_metacast(_clname);
}

int WindowConfiguration_Emitter_Mesh::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    }
    return _id;
}
QT_END_MOC_NAMESPACE