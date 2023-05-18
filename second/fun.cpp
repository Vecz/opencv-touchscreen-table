#include "Dll1.h"
#include <cstddef>
#include <pybind11/pybind11.h>
#include <iostream>
using namespace std;
namespace py = pybind11;


py::object interfac(py::list & array_c, py::list& array_p, py::list& cords, py::list& ssize_a, py::list& ssize_b) {
    int size_a = ssize_a[0].cast<int>(), size_b = ssize_b[0].cast<int>();
    /*
    int** a_c = new int*[size_a];
    int** a_p = new int* [size_b];
    for (int i = 0; i < size_a; i++) {
        a_c[i] = new int[size_b];
        a_p[i] = new int[size_b];
    }
    int j = 0;
    for (int i = 0; j < size_b; i++) {
        if (i == size_a) {
            j++;
            i = 0;
        }
        //cout << i << " " << j << endl;
        //cout << array_c[i].cast<int>() << endl;
        a_c[i][j] = array_c[i].cast<int>();
        a_p[i][j] = array_p[i].cast<int>();
    }
    */
    int* a_c = new int[array_c.size()];
    int* a_p = new int[array_c.size()];
    for (int i = 0; i < array_c.size(); i++) {
        a_c[i] = array_c[i].cast<int>();
        a_p[i] = array_p[i].cast<int>();
    }
    int* p = new int[4];
    for (int i = 0; i < 4; i++) {
        p[i] = cords[i].cast<int>();
    }
	int* ans = NULL;
	loading(a_c, a_p, p, ans);
    /*
    for (int i = 0; i < size_a; i++) {
        delete[] a_c[i];
        delete[] a_p[i];
    }
    */
    delete[] a_c; delete[] a_p; delete[] p;
    py::list q;
    q.append(ans[0]);
    q.append(ans[1]);
	return q;
}

PYBIND11_MODULE(opencv_job, m) {
    m.def("interface", &interfac, R"pbdoc(
        interface
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}