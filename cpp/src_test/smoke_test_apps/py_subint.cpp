#include <Python.h>
#include <pybind11/embed.h>

#include <string>
#include <thread>

#include "YABTE/Utilities/Python/Interpreters.hpp"

using std::string_literals::operator""s;

using YABTE::Utilities::Python::Interpreter,
    YABTE::Utilities::Python::SubInterpreter,
    YABTE::Utilities::Python::ThreadsAllowedScope;

namespace py = pybind11;

// runs in a new thread
void f(PyInterpreterState* interp, const char* tname) {
    auto code = R"PY(

import sys
import _thread

print(f"TNAME: {_thread.get_native_id()} sys.xxx={getattr(sys, 'xxx', 'attribute not set')}")

    )PY"s;

    code.replace(code.find("TNAME"), 5, tname);

    SubInterpreter::ThreadScope scope(interp);
    // PyRun_SimpleString(code.c_str());

    py::exec(code.c_str());
}

int main() {
    Interpreter init;

    SubInterpreter s1;
    SubInterpreter s2;

    PyRun_SimpleString(R"PY(

# set sys.xxx, it will only be reflected in t4, which runs in the context of the main interpreter

import sys
import _thread

sys.xxx = ['abc']
print(f'main: {_thread.get_native_id()} setting sys.xxx={sys.xxx}')

    )PY");

    std::thread t1{f, s1.interp(), "t1(s1)"};
    std::thread t2{f, s2.interp(), "t2(s2)"};
    std::thread t3{f, s1.interp(), "t3(s1)"};
    std::thread t4{f, SubInterpreter::current(), "t4(main)"};

    ThreadsAllowedScope t;

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    return 0;
}
