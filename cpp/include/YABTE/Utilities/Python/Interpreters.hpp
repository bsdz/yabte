#pragma once

#include <Python.h>
#include <glog/logging.h>

// orig -
// https://github.com/sterin/python-sub-interpreters-multiple-threads-example

namespace YABTE::Utilities::Python {

// https://docs.python.org/3/c-api/init.html#initializing-and-finalizing-the-interpreter
struct Interpreter {
    Interpreter() { Py_InitializeEx(1); }

    ~Interpreter() {
        if (Py_FinalizeEx() < 0) {
            LOG(ERROR) << "Interpreter finalization failed";
        }
    }
};

// https://docs.python.org/3/c-api/init.html#releasing-the-gil-from-extension-code
class ThreadsAllowedScope {
   public:
    ThreadsAllowedScope() { _state = PyEval_SaveThread(); }

    ~ThreadsAllowedScope() { PyEval_RestoreThread(_state); }

   private:
    PyThreadState* _state;
};

// restore the thread state when the object goes out of scope
class ThreadStateRestoreScope {
   public:
    ThreadStateRestoreScope() { _ts = PyThreadState_Get(); }

    ~ThreadStateRestoreScope() { PyThreadState_Swap(_ts); }

   private:
    PyThreadState* _ts;
};

// swap the current thread state with ts, restore when the object goes out of
// scope
class ThreadStateSwapScope {
   public:
    ThreadStateSwapScope(PyThreadState* ts) { _ts = PyThreadState_Swap(ts); }

    ~ThreadStateSwapScope() { PyThreadState_Swap(_ts); }

   private:
    PyThreadState* _ts;
};

// create new thread state for interpreter interp, make it current, and clean up
// on destruction
class ThreadState {
   public:
    ThreadState(PyInterpreterState* interp) {
        _ts = PyThreadState_New(interp);
        PyEval_RestoreThread(_ts);
    }

    ~ThreadState() {
        PyThreadState_Clear(_ts);
        PyThreadState_DeleteCurrent();
    }

    operator PyThreadState*() { return _ts; }

    static PyThreadState* current() { return PyThreadState_Get(); }

   private:
    PyThreadState* _ts;
};

// represent a sub interpreter
class SubInterpreter {
   public:
    // perform the necessary setup and cleanup for a new thread running using a
    // specific interpreter
    struct ThreadScope {
        ThreadState _state;
        ThreadStateSwapScope _swap{_state};

        ThreadScope(PyInterpreterState* interp) : _state(interp) {}
    };

    SubInterpreter() {
        ThreadStateRestoreScope restore;

        // https://docs.python.org/3/c-api/init.html#a-per-interpreter-gil
        PyInterpreterConfig config = {
            .check_multi_interp_extensions = 1,
            .gil = PyInterpreterConfig_OWN_GIL,
        };

        PyStatus status = Py_NewInterpreterFromConfig(&_ts, &config);
        if (PyStatus_Exception(status)) {
            throw std::runtime_error("Py_NewInterpreterFromConfig failed");
        }
    }

    ~SubInterpreter() {
        if (_ts) {
            ThreadStateSwapScope sts(_ts);

            Py_EndInterpreter(_ts);
        }
    }

    PyInterpreterState* interp() { return _ts->interp; }

    static PyInterpreterState* current() {
        return ThreadState::current()->interp;
    }

   private:
    PyThreadState* _ts;
};

}  // namespace YABTE::Utilities::Python
