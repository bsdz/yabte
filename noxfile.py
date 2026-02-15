import nox
from dependency_groups import resolve

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["do-lint", "build-cpp(debug)", "test-3.13(debug)"]

# Path to a locally built pyarrow wheel, or None to install from PyPI
pyarrow_wheel_path = None


def get_optional_dependencies(*groups):
    pyproject = nox.project.load_toml("pyproject.toml")
    dep_groups = pyproject["project"]["optional-dependencies"]
    return resolve(dep_groups, *groups)


def get_dependencies():
    pyproject = nox.project.load_toml("pyproject.toml")
    return pyproject["project"]["dependencies"]


@nox.session(name="do-lint")
def do_lint(session):
    session.install(*get_optional_dependencies("dev"))

    session.run("black", "yabte", "noxfile.py")
    session.run("isort", "yabte", "noxfile.py", "--profile", "black")
    session.run(
        "docformatter",
        "yabte",
        "noxfile.py",
        "--recursive",
        "-i",
        "--black",
        "--exclude",
        "_unittest_numpy_extensions.py",
    )


@nox.session
def check_lint(session):
    session.install(*get_optional_dependencies("dev"))

    session.run("black", "yabte", "noxfile.py", "--check")
    session.run("isort", "yabte", "noxfile.py", "--check-only", "--profile", "black")
    session.run(
        "docformatter",
        "yabte",
        "noxfile.py",
        "--recursive",
        "--check",
        "--diff",
        "--black",
        "--exclude",
        "_unittest_numpy_extensions.py",
    )


@nox.session(python=["3.13"])
@nox.parametrize("build_type", ["Debug", "Release"], ids=["debug", "release"])
def test(session, build_type):
    session.install("-e", ".")
    session.install(*get_dependencies())
    session.install(*get_optional_dependencies("dev", "notebooks"))
    if pyarrow_wheel_path:
        session.install(pyarrow_wheel_path)

    # Ensure C++ extension is found (assuming it was built with `nox -s build-cpp`)
    # We add the location of the built .so to PYTHONPATH
    env = {"PYTHONPATH": f"cpp/build/{build_type}/pybind"}

    # accept an optional argument to specify which tests to run, otherwise run all tests
    pattern = session.posargs[0] if session.posargs else "test_*.py"

    session.run(
        "python",
        "-m",
        "unittest",
        "discover",
        "-v",
        "-s",
        "./yabte/",
        "-p",
        pattern,
        env=env,
    )

    if not session.posargs:
        session.run(
            "coverage",
            "run",
            "-m",
            "unittest",
            "discover",
            "-s",
            "./yabte/",
            "-p",
            "test_*.py",
            env=env,
        )
        session.run("coverage", "report", "-m")


@nox.session(name="build-cpp")
@nox.parametrize("build_type", ["Debug", "Release"], ids=["debug", "release"])
def cpp_build(session, build_type):
    session.install("conan")
    if pyarrow_wheel_path:
        session.install(pyarrow_wheel_path)
    else:
        session.install("pyarrow")
    session.install("numpy")

    output_folder = f"build/{build_type}"

    with session.chdir("cpp"):
        session.run(
            "conan",
            "install",
            ".",
            f"--output-folder={output_folder}",
            "--build=missing",
            f"--settings=build_type={build_type}",
        )

        with session.chdir(output_folder):
            session.run(
                "cmake",
                "../..",
                f"-DCMAKE_BUILD_TYPE={build_type}",
            )

            session.run("cmake", "--build", ".", "--config", build_type)

            with session.chdir("src_test"):
                session.run("ctest", "--output-on-failure", "-C", build_type)


@nox.session(name="clean-cpp")
@nox.parametrize("build_type", ["Debug", "Release"], ids=["debug", "release"])
def cpp_clean(session, build_type):
    output_folder = f"build/{build_type}"

    with session.chdir("cpp"):
        with session.chdir(output_folder):
            session.run(
                "cmake",
                "../..",
                f"-DCMAKE_BUILD_TYPE={build_type}",
            )

            session.run(
                "cmake", "--build", ".", "--target", "clean", "--config", build_type
            )

        with session.chdir("build"):
            session.run("rm", "-rf", build_type)
