import nox
from dependency_groups import resolve

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["do-lint", "test-3.13(debug)"]


def get_optional_dependencies(*groups):
    pyproject = nox.project.load_toml("pyproject.toml")
    dep_groups = pyproject["project"]["optional-dependencies"]
    return resolve(dep_groups, *groups)


def get_dependencies():
    pyproject = nox.project.load_toml("pyproject.toml")
    return pyproject["project"]["dependencies"]


def build_cpp_impl(session, build_type):
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


def clean_cpp_impl(session, build_type):
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

    # We need to build the C++ extension first
    # This requires conan, pyarrow and numpy
    session.install("conan", "pyarrow", "numpy")
    build_cpp_impl(session, build_type)

    # Ensure C++ extension is found (assuming it was built with `nox -s build-cpp`)
    # We add the location of the built .so to PYTHONPATH
    env = {"PYTHONPATH": f"cpp/build/{build_type}/pybind"}

    # Filter out "debug" from args to find the test pattern
    args = [arg for arg in session.posargs if arg != "debug"]
    pattern = "test_*.py"
    if args:
        pattern = args[0]

    session.run(
        "python",
        "-m",
        "unittest",
        "discover",
        "-v",
        "-s",
        "./yabte/tests/",
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
            "./yabte/tests",
            "-p",
            "test_*.py",
            env=env,
        )
        session.run("coverage", "report", "-m")


@nox.session(name="build-cpp")
@nox.parametrize("build_type", ["Debug", "Release"], ids=["debug", "release"])
def cpp_build(session, build_type):
    session.install("conan")
    session.install("pyarrow")
    session.install("numpy")
    build_cpp_impl(session, build_type)


@nox.session(name="clean-cpp")
@nox.parametrize("build_type", ["Debug", "Release"], ids=["debug", "release"])
def cpp_clean(session, build_type):
    clean_cpp_impl(session, build_type)
