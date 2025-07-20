import nox
from dependency_groups import resolve


def get_optional_dependencies(*groups):
    pyproject = nox.project.load_toml("pyproject.toml")
    dep_groups = pyproject["project"]["optional-dependencies"]
    return resolve(dep_groups, *groups)


def get_dependencies():
    pyproject = nox.project.load_toml("pyproject.toml")
    return pyproject["project"]["dependencies"]


@nox.session(default=False, name="do-lint")
def do_lint(session):
    session.install(*get_optional_dependencies("dev"))

    session.run("black", ".")
    session.run("isort", ".", "--profile", "black")
    session.run(
        "docformatter",
        ".",
        "--recursive",
        "-i",
        "--black",
        " --exclude",
        "_unittest_numpy_extensions.py",
    )


@nox.session
def check_lint(session):
    session.install(*get_optional_dependencies("dev"))

    session.run("black", ".", "--check")
    session.run("isort", ".", "--check-only", "--profile", "black")
    session.run(
        "docformatter",
        ".",
        "--recursive",
        "--check",
        "--diff",
        "--black",
        " --exclude",
        "_unittest_numpy_extensions.py",
    )


@nox.session(python=["3.12", "3.13"])
def test(session):
    session.install("-e", ".")
    session.install(*get_dependencies())
    session.install(*get_optional_dependencies("dev", "notebooks"))
    session.run(
        "python",
        "-m",
        "unittest",
        "discover",
        "-v",
        "-s",
        "./yabte/tests/",
        "-p",
        "test_*.py",
    )

    session.run("coverage", "run", "-m", "unittest")
    session.run("coverage", "report", "-m")


@nox.session(default=False, name="build-cpp")
def do_lint(session):

    debug = session.posargs and "debug" in session.posargs

    session.chdir("cpp")

    build_type = "Debug" if debug else "Release"
    output_folder = f"build/{build_type}"

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
