import os
import platform
import random
import re
import sys
import time
import traceback
import warnings

import pytest
from _pytest.outcomes import fail
from _pytest.runner import runtestprotocol
from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution
from pkg_resources import parse_version

HAS_RESULTLOG = False
DELAY_BACKOFF_BASE_DEFAULT = 2

try:
    from _pytest.resultlog import ResultLog

    HAS_RESULTLOG = True
except ImportError:
    # We have a pytest >= 6.1
    pass


PYTEST_GTE_54 = parse_version(pytest.__version__) >= parse_version("5.4")
PYTEST_GTE_63 = parse_version(pytest.__version__) >= parse_version("6.3.0.dev")


def works_with_current_xdist():
    """Return compatibility with installed pytest-xdist version.

    When running tests in parallel using pytest-xdist < 1.20.0, the first
    report that is logged will finish and terminate the current node rather
    rerunning the test. Thus we must skip logging of intermediate results under
    these circumstances, otherwise no test is rerun.

    """
    try:
        d = get_distribution("pytest-xdist")
        return d.parsed_version >= parse_version("1.20")
    except DistributionNotFound:
        return None


# command line options
def pytest_addoption(parser):
    group = parser.getgroup(
        "rerunfailures", "re-run failing tests to eliminate flaky failures"
    )
    group._addoption(
        "--only-rerun",
        action="append",
        dest="only_rerun",
        type=str,
        default=None,
        help="If passed, only rerun errors matching the regex provided. "
        "Pass this flag multiple times to accumulate a list of regexes "
        "to match",
    )
    group._addoption(
        "--reruns",
        action="store",
        dest="reruns",
        type=int,
        default=0,
        help="number of times to re-run failed tests. defaults to 0.",
    )
    group._addoption(
        "--reruns-delay",
        action="store",
        dest="reruns_delay",
        type=float,
        default=0,
        help="add time (seconds) delay between reruns.",
    )
    group._addoption(
        "--reruns-delay-backoff",
        action="store_true",
        dest="reruns_delay_backoff",
        help="vary delay between reruns with random exponential backoff "
             "algorithm. Max delay between reruns starts from 'reruns-delay'."
    )
    group._addoption(
        "--reruns-delay-backoff-base",
        dest="reruns_delay_backoff_base",
        type=int,
        default=DELAY_BACKOFF_BASE_DEFAULT,
        help="Set the base for the exponentiation operation that computes the "
             "upper bound for the random backoff delay."
    )
    group._addoption(
        "--reruns-delay-backoff-max",
        dest="reruns_delay_backoff_max",
        type=int,
        help="The maximum value (seconds) used for the upper bound in "
             "selecting the random backoff delay."
    )


def _get_resultlog(config):
    if not HAS_RESULTLOG:
        return None
    elif PYTEST_GTE_54:
        # hack
        from _pytest.resultlog import resultlog_key

        return config._store.get(resultlog_key, default=None)
    else:
        return getattr(config, "_resultlog", None)


def _set_resultlog(config, resultlog):
    if not HAS_RESULTLOG:
        pass
    elif PYTEST_GTE_54:
        # hack
        from _pytest.resultlog import resultlog_key

        config._store[resultlog_key] = resultlog
    else:
        config._resultlog = resultlog


# making sure the options make sense
# should run before / at the beginning of pytest_cmdline_main
def check_options(config):
    val = config.getvalue
    if not val("collectonly"):
        if config.option.reruns != 0:
            if config.option.usepdb:  # a core option
                raise pytest.UsageError("--reruns incompatible with --pdb")

    resultlog = _get_resultlog(config)
    if resultlog:
        logfile = resultlog.logfile
        config.pluginmanager.unregister(resultlog)
        new_resultlog = RerunResultLog(config, logfile)
        _set_resultlog(config, new_resultlog)
        config.pluginmanager.register(new_resultlog)


def _get_marker(item):
    try:
        return item.get_closest_marker("flaky")
    except AttributeError:
        # pytest < 3.6
        return item.get_marker("flaky")


# Class to generate the backoff delay given the number of attempts.
class RerunDelay():
    def __init__(self, delay):
        self._delay = delay

        if self._delay < 0:
            self._delay = 0
            warnings.warn(
                "Delay time between re-runs cannot be < 0. Using default "
                "value: 0"
            )

    def next_rerun_delay(self, iteration_nr):
        raise NotImplementedError


# Constant delay among reruns.
class RerunDelayConst(RerunDelay):
    def next_rerun_delay(self, _):
        return self._delay


# Random exponential backoff delay.
class RerunDelayExp(RerunDelay):
    def __init__(self, factor, base, max):
        super().__init__(factor)
        self._base = base
        self._max = max

        if self._base < 0:
            self._base = DELAY_BACKOFF_BASE_DEFAULT
            warnings.warn(
                "Exponential backoff base cannot be < 0. Using default value "
                f"{DELAY_BACKOFF_BASE_DEFAULT}"
            )

        if self._max and self._max < 0:
            self._max = None
            warnings.warn(
                "Max value for exponential backoff cannot be < 0. "
                "Not enforcing it."
            )

    def next_rerun_delay(self, iteration_nr):
        upper = self._delay * (self._base**iteration_nr)
        if self._max:
            upper = min(self._max, upper)

        return random.randint(self._delay, upper)


def get_reruns_condition(item):
    rerun_marker = _get_marker(item)

    condition = True
    if rerun_marker is not None and "condition" in rerun_marker.kwargs:
        condition = evaluate_condition(
            item, rerun_marker, rerun_marker.kwargs["condition"]
        )

    return condition


def evaluate_condition(item, mark, condition: object) -> bool:
    # copy from python3.8 _pytest.skipping.py

    result = False
    # String condition.
    if isinstance(condition, str):
        globals_ = {
            "os": os,
            "sys": sys,
            "platform": platform,
            "config": item.config,
        }
        if hasattr(item, "obj"):
            globals_.update(item.obj.__globals__)  # type: ignore[attr-defined]
        try:
            filename = f"<{mark.name} condition>"
            condition_code = compile(condition, filename, "eval")
            result = eval(condition_code, globals_)
        except SyntaxError as exc:
            msglines = [
                "Error evaluating %r condition" % mark.name,
                "    " + condition,
                "    " + " " * (exc.offset or 0) + "^",
                "SyntaxError: invalid syntax",
            ]
            fail("\n".join(msglines), pytrace=False)
        except Exception as exc:
            msglines = [
                "Error evaluating %r condition" % mark.name,
                "    " + condition,
                *traceback.format_exception_only(type(exc), exc),
            ]
            fail("\n".join(msglines), pytrace=False)

    # Boolean condition.
    else:
        try:
            result = bool(condition)
        except Exception as exc:
            msglines = [
                "Error evaluating %r condition as a boolean" % mark.name,
                *traceback.format_exception_only(type(exc), exc),
            ]
            fail("\n".join(msglines), pytrace=False)
    return result


class RerunManager():
    """
    Manager for rerunning failed tests.
    """

    def __init__(self, item, reruns=None, delay=None, delay_backoff=None,
                 delay_backoff_base=None, delay_backoff_max=None):
        self._item = item

        self._reruns = reruns or 0

        delay = delay or 0
        self._delay_generator = RerunDelayConst(delay)
        if delay_backoff:
            self._delay_generator = RerunDelayExp(
                delay,
                delay_backoff_base or DELAY_BACKOFF_BASE_DEFAULT,
                delay_backoff_max
            )

        self._execution_count = 0

    @property
    def max_reruns(self):
        return self._reruns

    def match_rerun_policy(self, report):
        """
        Check if @report matches the rerun policy. Return a string with the
        reason if matching, or None otherwise
        """
        assert report.outcome == "failed"

        raise NotImplementedError()

    def next_delay(self):
        return self._delay_generator.next_rerun_delay(self._execution_count)

    def prepare_rerun_test(self, report):
        self._execution_count += 1

        report.outcome = "rerun"
        time.sleep(
            self._delay_generator.next_rerun_delay(self._execution_count - 1)
        )

        if not (hasattr(self._item.config, "workerinput") or
                hasattr(self._item.config, "slaveinput")) or \
           works_with_current_xdist():
            # will rerun test, log intermediate result
            self._item.ihook.pytest_runtest_logreport(report=report)

        # cleaning item's cashed results from any level of setups
        _remove_cached_results_from_failed_fixtures(self._item)
        _remove_failed_setup_state_from_session(self._item)


class FlakyTestRerunManager(RerunManager):
    """
    per-test rerunning setting.
    """

    def __init__(self, item):
        reruns = None
        delay = None
        delay_backoff = None
        delay_backoff_base = None
        delay_backoff_max = None
        self._rerun_marker = _get_marker(item)
        if self._rerun_marker:
            reruns = 1
            if "reruns" in self._rerun_marker.kwargs:
                # check for keyword arguments
                reruns = self._rerun_marker.kwargs["reruns"]
            elif len(self._rerun_marker.args) > 0:
                # check for arguments
                reruns = self._rerun_marker.args[0]

            if "reruns_delay" in self._rerun_marker.kwargs:
                delay = self._rerun_marker.kwargs["reruns_delay"]
            elif len(self._rerun_marker.args) > 1:
                # check for argument
                delay = self._rerun_marker.args[1]

            if "reruns_delay_backoff" in self._rerun_marker.kwargs:
                delay_backoff = self._rerun_marker.kwargs["reruns_delay_backoff"]
            elif len(self._rerun_marker.args) > 2:
                # check for argument
                delay_backoff = self._rerun_marker.args[2]

            if "reruns_delay_backoff_base" in self._rerun_marker.kwargs:
                delay_backoff_base = \
                    self._rerun_marker.kwargs["reruns_delay_backoff_base"]
            elif len(self._rerun_marker.args) > 3:
                # check for argument
                delay_backoff_base = self._rerun_marker.args[3]

            if "reruns_delay_backoff_max" in self._rerun_marker.kwargs:
                delay_backoff_max = \
                    self._rerun_marker.kwargs["reruns_delay_backoff_max"]
            elif len(self._rerun_marker.args) > 4:
                # check for argument
                delay_backoff_max = self._rerun_marker.args[4]

        super().__init__(
            item,
            reruns,
            delay,
            delay_backoff,
            delay_backoff_base,
            delay_backoff_max
        )

    def evaluate_rerun_policy(self, report):
        assert report.outcome == "failed"

        if self._execution_count + 1 > self.max_reruns:
            return False

        condition = get_reruns_condition(self._item)
        if condition:
            report.reason = ""
            return True
        return False


class GlobalRerunManager(RerunManager):
    """
    Global test rerunning policy
    """
    def __init__(self, item):
        delay = item.session.config.option.reruns_delay
        delay_backoff = item.session.config.option.reruns_delay_backoff
        delay_backoff_base = \
            item.session.config.option.reruns_delay_backoff_base
        delay_backoff_max = item.session.config.option.reruns_delay_backoff_max

        self._rerun_errors = item.session.config.option.only_rerun

        super().__init__(
            item,
            item.session.config.option.reruns,
            delay,
            delay_backoff,
            delay_backoff_base,
            delay_backoff_max
        )

    def evaluate_rerun_policy(self, report):
        assert report.outcome == "failed"

        if self._execution_count + 1 > self.max_reruns:
            return False

        if not self._rerun_errors:
            report.reason = ""
            return True

        for rerun_regex in self._rerun_errors:
            if re.search(rerun_regex, report.longrepr.reprcrash.message):
                report.reason = f"Error matching '{rerun_regex}'"
                return True
        return False


def _remove_cached_results_from_failed_fixtures(item):
    """Note: remove all cached_result attribute from every fixture."""
    cached_result = "cached_result"
    fixture_info = getattr(item, "_fixtureinfo", None)
    for fixture_def_str in getattr(fixture_info, "name2fixturedefs", ()):
        fixture_defs = fixture_info.name2fixturedefs[fixture_def_str]
        for fixture_def in fixture_defs:
            if getattr(fixture_def, cached_result, None) is not None:
                result, _, err = getattr(fixture_def, cached_result)
                if err:  # Deleting cached results for only failed fixtures
                    if PYTEST_GTE_54:
                        setattr(fixture_def, cached_result, None)
                    else:
                        delattr(fixture_def, cached_result)


def _remove_failed_setup_state_from_session(item):
    """
    Clean up setup state.

    Note: remove all failures from every node in _setupstate stack
          and clean the stack itself
    """
    setup_state = item.session._setupstate
    if PYTEST_GTE_63:
        setup_state.stack = {}
    else:
        for node in setup_state.stack:
            if hasattr(node, "_prepare_exc"):
                del node._prepare_exc
        setup_state.stack = []


def _failed_test(report):
    """
    Return whether the test has failed according to the @report. Failed means
    the test didn't pass or xfailed.
    """
    xfail = hasattr(report, "wasxfail")
    return (
        report.outcome == "failed" and
        report.failed and
        not xfail
    )


def pytest_configure(config):
    # add flaky marker
    config.addinivalue_line(
        "markers",
        "flaky(reruns=1, reruns_delay=0, reruns_delay_backoff=True, "
        "reruns_delay_backoff_base=2, reruns_delay_backoff_max=3600): mark "
        "test to re-run up to 'reruns' times. Add a delay of 'reruns_delay' "
        "seconds between re-runs. Vary delay according to random exponential "
        "backoff with 'reruns_delay_backoff' (base and max delay set with "
        "'reruns_delay_backoff_base' and 'reruns_delay_backoff_max' "
        "arguments)",
    )


def pytest_runtest_protocol(item, nextitem):
    """
    Run the test protocol.

    Note: when teardown fails, two reports are generated for the case, one for
    the test case and the other for the teardown error.
    """

    test_rerun_policy = FlakyTestRerunManager(item)
    global_rerun_policy = GlobalRerunManager(item)

    if not test_rerun_policy.max_reruns and not global_rerun_policy.max_reruns:
        # global setting is not specified, and this test is not marked with
        # flaky
        return

    # while this doesn't need to be run with every item, it will fail on the
    # first item if necessary
    check_options(item.session.config)
    item.execution_count = 0

    need_to_run = True
    while need_to_run:
        item.execution_count += 1
        item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
        reports = runtestprotocol(item, nextitem=nextitem, log=False)

        for report in reports:  # 3 reports: setup, call, teardown
            report.rerun = item.execution_count - 1
            if _failed_test(report):
                if global_rerun_policy.evaluate_rerun_policy(report):
                    global_rerun_policy.prepare_rerun_test(report)
                    break  # trigger rerun
                elif test_rerun_policy.evaluate_rerun_policy(report):
                    # Reset @global_rerun_policy execution count.
                    global_rerun_policy = GlobalRerunManager(item)
                    test_rerun_policy.prepare_rerun_test(report)
                    break  # trigger rerun

            item.ihook.pytest_runtest_logreport(report=report)
        else:
            need_to_run = False

        item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)

    return True


def pytest_report_teststatus(report):
    # Adapted from https://pytest.org/latest/_modules/_pytest/skipping.html
    if report.outcome == "rerun":
        rerun_string = "RERUN"
        if report.reason:
            rerun_string += f": {report.reason}"
        return "rerun", "R", (rerun_string, {"yellow": True})


def pytest_terminal_summary(terminalreporter):
    # Adapted from https://pytest.org/latest/_modules/_pytest/skipping.html
    tr = terminalreporter
    if not tr.reportchars:
        return

    lines = []
    for char in tr.reportchars:
        if char in "rR":
            show_rerun(terminalreporter, lines)

    if lines:
        tr._tw.sep("=", "rerun test summary info")
        for line in lines:
            tr._tw.line(line)


def show_rerun(terminalreporter, lines):
    rerun = terminalreporter.stats.get("rerun")
    if rerun:
        for rep in rerun:
            pos = rep.nodeid
            lines.append(f"RERUN {pos}")


if HAS_RESULTLOG:

    class RerunResultLog(ResultLog):
        def __init__(self, config, logfile):
            ResultLog.__init__(self, config, logfile)

        def pytest_runtest_logreport(self, report):
            """Add support for rerun report."""
            if report.when != "call" and report.passed:
                return
            res = self.config.hook.pytest_report_teststatus(report=report)
            code = res[1]
            if code == "x":
                longrepr = str(report.longrepr)
            elif code == "X":
                longrepr = ""
            elif report.passed:
                longrepr = ""
            elif report.failed:
                longrepr = str(report.longrepr)
            elif report.skipped:
                longrepr = str(report.longrepr[2])
            elif report.outcome == "rerun":
                longrepr = str(report.longrepr)
            else:
                longrepr = str(report.longrepr)

            self.log_outcome(report, code, longrepr)
