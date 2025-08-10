"""Helpers for staged tests."""

from collections.abc import Awaitable, Callable, Collection
import functools
import inspect
from typing import Any

type StageHandler = Callable[..., Awaitable[None]]
type StageFunction = Callable[..., Awaitable[dict[str, Any] | None]]
type StageWrapper = Callable[[], Awaitable[dict[str, Any] | None]]


def staged_test(
    fn: StageHandler,
    *,
    stages: tuple[StageFunction, ...],
    extra_parameters: Collection[inspect.Parameter] | None = None,
):
    """Decorator for creating reusable tests that are split into stages.

    Args:
        fn: The function to wrap. Its function signature is used to determine
            the signature of the final wrapper function, ensuring pytest
            injected fixtures still work. The function will be called with the
            same number of positional arguments as were given for `stages`, but
            each will be a `StageWrapper`.
        stages: Functions that break down a test into smaller pieces. These are
            likely pieces that can be reused in various tests.
        extra_parameters: A collection of parameters to add to the decorated
            function (ensuring the values are injected by `pytest`).


    StageFunction:
        A callable with any number of arguments just like standard `pytest` test
        cases. Its function signature isare used to determine the signature of
        the final wrapper function created by the `staged_test` decorator,
        ensuring pytest injected fixtures still work.

        These may return a dictionary. The values in the dictionary will be
        merged into those values being passed to the next stage.

    StageWrapper:
        A callable with no arguments to be invoked & awaited within the wrapped
        function. This wrapper invokes the original stage function & handles
        merging new values into the `kwargs` of the next stage.


    Example:
        def _steup(my_fixture, **kwargs): return {"action": "light.turn_on"}
        def _actions(my_fixture, action, **kwargs): ...
        def _assertions(my_fixture, action, **kwargs): ...

        @staged_test(stages=(_setup, _actions, _assertions))
        async def test_something(
            setup,
            actions,
            assertions,
            hass: HomeAssistant,  # pytest injectected fixtures still work
            **kwargs: Any,
        ):
            await setup()
            await actions()
            await assertions()
    """

    @functools.wraps(fn)
    async def wrapper(**kwargs: Any) -> dict[str, Any]:
        extra_kwargs: dict[str, Any] = {}

        # this small helper function is used to ensure that the wrappers are
        # created with `stage_fn` bound to a value (which would not be the case
        # if `stage_wrapper` were defined within a `for` loop).
        def make_stage_wrapper(stage_fn):
            @functools.wraps(stage_fn)
            async def stage_wrapper():
                extra_kwargs.update(result := await stage_fn(**(kwargs | extra_kwargs)))
                return result

            return stage_wrapper

        # call the wrapped function with each of the stage wrapper functions
        # that will update the `extra_kwargs` with the return value of the stage
        # function.
        await fn(*[make_stage_wrapper(stage_fn) for stage_fn in stages], **kwargs)

        return kwargs | extra_kwargs

    # start creating the final wrapper function signature by iterating through
    # the parameters on the wrapped function, `fn`, but ignore the first
    # positional arguments that correspond to each individual stage.
    fn_fixture_parameters = []
    fn_callback_parameters_to_remove = len(stages)

    for param in inspect.signature(fn).parameters.values():
        if fn_callback_parameters_to_remove and param.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            fn_callback_parameters_to_remove -= 1
            continue

        fn_fixture_parameters.append(param)

    # concatenate all possible parameters, ready to create a final signature
    wrapper_parameters = [
        *[
            parameter
            for stage_fn in stages
            for parameter in inspect.signature(stage_fn).parameters.values()
        ],
        *fn_fixture_parameters,
        *(extra_parameters or []),
    ]

    # remove duplicate param names
    wrapper_parameters = [
        *{(param.name): param for param in wrapper_parameters}.values()
    ]

    # create the final wrapper signature
    wrapper_signature = inspect.Signature()
    wrapper_signature = wrapper_signature.replace(
        parameters=[
            param
            for param in sorted(wrapper_parameters, key=lambda param: param.kind)
            if not param.name.startswith("_")
            and param.kind
            not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
        ],
    )
    wrapper.__signature__ = wrapper_signature  # type: ignore[attr-defined]

    return wrapper
