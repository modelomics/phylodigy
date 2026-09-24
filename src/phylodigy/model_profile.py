"""Dynamic PyTorch frontend for graph-derived architectural genomes.

The extractor emits only an executable operator/layer graph and characters
generated from that graph. It does not inspect architecture names, classify
module families, translate configuration keys, or turn reflected labels into
structural evidence.
Static FX call arguments and declared tensor-resource relationships are
retained as structural attributes; incidental Python fields and parameter
values are never structural evidence.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import hashlib
import importlib
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .computation_graph import GraphProfile, build_graph_profile
from .schema import ArchitecturalGenome


EXTRACTOR_VERSION = "3"

__all__ = [
    "GraphExtractionError",
    "TorchUnavailableError",
    "canonical_profile_json",
    "extract_architectural_genome",
    "extract_model_profile",
    "extract_model_profile_from_factory",
    "profile_digest",
    "torch_available",
]


class TorchUnavailableError(RuntimeError):
    """Raised when PyTorch graph extraction is requested without PyTorch."""


class GraphExtractionError(RuntimeError):
    """Raised when no executable graph can be obtained from a model."""


def torch_available() -> bool:
    """Return whether PyTorch can be imported without importing it eagerly."""

    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TorchUnavailableError(
            "PyTorch is required for the torch.fx graph frontend; install "
            "'phylodigy[model]' or use a different graph frontend."
        ) from exc


def extract_architectural_genome(
    model: Any | Callable[..., Any] | None = None,
    *,
    model_factory: Callable[..., Any] | None = None,
    factory_args: Sequence[Any] = (),
    factory_kwargs: Mapping[str, Any] | None = None,
    example_inputs: Any | None = None,
    example_kwargs: Mapping[str, Any] | None = None,
    propagate_shapes: bool = True,
    allow_real_shape_propagation: bool = False,
    model_id: str | None = None,
    artifact_id: str | None = None,
    name: str = "",
    release_date: str | None = None,
    date_min: str | None = None,
    date_max: str | None = None,
    radii: Sequence[int] = (0, 1, 2, 3),
) -> ArchitecturalGenome:
    """Trace ``model`` and return its graph-derived architectural genome.

    A failed trace is an error, not an invitation to substitute configuration
    guesses. One symbolic FX graph covers only paths visible to the tracer;
    example inputs specialize shape propagation, not tracing. Capture boundaries
    and resource-shape completeness are recorded explicitly.
    """

    torch = _require_torch()
    factory_kwargs = dict(factory_kwargs or {})
    example_kwargs = dict(example_kwargs or {})
    if model_factory is not None and model is not None:
        raise TypeError("pass either model or model_factory, not both")
    if model_factory is None and callable(model) and not isinstance(model, torch.nn.Module):
        model_factory = model
        model = None
    if model_factory is not None:
        model = _construct_on_meta(torch, model_factory, tuple(factory_args), factory_kwargs)
    if model is None:
        raise TypeError("a torch.nn.Module or model factory is required")
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"expected torch.nn.Module, received {_qualified_class_name(model)}")
    if model_id and artifact_id and model_id != artifact_id:
        raise ValueError("model_id and artifact_id must match when both are supplied")
    graph_profile, trace_metadata = _extract_fx_graph(
        torch,
        model,
        example_inputs=example_inputs,
        example_kwargs=example_kwargs,
        propagate_shapes=propagate_shapes,
        allow_real_shape_propagation=allow_real_shape_propagation,
        radii=radii,
    )
    identifier = (
        artifact_id
        or model_id
        or f"model:graph:{graph_profile.graph.structural_digest}"
    )
    resource_coverage = trace_metadata["resource_capture"]
    metadata = {
        "coverage": {
            "buffer_shapes_captured": resource_coverage[
                "all_declared_buffer_shapes_captured"
            ],
            "control_flow": "symbolically_visible_single_graph",
            "example_inputs_specialize_trace": False,
            "module_boundary_policy": trace_metadata["tracing"][
                "module_boundary_policy"
            ],
            "training_process_captured": False,
            "weight_shapes_captured": resource_coverage[
                "all_declared_parameter_shapes_captured"
            ],
            "weight_values_captured": False,
        },
        "factory_meta_requested": model_factory is not None,
        "trace": trace_metadata,
    }
    return ArchitecturalGenome(
        artifact_id=identifier,
        graph_profile=graph_profile,
        artifact_kind="model",
        name=name,
        release_date=release_date,
        date_min=date_min,
        date_max=date_max,
        extractor_name="phylodigy.model_profile",
        extractor_version=EXTRACTOR_VERSION,
        metadata=metadata,
    )


def extract_model_profile(*args: Any, **kwargs: Any) -> ArchitecturalGenome:
    """Compatibility spelling for :func:`extract_architectural_genome`."""

    # The removed ``use_fx`` switch previously allowed a non-graph result. That
    # state is no longer representable because the graph is the genome.
    if kwargs.pop("use_fx", True) is False:
        raise ValueError("graph extraction cannot be disabled for an architectural genome")
    kwargs.pop("include_parameter_shapes", None)
    return extract_architectural_genome(*args, **kwargs)


def extract_model_profile_from_factory(
    factory: Callable[..., Any],
    *factory_args: Any,
    **kwargs: Any,
) -> ArchitecturalGenome:
    """Construct a model on the meta device and trace its graph."""

    factory_kwargs = kwargs.pop("factory_kwargs", {})
    return extract_architectural_genome(
        model_factory=factory,
        factory_args=factory_args,
        factory_kwargs=factory_kwargs,
        **kwargs,
    )


def _extract_fx_graph(
    torch: Any,
    model: Any,
    *,
    example_inputs: Any | None,
    example_kwargs: Mapping[str, Any],
    propagate_shapes: bool,
    allow_real_shape_propagation: bool,
    radii: Sequence[int],
) -> tuple[GraphProfile, dict[str, Any]]:
    args = _normalize_example_inputs(example_inputs)
    frontend = "torch.fx"
    try:
        symbolic_trace = importlib.import_module("torch.fx").symbolic_trace
        graph_module = symbolic_trace(model)
    except Exception as fx_exc:
        if not args and not example_kwargs:
            raise GraphExtractionError(
                "torch.fx could not produce an executable graph and torch.export "
                "requires example inputs: "
                f"{type(fx_exc).__name__}: {fx_exc}"
            ) from fx_exc
        try:
            export_args = args
            export_kwargs: Mapping[str, Any] = example_kwargs
            if _module_is_meta(model):
                export_args = _tree_map(args, lambda value: _to_meta(torch, value))
                export_kwargs = _tree_map(
                    example_kwargs, lambda value: _to_meta(torch, value)
                )
            exported = torch.export.export(
                model,
                tuple(export_args),
                dict(export_kwargs),
                strict=False,
            )
            graph_module = exported.module()
            frontend = "torch.export"
        except Exception as export_exc:
            raise GraphExtractionError(
                "neither torch.fx nor torch.export could produce an executable "
                "graph; torch.fx failed with "
                f"{type(fx_exc).__name__}: {fx_exc}; torch.export failed with "
                f"{type(export_exc).__name__}: {export_exc}"
            ) from export_exc

    shape_status = "not_requested"
    if (
        frontend == "torch.fx"
        and propagate_shapes
        and (example_inputs is not None or example_kwargs)
    ):
        try:
            ordered_args = _ordered_forward_arguments(model, args, example_kwargs)
            if _module_is_meta(model):
                meta_args = _tree_map(ordered_args, lambda value: _to_meta(torch, value))
                shape_prop = importlib.import_module("torch.fx.passes.shape_prop").ShapeProp
                shape_prop(graph_module).propagate(*meta_args)
                shape_status = "meta"
            elif _fake_shape_propagate(torch, graph_module, ordered_args):
                shape_status = "fake"
            elif allow_real_shape_propagation:
                shape_prop = importlib.import_module("torch.fx.passes.shape_prop").ShapeProp
                shape_prop(graph_module).propagate(*ordered_args)
                shape_status = "real_opt_in"
            else:
                shape_status = "skipped_non_meta_model"
        except Exception as exc:
            shape_status = f"failed:{type(exc).__name__}"

    fx_nodes = tuple(graph_module.graph.nodes)
    records: list[dict[str, Any]] = []
    records_by_name: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(fx_nodes):
        edges = _fx_input_edges(node)
        target_type = _fx_target_type(graph_module, node)
        record: dict[str, Any] = {
            "index": index,
            "input_edges": edges,
            "inputs": [edge["source"] for edge in edges],
            "name": str(node.name),
            "op": str(node.op),
            "target": target_type or _fx_target(node.target),
        }
        if target_type is not None:
            record["target_type"] = target_type
        attributes = _fx_node_attributes(graph_module, node)
        if attributes:
            record["attributes"] = attributes
        observations = _fx_node_observations(graph_module, node)
        if observations:
            record["observations"] = observations
        tensor_meta = _node_tensor_metadata(node)
        if tensor_meta is not None:
            record["tensor_meta"] = tensor_meta
        records.append(record)
        records_by_name[str(node.name)] = record

    resource_records, resource_edges, resource_capture = _fx_resource_records(
        model,
        graph_module,
        fx_nodes,
        occupied_names=set(records_by_name),
        first_index=len(records),
    )
    for consumer_name, edge in resource_edges:
        record = records_by_name[consumer_name]
        record["input_edges"].append(edge)
        record["inputs"].append(edge["source"])
    records.extend(resource_records)

    probe_signature = {
        "args": _describe_value_tree(args),
        "kwargs": _describe_value_tree(example_kwargs),
    }
    metadata = {
        "frontend": frontend,
        "frontend_version": str(getattr(torch, "__version__", "unknown")),
        "probe_digest": profile_digest(probe_signature),
        "probe_signature": probe_signature,
        "resource_capture": resource_capture,
        "shape_propagation": shape_status,
        "trace_node_count": len(fx_nodes),
        "structural_resource_node_count": len(resource_records),
        "tracing": {
            "call_module_boundaries": (
                "atomic" if frontend == "torch.fx" else "lowered_to_aten"
            ),
            "leaf_policy": (
                "torch.fx default"
                if frontend == "torch.fx"
                else "torch.export strict_false"
            ),
            "module_boundary_policy": (
                "torch_fx_default_leaf_modules_are_atomic"
                if frontend == "torch.fx"
                else "torch_export_lowers_modules_to_aten_operators"
            ),
            "root_module": "traced_through",
        },
    }
    try:
        profile = build_graph_profile(
            records,
            frontend=frontend,
            metadata=metadata,
            radii=radii,
        )
    except (TypeError, ValueError) as exc:
        raise GraphExtractionError(
            f"FX records could not be normalized into a computation graph: {exc}"
        ) from exc
    return profile, metadata


def _construct_on_meta(
    torch: Any,
    factory: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    device = getattr(torch, "device", None)
    context = device("meta") if callable(device) else contextlib.nullcontext()
    try:
        with context:
            return factory(*args, **dict(kwargs))
    except Exception as exc:
        raise GraphExtractionError(
            f"model factory could not construct on the meta device: {type(exc).__name__}: {exc}"
        ) from exc


def _normalize_example_inputs(example_inputs: Any | None) -> tuple[Any, ...]:
    if example_inputs is None:
        return ()
    return example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)


def _ordered_forward_arguments(
    model: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[Any, ...]:
    if not kwargs:
        return args
    signature = inspect.signature(model.forward)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    ordered: list[Any] = []
    for name, parameter in signature.parameters.items():
        value = bound.arguments.get(name)
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            ordered.extend(value or ())
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            for key in sorted(value or {}):
                ordered.append(value[key])
        elif name in bound.arguments:
            ordered.append(value)
    return tuple(ordered)


def _fake_shape_propagate(torch: Any, graph_module: Any, args: tuple[Any, ...]) -> bool:
    try:
        fake_module = importlib.import_module("torch._subclasses.fake_tensor")
        prop_module = importlib.import_module("torch.fx.passes.fake_tensor_prop")
        mode = fake_module.FakeTensorMode(allow_non_fake_inputs=True)

        def to_fake(value: Any) -> Any:
            return mode.from_tensor(value) if torch.is_tensor(value) else value

        prop_module.FakeTensorProp(graph_module, mode=mode).propagate(
            *_tree_map(args, to_fake)
        )
        return True
    except (ImportError, AttributeError):
        return False


def _module_is_meta(module: Any) -> bool:
    try:
        tensors = tuple(module.parameters()) + tuple(module.buffers())
    except Exception:
        return False
    return bool(tensors) and all(bool(getattr(tensor, "is_meta", False)) for tensor in tensors)


def _to_meta(torch: Any, value: Any) -> Any:
    if not torch.is_tensor(value) or bool(getattr(value, "is_meta", False)):
        return value
    shape = tuple(int(size) for size in value.shape)
    try:
        stride = tuple(int(size) for size in value.stride())
        return torch.empty_strided(
            shape,
            stride,
            dtype=value.dtype,
            device="meta",
            requires_grad=bool(value.requires_grad),
        )
    except Exception:
        return torch.empty(shape, dtype=value.dtype, device="meta")


def _node_tensor_metadata(node: Any) -> Any | None:
    metadata = getattr(node, "meta", {}) or {}
    tensor_meta = metadata.get("tensor_meta")
    if tensor_meta is not None:
        return _tensor_metadata_value(tensor_meta)
    value = metadata.get("val")
    return _describe_tensor_tree(value) if value is not None else None


def _tensor_metadata_value(meta: Any) -> Any:
    if isinstance(meta, (list, tuple)) and not hasattr(meta, "shape"):
        return [_tensor_metadata_value(item) for item in meta]
    result: dict[str, Any] = {}
    for name in ("shape", "dtype", "requires_grad", "stride", "memory_format", "is_quantized"):
        value = getattr(meta, name, None)
        if value is not None:
            result[name] = _json_value(value)
    return result or _json_value(meta)


def _describe_tensor_tree(value: Any) -> Any | None:
    if _looks_like_tensor(value):
        return _tensor_signature(value)
    if isinstance(value, (tuple, list)):
        return [_describe_tensor_tree(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _describe_tensor_tree(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return None


def _describe_value_tree(value: Any) -> Any:
    if _looks_like_tensor(value):
        return _tensor_signature(value)
    if isinstance(value, Mapping):
        return {
            str(key): _describe_value_tree(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_describe_value_tree(item) for item in value]
    return _json_value(value)


def _tensor_signature(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dtype": _dtype_name(getattr(value, "dtype", None)),
        "shape": _safe_shape(value),
    }
    requires_grad = getattr(value, "requires_grad", None)
    if requires_grad is not None:
        result["requires_grad"] = bool(requires_grad)
    return result


def _resource_tensor_signature(value: Any) -> dict[str, Any]:
    """Describe declared tensor structure without training state or values."""

    return {
        "dtype": _dtype_name(getattr(value, "dtype", None)),
        "shape": _safe_shape(value),
    }


def _named_tensor_values(
    module: Any,
    method_name: str,
    *,
    recurse: bool,
) -> tuple[tuple[Any, ...], bool]:
    """Return declared tensors and whether the framework enumeration succeeded."""

    method = getattr(module, method_name, None)
    if not callable(method):
        return (), False
    try:
        try:
            named = tuple(method(recurse=recurse, remove_duplicate=False))
        except TypeError:
            try:
                named = tuple(method(recurse=recurse))
            except TypeError:
                named = tuple(method())
    except Exception:
        return (), False
    values = tuple(value for _, value in named if _looks_like_tensor(value))
    return values, True


def _unique_objects(values: Sequence[Any]) -> tuple[Any, ...]:
    """Deduplicate live objects by identity without serializing their IDs."""

    result: list[Any] = []
    seen: dict[int, Any] = {}
    for value in values:
        token = id(value)
        previous = seen.get(token)
        if previous is value:
            continue
        seen[token] = value
        result.append(value)
    return tuple(result)


def _fx_resource_records(
    model: Any,
    graph_module: Any,
    nodes: Sequence[Any],
    *,
    occupied_names: set[str],
    first_index: int,
) -> tuple[
    list[dict[str, Any]],
    list[tuple[str, dict[str, str]]],
    dict[str, Any],
]:
    """Create anonymous resource nodes and their incidence relationships.

    Python object identity is used only to discover aliases during this one
    extraction.  Serialized records contain resource kind, shape/dtype, and
    graph incidence, never object IDs, attribute paths, parameter names, or
    values.
    """

    declared_parameters, parameters_enumerated = _named_tensor_values(
        model, "named_parameters", recurse=True
    )
    declared_buffers, buffers_enumerated = _named_tensor_values(
        model, "named_buffers", recurse=True
    )
    declared_parameters = _unique_objects(declared_parameters)
    declared_buffers = _unique_objects(declared_buffers)

    graph_parameters, _ = _named_tensor_values(
        graph_module, "named_parameters", recurse=True
    )
    graph_buffers, _ = _named_tensor_values(
        graph_module, "named_buffers", recurse=True
    )
    parameter_ids = {id(value) for value in declared_parameters + graph_parameters}
    buffer_ids = {id(value) for value in declared_buffers + graph_buffers}

    resources: dict[tuple[str, int], dict[str, Any]] = {}
    call_module_count = 0
    unresolved_call_modules = 0
    recursive_enumeration_complete = True

    def add_resource(
        value: Any,
        role: str,
        consumer_name: str,
    ) -> None:
        family = "module" if role == "module" else "tensor"
        key = (family, id(value))
        resource = resources.get(key)
        if resource is not None and resource["value"] is not value:
            # Values are retained below, so a live-object ID collision is only
            # defensive. Keep the records distinct without exposing the ID.
            suffix = 1
            while (f"{family}:{suffix}", id(value)) in resources:
                suffix += 1
            key = (f"{family}:{suffix}", id(value))
            resource = None
        if resource is None:
            resource = {
                "consumers": set(),
                "roles": set(),
                "value": value,
            }
            resources[key] = resource
        resource["roles"].add(role)
        resource["consumers"].add((consumer_name, role))

    for node in nodes:
        operation = str(node.op)
        consumer_name = str(node.name)
        if operation == "call_module":
            call_module_count += 1
            target = str(node.target)
            module = _resolve_attribute(model, target)
            if module is None:
                module = _fx_target_module(graph_module, node)
            if module is None:
                unresolved_call_modules += 1
                recursive_enumeration_complete = False
                continue
            add_resource(module, "module", consumer_name)
            parameters, parameter_ok = _named_tensor_values(
                module, "named_parameters", recurse=True
            )
            buffers, buffer_ok = _named_tensor_values(
                module, "named_buffers", recurse=True
            )
            recursive_enumeration_complete &= parameter_ok and buffer_ok
            for tensor in _unique_objects(parameters):
                add_resource(tensor, "parameter", consumer_name)
            for tensor in _unique_objects(buffers):
                add_resource(tensor, "buffer", consumer_name)
        elif operation == "get_attr":
            target = str(node.target)
            value = _resolve_attribute(model, target)
            if value is None:
                value = _resolve_attribute(graph_module, target)
            if not _looks_like_tensor(value):
                continue
            token = id(value)
            if token in parameter_ids:
                role = "parameter"
            elif token in buffer_ids:
                role = "buffer"
            else:
                role = "tensor"
            add_resource(value, role, consumer_name)

    resource_records: list[dict[str, Any]] = []
    resource_edges: list[tuple[str, dict[str, str]]] = []
    used_names = set(occupied_names)
    captured_parameter_ids: set[int] = set()
    captured_buffer_ids: set[int] = set()
    role_counts = {"buffer": 0, "module": 0, "parameter": 0, "tensor": 0}

    for resource_index, resource in enumerate(resources.values()):
        raw_name = f"__phylodigy_resource_{resource_index:06d}"
        while raw_name in used_names:
            raw_name = "_" + raw_name
        used_names.add(raw_name)
        roles = tuple(sorted(resource["roles"]))
        value = resource["value"]
        if roles == ("module",):
            resource_kind = "module"
            attributes: dict[str, Any] = {"resource_roles": ["module"]}
        else:
            resource_kind = roles[0] if len(roles) == 1 else "tensor"
            attributes = {
                "resource_roles": list(roles),
                "tensor_signature": _resource_tensor_signature(value),
            }
            if "parameter" in roles:
                captured_parameter_ids.add(id(value))
            if "buffer" in roles:
                captured_buffer_ids.add(id(value))
        for role in roles:
            role_counts[role] += 1
        resource_records.append(
            {
                "attributes": attributes,
                "index": first_index + resource_index,
                "input_edges": [],
                "inputs": [],
                "name": raw_name,
                "op": "resource",
                "target": resource_kind,
            }
        )
        for consumer_name, role in sorted(resource["consumers"]):
            resource_edges.append(
                (
                    consumer_name,
                    {"position": f"resource:{role}", "source": raw_name},
                )
            )

    declared_parameter_ids = {id(value) for value in declared_parameters}
    declared_buffer_ids = {id(value) for value in declared_buffers}
    captured_declared_parameters = declared_parameter_ids & captured_parameter_ids
    captured_declared_buffers = declared_buffer_ids & captured_buffer_ids
    resource_capture = {
        "alias_detection": "in_process_object_identity_not_serialized",
        "all_declared_buffer_shapes_captured": (
            buffers_enumerated
            and recursive_enumeration_complete
            and declared_buffer_ids <= captured_buffer_ids
        ),
        "all_declared_parameter_shapes_captured": (
            parameters_enumerated
            and recursive_enumeration_complete
            and declared_parameter_ids <= captured_parameter_ids
        ),
        "atomic_call_module_node_count": call_module_count,
        "captured_declared_buffer_count": len(captured_declared_buffers),
        "captured_declared_parameter_count": len(captured_declared_parameters),
        "declared_buffer_count": len(declared_buffer_ids),
        "declared_parameter_count": len(declared_parameter_ids),
        "resource_node_counts": dict(sorted(role_counts.items())),
        "resource_relationship_count": len(resource_edges),
        "scope": "executed_fx_nodes_and_recursive_resources_of_atomic_modules",
        "serialized_identity": "anonymous_resource_graph_incidence",
        "unresolved_call_module_node_count": unresolved_call_modules,
    }
    return resource_records, resource_edges, resource_capture


def _fx_target(target: Any) -> str:
    if isinstance(target, str):
        return target
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return _qualified_class_name(target)


def _fx_target_module(graph_module: Any, node: Any) -> Any | None:
    if str(node.op) != "call_module":
        return None
    try:
        return graph_module.get_submodule(str(node.target))
    except (AttributeError, KeyError, TypeError):
        return None


def _fx_target_type(graph_module: Any, node: Any) -> str | None:
    module = _fx_target_module(graph_module, node)
    return _qualified_class_name(module) if module is not None else None


def _fx_node_attributes(graph_module: Any, node: Any) -> dict[str, Any]:
    """Return only values proven to participate in the captured FX call."""

    result: dict[str, Any] = {}
    arguments = _fx_static_arguments(node)
    if arguments:
        result["arguments"] = arguments
    if str(node.op) == "get_attr":
        attribute = _resolve_attribute(graph_module, str(node.target))
        # Tensor structure and aliasing are represented exactly once by the
        # anonymous resource node connected to this access. Non-tensor constants
        # remain ordinary execution-relevant node attributes.
        if attribute is not None and not _looks_like_tensor(attribute):
            result["attribute_signature"] = _describe_value_tree(attribute)
    return result


def _fx_node_observations(graph_module: Any, node: Any) -> dict[str, Any]:
    """Retain incidental leaf-module state outside structural evidence."""

    module = _fx_target_module(graph_module, node)
    if module is None:
        return {}
    public_state = _fx_public_module_observations(module)
    return {"module_public_state": public_state} if public_state else {}


def _resolve_attribute(root: Any, path: str) -> Any | None:
    value = root
    for token in path.split("."):
        value = getattr(value, token, None)
        if value is None:
            return None
    return value


def _fx_static_arguments(node: Any) -> dict[str, Any]:
    node_type = type(node)
    result: dict[str, Any] = {}

    def visit(value: Any, position: str) -> None:
        if isinstance(value, node_type):
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{position}[{index}]")
            return
        if isinstance(value, Mapping):
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                visit(item, f"{position}[{str(key)!r}]")
            return
        if isinstance(value, slice):
            visit(value.start, f"{position}.start")
            visit(value.stop, f"{position}.stop")
            visit(value.step, f"{position}.step")
            return
        try:
            result[position] = _json_value(value)
        except (TypeError, ValueError):
            pass

    visit(node.args, "args")
    visit(node.kwargs, "kwargs")
    return dict(sorted(result.items()))


def _fx_public_module_observations(module: Any) -> dict[str, Any]:
    """Describe public Python state as non-structural trace observations."""

    result: dict[str, Any] = {}
    try:
        public_items = vars(module).items()
    except TypeError:
        public_items = ()
    for name, value in sorted(public_items, key=lambda pair: str(pair[0])):
        key = str(name)
        if key.startswith("_") or key == "training" or callable(value):
            continue
        try:
            result[key] = _describe_value_tree(value)
        except (TypeError, ValueError):
            continue
    return dict(sorted(result.items()))


def _fx_input_edges(node: Any) -> list[dict[str, str]]:
    node_type = type(node)
    result: list[dict[str, str]] = []

    def visit(value: Any, position: str) -> None:
        if isinstance(value, node_type):
            result.append({"position": position, "source": str(value.name)})
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{position}[{index}]")
            return
        if isinstance(value, Mapping):
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                visit(item, f"{position}[{str(key)!r}]")
            return
        if isinstance(value, slice):
            visit(value.start, f"{position}.start")
            visit(value.stop, f"{position}.stop")
            visit(value.step, f"{position}.step")

    visit(node.args, "args")
    visit(node.kwargs, "kwargs")
    return result


def _tree_map(value: Any, function: Callable[[Any], Any]) -> Any:
    if isinstance(value, tuple):
        return tuple(_tree_map(item, function) for item in value)
    if isinstance(value, list):
        return [_tree_map(item, function) for item in value]
    if isinstance(value, Mapping):
        return {key: _tree_map(item, function) for key, item in value.items()}
    return function(value)


def _qualified_class_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _safe_shape(value: Any) -> list[int] | None:
    try:
        return [int(item) for item in value.shape]
    except (AttributeError, TypeError, ValueError):
        return None


def _dtype_name(dtype: Any) -> str | None:
    if dtype is None:
        return None
    return str(dtype).removeprefix("torch.")


def _looks_like_tensor(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if _looks_like_tensor(value):
        return _tensor_signature(value)
    if isinstance(value, enum.Enum):
        return _json_value(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_json_value(item) for item in value]
        return sorted(items, key=canonical_json)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist") and callable(value.tolist):
        converted = value.tolist()
        # Non-tensor operator attributes may be small arrays. Their values are
        # part of the operator configuration rather than learned checkpoints.
        return _json_value(converted)
    text = str(value)
    if " at 0x" in text:
        raise TypeError("process-local object representations are not canonical")
    return text


def canonical_profile_json(profile: Any) -> str:
    """Return deterministic JSON for a genome or JSON-compatible value."""

    if hasattr(profile, "to_dict") and callable(profile.to_dict):
        profile = profile.to_dict()
    return canonical_json(profile)


def _profile_digest_content(profile: Any) -> Any:
    """Return artifact content without its own redundant digest envelope."""

    method = getattr(profile, "to_dict", None)
    if callable(method):
        try:
            parameters = inspect.signature(method).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        accepts_include_digest = any(
            parameter.name == "include_digest"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        profile = (
            method(include_digest=False) if accepts_include_digest else method()
        )
    if isinstance(profile, Mapping) and "artifact_type" in profile and "digest" in profile:
        profile = dict(profile)
        profile.pop("digest", None)
    return profile


def profile_digest(profile: Any, *, algorithm: str = "sha256") -> str:
    """Hash canonical artifact content without double-hashing its digest field."""

    payload = canonical_json(_profile_digest_content(profile)).encode("utf-8")
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"unsupported digest algorithm: {algorithm!r}") from exc
    digest.update(payload)
    return digest.hexdigest()
