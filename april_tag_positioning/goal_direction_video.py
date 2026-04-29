import threading

import numpy as np

try:
    from dash import Dash, Input, Output, dcc, html
    from flask import Response
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - depends on local environment
    Dash = None
    Input = None
    Output = None
    dcc = None
    html = None
    Response = None
    go = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class GoalDirectionWebVisualizer:
    def __init__(
        self,
        goal_point,
        tag_origin=(0.0, 0.0, 0.0),
        host="0.0.0.0",
        port=8050,
        refresh_ms=100,
        max_render_points=1500,
    ):
        self.goal_point = np.asarray(goal_point, dtype=float)
        self.tag_origin = np.asarray(tag_origin, dtype=float)
        self.host = host
        self.port = port
        self.refresh_ms = int(refresh_ms)
        self.max_render_points = int(max_render_points)

        self._lock = threading.Lock()
        self._camera_condition = threading.Condition(self._lock)
        self._positions = []
        self._latest_position = None
        self._latest_direction = None
        self._camera_topic_available = False
        self._camera_has_frame = False
        self._camera_frame_bytes = None
        self._camera_frame_index = 0
        self._server_thread = None
        self._app = None

    def start(self):
        _ensure_dependencies()

        if self._server_thread is not None and self._server_thread.is_alive():
            return

        self._app = self._build_app()
        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="goal-direction-dashboard",
        )
        self._server_thread.start()

    def update(self, drone_position):
        position = np.asarray(drone_position, dtype=float)
        direction = self.goal_point - position

        with self._lock:
            self._positions.append(position.copy())
            self._latest_position = position.copy()
            self._latest_direction = direction.copy()

    def set_camera_topic_available(self, available):
        with self._camera_condition:
            self._camera_topic_available = bool(available)
            if not self._camera_topic_available:
                self._camera_has_frame = False
                self._camera_frame_bytes = None
            self._camera_condition.notify_all()

    def update_camera_image(self, image_bytes):
        with self._camera_condition:
            self._camera_topic_available = True
            self._camera_has_frame = True
            self._camera_frame_bytes = bytes(image_bytes)
            self._camera_frame_index += 1
            self._camera_condition.notify_all()

    def get_local_url(self):
        return f"http://127.0.0.1:{self.port}"

    def _build_app(self):
        app = Dash(__name__)
        app.title = "Drone Goal Viewer"

        @app.server.route("/camera_stream.mjpg")
        def camera_stream():
            return self._camera_stream_response()

        app.layout = html.Div(
            [
                html.H2("Drone, Tag e Goal em Tempo Real"),
                html.Div(
                    (
                        "Atualize a pagina livremente. O grafico segue o "
                        "estado mais recente recebido pelo ROS2."
                    ),
                    style={"marginBottom": "12px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(id="status-panel"),
                                dcc.Graph(
                                    id="goal-direction-graph",
                                    style={"height": "82vh"},
                                    config={"displaylogo": False},
                                ),
                            ],
                            style={"flex": "2 1 700px", "minWidth": "480px"},
                        ),
                        html.Div(
                            id="camera-panel",
                            style={"display": "none"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "16px",
                        "alignItems": "flex-start",
                        "flexWrap": "wrap",
                    },
                ),
                dcc.Interval(
                    id="graph-refresh",
                    interval=self.refresh_ms,
                    n_intervals=0,
                ),
            ],
            style={"padding": "16px", "fontFamily": "Arial, sans-serif"},
        )

        @app.callback(
            Output("goal-direction-graph", "figure"),
            Output("status-panel", "children"),
            Output("camera-panel", "children"),
            Output("camera-panel", "style"),
            Input("graph-refresh", "n_intervals"),
        )
        def refresh_graph(_):
            snapshot = self._snapshot()
            figure = build_goal_direction_figure(
                positions=snapshot["positions"],
                goal_point=self.goal_point,
                tag_origin=self.tag_origin,
                max_render_points=self.max_render_points,
            )
            status = build_status_panel(
                latest_position=snapshot["latest_position"],
                latest_direction=snapshot["latest_direction"],
                num_samples=snapshot["num_samples"],
            )
            camera_panel, camera_panel_style = build_camera_panel(
                camera_topic_available=snapshot["camera_topic_available"],
                camera_has_frame=snapshot["camera_has_frame"],
            )
            return figure, status, camera_panel, camera_panel_style

        return app

    def _run_server(self):
        self._app.run(
            host=self.host,
            port=self.port,
            debug=False,
            use_reloader=False,
        )

    def _snapshot(self):
        with self._lock:
            if self._positions:
                positions = np.vstack(self._positions)
            else:
                positions = np.empty((0, 3), dtype=float)

            latest_position = None
            if self._latest_position is not None:
                latest_position = self._latest_position.copy()

            latest_direction = None
            if self._latest_direction is not None:
                latest_direction = self._latest_direction.copy()

            camera_topic_available = self._camera_topic_available
            camera_has_frame = self._camera_has_frame
            num_samples = len(self._positions)

        return {
            "positions": positions,
            "latest_position": latest_position,
            "latest_direction": latest_direction,
            "camera_topic_available": camera_topic_available,
            "camera_has_frame": camera_has_frame,
            "num_samples": num_samples,
        }

    def _camera_stream_response(self):
        response = Response(
            self._camera_stream_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    def _camera_stream_generator(self):
        last_frame_index = -1

        while True:
            with self._camera_condition:
                self._camera_condition.wait_for(
                    lambda: (
                        self._camera_frame_bytes is not None
                        and self._camera_frame_index != last_frame_index
                    ),
                    timeout=1.0,
                )

                frame_bytes = self._camera_frame_bytes
                frame_index = self._camera_frame_index

            if frame_bytes is None or frame_index == last_frame_index:
                continue

            last_frame_index = frame_index

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode("ascii")
                + frame_bytes
                + b"\r\n"
            )


def build_goal_direction_figure(
    positions,
    goal_point,
    tag_origin,
    max_render_points=1500,
):
    _ensure_dependencies()

    positions = np.asarray(positions, dtype=float)
    goal_point = np.asarray(goal_point, dtype=float)
    tag_origin = np.asarray(tag_origin, dtype=float)

    displayed_positions = _downsample_positions(positions, max_render_points)
    current_position = displayed_positions[-1] if len(displayed_positions) else None

    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=[tag_origin[0]],
            y=[tag_origin[1]],
            z=[tag_origin[2]],
            mode="markers+text",
            marker={"size": 7, "color": "black"},
            text=["Tag original"],
            textposition="top center",
            name="Tag",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[goal_point[0]],
            y=[goal_point[1]],
            z=[goal_point[2]],
            mode="markers+text",
            marker={"size": 8, "color": "green"},
            text=["Goal"],
            textposition="top center",
            name="Goal",
        )
    )

    if len(displayed_positions):
        figure.add_trace(
            go.Scatter3d(
                x=displayed_positions[:, 0],
                y=displayed_positions[:, 1],
                z=displayed_positions[:, 2],
                mode="lines",
                line={"width": 5, "color": "#1f77b4"},
                name="Trajetoria",
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=[current_position[0]],
                y=[current_position[1]],
                z=[current_position[2]],
                mode="markers+text",
                marker={"size": 7, "color": "#1f77b4"},
                text=["Drone"],
                textposition="top center",
                name="Drone",
            )
        )
        direction = goal_point - current_position
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0.0:
            scene_span = _scene_span(positions, goal_point, tag_origin)
            arrow_tip, shaft_tip, head_vector = _build_direction_arrow(
                current_position,
                direction,
                scene_span,
            )
            figure.add_trace(
                go.Scatter3d(
                    x=[current_position[0], goal_point[0]],
                    y=[current_position[1], goal_point[1]],
                    z=[current_position[2], goal_point[2]],
                    mode="lines",
                    line={"width": 3, "color": "rgba(214, 39, 40, 0.25)"},
                    name="Linha para goal",
                )
            )
            figure.add_trace(
                go.Scatter3d(
                    x=[current_position[0], shaft_tip[0]],
                    y=[current_position[1], shaft_tip[1]],
                    z=[current_position[2], shaft_tip[2]],
                    mode="lines",
                    line={"width": 8, "color": "#ff5a36"},
                    name="Vetor para goal",
                )
            )
            figure.add_trace(
                go.Cone(
                    x=[arrow_tip[0]],
                    y=[arrow_tip[1]],
                    z=[arrow_tip[2]],
                    u=[head_vector[0]],
                    v=[head_vector[1]],
                    w=[head_vector[2]],
                    anchor="tip",
                    showscale=False,
                    sizemode="absolute",
                    sizeref=max(np.linalg.norm(head_vector) * 0.75, 0.08),
                    colorscale=[[0.0, "#ff5a36"], [1.0, "#ff5a36"]],
                    name="Direcao",
                )
            )

    axis_ranges = _axis_ranges(positions, goal_point, tag_origin)
    figure.update_layout(
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
        scene={
            "xaxis": {"title": "X [m]", "range": axis_ranges["x"]},
            "yaxis": {"title": "Y [m]", "range": axis_ranges["y"]},
            "zaxis": {"title": "Z [m]", "range": axis_ranges["z"]},
            "aspectmode": "cube",
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )

    return figure


def build_status_panel(latest_position, latest_direction, num_samples):
    if latest_position is None or latest_direction is None:
        return html.Div(
            "Aguardando a primeira pose do drone...",
            style={"marginBottom": "12px", "fontSize": "16px"},
        )

    return html.Div(
        [
            html.Div(f"Amostras recebidas: {num_samples}"),
            html.Div(
                "Posicao do drone: "
                f"x={latest_position[0]:.3f}, "
                f"y={latest_position[1]:.3f}, "
                f"z={latest_position[2]:.3f}"
            ),
            html.Div(
                "Vetor para o goal: "
                f"dx={latest_direction[0]:.3f}, "
                f"dy={latest_direction[1]:.3f}, "
                f"dz={latest_direction[2]:.3f}"
            ),
        ],
        style={"marginBottom": "12px", "fontSize": "16px"},
    )


def build_camera_panel(camera_topic_available, camera_has_frame):
    if not camera_topic_available:
        return [], {"display": "none"}

    panel_style = {
        "flex": "1 1 340px",
        "minWidth": "320px",
        "maxWidth": "520px",
        "padding": "12px",
        "border": "1px solid #d9d9d9",
        "borderRadius": "10px",
        "backgroundColor": "#fafafa",
    }

    if not camera_has_frame:
        return (
            html.Div(
                [
                    html.H4("Camera auxiliar", style={"marginTop": "0"}),
                    html.Div("Topico encontrado. Aguardando frames..."),
                ]
            ),
            panel_style,
        )

    return (
        html.Div(
            [
                html.H4("Camera auxiliar", style={"marginTop": "0"}),
                html.Div(
                    "/camera/camera/color/image_raw",
                    style={"marginBottom": "8px", "fontSize": "14px"},
                ),
                html.Img(
                    src="/camera_stream.mjpg",
                    style={
                        "width": "100%",
                        "height": "auto",
                        "display": "block",
                        "borderRadius": "8px",
                    },
                ),
            ]
        ),
        panel_style,
    )


def _downsample_positions(positions, max_render_points):
    if len(positions) <= max_render_points:
        return positions

    indices = np.linspace(0, len(positions) - 1, max_render_points).astype(int)
    return positions[indices]


def _axis_ranges(positions, goal_point, tag_origin):
    span = _scene_span(positions, goal_point, tag_origin)
    all_points = [goal_point, tag_origin]
    if len(positions):
        all_points.append(np.asarray(positions, dtype=float))

    stacked = np.vstack(all_points)
    center = (stacked.max(axis=0) + stacked.min(axis=0)) / 2.0
    half_extent = span / 2.0 + 0.2 * span

    return {
        "x": [center[0] - half_extent, center[0] + half_extent],
        "y": [center[1] - half_extent, center[1] + half_extent],
        "z": [center[2] - half_extent, center[2] + half_extent],
    }


def _scene_span(positions, goal_point, tag_origin):
    all_points = [goal_point, tag_origin]
    if len(positions):
        all_points.append(np.asarray(positions, dtype=float))

    stacked = np.vstack(all_points)
    span = stacked.max(axis=0) - stacked.min(axis=0)
    return max(float(span.max()), 1.0)


def _build_direction_arrow(current_position, direction, scene_span):
    direction = np.asarray(direction, dtype=float)
    current_position = np.asarray(current_position, dtype=float)

    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0.0:
        return current_position, current_position, np.zeros(3, dtype=float)

    direction_unit = direction / direction_norm
    arrow_length = min(direction_norm, max(scene_span * 0.16, 0.3))
    head_length = min(max(scene_span * 0.05, 0.1), arrow_length * 0.4)

    arrow_tip = current_position + direction_unit * arrow_length
    head_vector = direction_unit * head_length
    shaft_tip = arrow_tip - head_vector

    return arrow_tip, shaft_tip, head_vector


def _ensure_dependencies():
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing web visualization dependencies. "
            "Install them with: pip install dash plotly"
        ) from _IMPORT_ERROR
