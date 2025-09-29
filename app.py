import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import io, json, base64
from PIL import Image, ImageOps, ImageDraw
import numpy as np

# ---------------------
# Helpers
# ---------------------
# ---------------------
# Helpers
# ---------------------
def make_circle(img_bytes, size=(100, 100)):
    """Return a circular PNG (bytes) from uploaded image (no border)."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    img = img.resize(size, Image.LANCZOS)

    # Mask for circle
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size[0], size[1]), fill=255)

    result = ImageOps.fit(img, size, centering=(0.5, 0.5))
    result.putalpha(mask)

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return buffer.getvalue()

class Person:
    def __init__(self, name, image_bytes=None, border_color=(0,0,0,255), border_width=5):
        self.name = name
        self.image_bytes = image_bytes
        self.border_color = border_color
        self.border_width = border_width
        self.connections = []

    def add_connection(self, other):
        if other not in self.connections:
            self.connections.append(other)

# ---------------------
# UI Components
# ---------------------
def add_people_ui(people_dict):
    st.subheader("Step 1: Add People")
    if "temp_name" not in st.session_state:
        st.session_state.temp_name = ""

    # UI for border customization
    border_color_hex = st.color_picker("Border color", "#000000")

    with st.form("add_person_form", clear_on_submit=True):
        name = st.text_input("Name", key="temp_name")
        uploaded_img = st.file_uploader("Upload a picture (optional)", type=["png", "jpg", "jpeg"])
        submitted = st.form_submit_button("Add")

        if submitted and name:
            img_bytes = None
            rgba = tuple(int(border_color_hex[i:i+2], 16) for i in (1, 3, 5)) + (255,)
            if uploaded_img:
                img_bytes = make_circle(uploaded_img.read())  # plus de bordure ici
            people_dict[name] = Person(name, img_bytes, border_color=rgba, border_width=5)
            st.success(f"Added {name}")


    if people_dict:
        st.write("✅ People added:")
        st.write(", ".join(people_dict.keys()))

def add_connections_ui(people_dict):
    st.subheader("Step 2: Define Connections")
    names = list(people_dict.keys())

    for person_name in names:
        already_connected_to_me = [
            p.name for p in people_dict.values() if person_name in [c.name for c in p.connections]
        ]
        available_options = [n for n in names if n != person_name and n not in already_connected_to_me]

        default_selection = [
            p.name for p in people_dict[person_name].connections if p.name in available_options
        ]

        selected_names = st.multiselect(
            f"{person_name} knows:",
            options=available_options,
            default=default_selection,
            key=f"conn_{person_name}"
        )

        people_dict[person_name].connections = [people_dict[n] for n in selected_names]

# ---------------------
# Graph Drawing (unchanged)
# ---------------------
def draw_graph(people_dict, image_size=0.3, show_names=True):
    G = nx.Graph()
    for name, person in people_dict.items():
        G.add_node(name, image=person.image_bytes)
        for conn in person.connections:
            G.add_edge(name, conn.name)

    num_nodes = len(G)
    pos = nx.spring_layout(G, seed=42, k=1.5/num_nodes**0.5, iterations=100)

    # Normalize positions to [0,1]
    x_vals = np.array([x for x, y in pos.values()])
    y_vals = np.array([y for x, y in pos.values()])
    pad = 0.1
    x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min()) * (1 - 2*pad) + pad
    y_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min()) * (1 - 2*pad) + pad
    for i, node in enumerate(pos):
        pos[node] = (x_norm[i], y_norm[i])

    # Edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#888"),
        hoverinfo="none"
    )

    # Nodes, images, borders, annotations
    node_traces = []
    images = []
    annotations = []
    border_traces = []

    for node, (x, y) in pos.items():
        person = people_dict[node]
        if person.image_bytes:
            # Image slightly smaller than border
            img_b64 = base64.b64encode(person.image_bytes).decode("ascii")
            images.append(dict(
                source="data:image/png;base64," + img_b64,
                xref="x", yref="y",
                x=x, y=y,
                sizex=image_size * 0.8,
                sizey=image_size * 0.8,
                xanchor="center",
                yanchor="middle",
                layer="above"
            ))

            # Border circle
            border_radius = image_size / 2.2
            circle_points = 50
            theta = np.linspace(0, 2*np.pi, circle_points)
            border_x = x + border_radius * np.cos(theta)
            border_y = y + border_radius * np.sin(theta)

            border_traces.append(go.Scatter(
                x=border_x,
                y=border_y,
                mode="lines",
                line=dict(
                    color=f"rgba{person.border_color}",
                    width=st.session_state.global_border_width * 0.02  # slider global
                ),
                hoverinfo="skip",
                showlegend=False
            ))


            # Name
            if show_names:
                annotations.append(dict(
                    x=x,
                    y=y - border_radius - 0.005,
                    text=node,
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=15),
                    align="center"
                ))
        else:
            node_traces.append(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                text=[node],
                textposition="top center",
                marker=dict(size=image_size*100, color="#636EFA"),
                hoverinfo="text",
                name=node
            ))

    fig = go.Figure([edge_trace] + border_traces + node_traces)
    fig.update_layout(
        title="🎉 Relationship Map",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        images=images,
        annotations=annotations,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)




# ---------------------
# Export / Import
# ---------------------
def export_to_json(people_dict):
    data = {"people": []}
    for p in people_dict.values():
        data["people"].append({
            "name": p.name,
            "connections": [c.name for c in p.connections],
            "image": base64.b64encode(p.image_bytes).decode("ascii") if p.image_bytes else None,
            "border_color": p.border_color,  # sauvegarde RGBA tuple
            "border_width": p.border_width
        })
    return json.dumps(data, indent=2)

def import_from_json(uploaded_file):
    people_dict = {}
    data = json.load(uploaded_file)
    for p in data["people"]:
        img_bytes = base64.b64decode(p["image"]) if p.get("image") else None
        border_color = tuple(p.get("border_color", (0,0,0,255)))
        border_width = p.get("border_width", 5)
        people_dict[p["name"]] = Person(p["name"], img_bytes, border_color, border_width)
    for p in data["people"]:
        person = people_dict[p["name"]]
        for conn_name in p["connections"]:
            if conn_name in people_dict:
                person.add_connection(people_dict[conn_name])
    return people_dict

# ---------------------
# App Start
# ---------------------
st.title("📍 Fun Relationship Mapper")
st.text("Visualize relationships at weddings or events.")

if "people_dict" not in st.session_state:
    st.session_state.people_dict = {}

# Import / Reset
st.subheader("Import or Reset Map")
uploaded_file = st.file_uploader("📥 Import JSON", type=["json"], key="import_json")
if uploaded_file:
    st.session_state.people_dict = import_from_json(uploaded_file)
    st.success("People imported!")

if st.button("🔄 Reset Map", key="reset_map"):
    st.session_state.people_dict = {}
    st.rerun()

# Add people
add_people_ui(st.session_state.people_dict)

# Connections only if Step 1 has at least 2 people
if len(st.session_state.people_dict) >= 2:
    add_connections_ui(st.session_state.people_dict)

    # Deduplicate symmetric connections
    for person in st.session_state.people_dict.values():
        person.connections = [
            c for c in person.connections
            if person.name not in [conn.name for conn in c.connections]
        ]

    # Node size slider
    st.subheader("Adjust node size and border")
    image_size = st.slider("Node image size", min_value=0.05, max_value=1.0, value=0.3, step=0.01)

    # Slider global pour l'épaisseur des bordures
    if "global_border_width" not in st.session_state:
        st.session_state.global_border_width = 5

    st.session_state.global_border_width = st.slider(
        "Border width (global for all nodes)",
        min_value=1,
        max_value=100,
        value=st.session_state.global_border_width
    )


    # Show names toggle
    show_names = st.checkbox("Show names below images", value=True)

    # Draw graph
    draw_graph(st.session_state.people_dict, image_size=image_size, show_names=show_names)

    # Export
    json_str = export_to_json(st.session_state.people_dict)
    st.download_button("📤 Export JSON", json_str, "relationship_map.json", "application/json")
else:
    st.info("👉 Please add at least 2 people before defining connections.")
