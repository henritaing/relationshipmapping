import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import json
import itertools

# Define the Person class
class Person:
    def __init__(self, first_name, last_name, age, sex, hobby, job, met_at):
        self.first_name = first_name
        self.last_name = last_name
        self.age = int(age)
        self.sex = sex
        self.hobby = hobby
        self.job = job
        self.met_at = met_at
        self.connections = []

    def add_connection(self, other_person):
        if other_person not in self.connections:
            self.connections.append(other_person)

    def full_name(self):
        return f"{self.first_name} {self.last_name}"

# Load from JSON
def load_people_from_json(uploaded_file):
    people_dict = {}
    if uploaded_file is not None:
        data = json.load(uploaded_file)
        for person_data in data["people"]:
            p = Person(**{k: person_data[k] for k in ["first_name", "last_name", "age", "sex", "hobby", "job", "met_at"]})
            people_dict[p.full_name()] = p
        # Add connections
        for person_data in data["people"]:
            p = people_dict[f"{person_data['first_name']} {person_data['last_name']}"]
            for conn_name in person_data.get("connections", []):
                if conn_name in people_dict:
                    p.add_connection(people_dict[conn_name])
    return people_dict

# Manual input
def add_person_ui(people_dict):
    with st.form("Add Person"):
        st.write("Add a new person:")
        fname = st.text_input("First name")
        lname = st.text_input("Last name")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        sex = st.selectbox("Sex", ["M", "F", "Other"])
        hobby = st.text_input("Hobby (Running, Volleyball, Travel, etc.)")
        job = st.text_input("What's your job? Or are you a student?")
        met_at = st.text_input("Where did you meet?")
        submitted = st.form_submit_button("Add")

        if submitted and fname and lname:
            new_person = Person(fname, lname, age, sex, hobby, job, met_at)
            people_dict[new_person.full_name()] = new_person
            st.success(f"Added {new_person.full_name()}")

    if people_dict:
        st.write("Now, define their connections:")
        names = list(people_dict.keys())
        for person_name in names:
            options = st.multiselect(f"Who does {person_name} know?", [n for n in names if n != person_name], key=person_name)
            person = people_dict[person_name]
            person.connections = [people_dict[n] for n in options]


# Visualize graph
def draw_graph(people_dict, group_by=None):
    G = nx.Graph()
    for name, person in people_dict.items():
        G.add_node(name, met_at=person.met_at, hobby=person.hobby)
        for conn in person.connections:
            G.add_edge(name, conn.full_name())

    annotations = []
    group_colors = {}
    pos = {}

    if group_by in ["met_at", "hobby"]:
        # Group nodes
        groups = {}
        for node, data in G.nodes(data=True):
            key = data.get(group_by, "Unknown")
            groups.setdefault(key, []).append(node)

        # Assign colors to groups
        colors = itertools.cycle([
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
            "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"
        ])
        group_colors = {group: next(colors) for group in groups}

        # Assign layout positions for each group
        spacing = 5.0
        group_centers = {}
        for idx, group in enumerate(groups):
            group_centers[group] = (idx * spacing, 0)

        for group, nodes in groups.items():
            cx, cy = group_centers[group]
            for i, node in enumerate(nodes):
                x = cx + (i % 3) * 1.0
                y = cy + (i // 3) * 1.0
                pos[node] = (x, y)

            # annotations.append(dict(
            #     x=cx,
            #     y=cy + 2.5,
            #     text=f"{group}",
            #     showarrow=False,
            #     font=dict(size=14, color="black")
            # ))

    else:
        pos = nx.spring_layout(G, seed=42)

    # Draw edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1, color='#888'),
        hoverinfo='none'
    )

    legend_traces = {}
    # Draw nodes
    node_traces = []
    for node in G.nodes():
        x, y = pos[node]
        group_val = G.nodes[node].get(group_by, "Unknown") if group_by else None
        color = group_colors.get(group_val, "#636EFA") if group_by else "#636EFA"

        # Main node trace
        trace = go.Scatter(
            x=[x], y=[y], mode='markers+text',
            text=[node], textposition='top center',
            hoverinfo='text',
            marker=dict(size=5, color=color, line=dict(width=2)),
            name=node,
            showlegend=False
        )
        node_traces.append(trace)

        # Add to legend once per group
        if group_by and group_val not in legend_traces:
            legend_traces[group_val] = go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=5, color=color),
                legendgroup=group_val,
                name=f"{group_val}",
                showlegend=True
            )
    
    fig = go.Figure(data=[edge_trace] + list(legend_traces.values()) + node_traces)

    fig.update_layout(
        title="Your Relationship Map",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        margin=dict(b=20, l=5, r=5, t=40),
        # annotations=annotations
    )
    st.plotly_chart(fig, use_container_width=True)

    return fig


# Streamlit app start
st.title("📍 Interactive Relationship Mapper")
source = st.radio("Choose input method", ["Upload JSON", "Manual Input"])
st.text("If you plan to use a JSON, please follow the format : ")
st.html("<p>{'people': [{'first_name': 'Pou', 'last_name': 'Let', 'age': 32, 'sex': 'M', 'hobby': 'Volleyball', 'job': 'QA Engineer', 'met_at': 'HS', 'connections': ['Alice A', 'Anais V']}, {'first_name': 'Ro', 'last_name': 'Lex', 'age': 32, 'sex': 'M', 'hobby': 'Volleyball', 'job': 'QA Engineer', 'met_at': 'HS', 'connections': ['Alice A', 'Anais V']}]}</p>")

grouping_option = st.selectbox("Group nodes by", ["None", "met_at", "hobby"])

# Initialize people_dict here to preserve state
if 'people_dict' not in st.session_state:
    st.session_state.people_dict = {}

people_dict = st.session_state.people_dict

if source == "Upload JSON":
    uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])
    if uploaded_file:
        people_dict = load_people_from_json(uploaded_file)
        st.session_state.people_dict = people_dict  # Store it in session state
        st.success("People loaded from file.")
elif source == "Manual Input":
    add_person_ui(people_dict)
    st.session_state.people_dict = people_dict  # Store it in session state

if people_dict:
    fig = draw_graph(people_dict, group_by=grouping_option)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📤 Export Graph as HTML"):
            fig.write_html("relationship_map.html")
            with open("relationship_map.html", "rb") as f:
                st.download_button("Download HTML", f, "relationship_map.html", "text/html")

    with col2:
        if st.button("🖼️ Export Graph as PNG"):
            fig.write_image("relationship_map.png", format="png", width=1000, height=800)
            with open("relationship_map.png", "rb") as f:
                st.download_button("Download PNG", f, "relationship_map.png", "image/png")


