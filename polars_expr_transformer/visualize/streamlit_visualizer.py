from polars_expr_transformer.process.polars_expr_transformer import build_func
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import uuid


def build_expression_graph(func_obj):
    G = nx.DiGraph()
    node_meta = {}
    root_id = str(uuid.uuid4())
    _build_graph(G, node_meta, func_obj, root_id)
    return G, node_meta, root_id


def _is_pl_lit_wrapper(obj):
    if not hasattr(obj, '__class__') or obj.__class__.__name__ != 'Func':
        return False
    if not hasattr(obj.func_ref, 'val') or obj.func_ref.val != 'pl.lit':
        return False
    if len(obj.args) != 1:
        return False
    return hasattr(obj.args[0], '__class__') and obj.args[0].__class__.__name__ == 'Func'


def _get_unwrapped_obj(obj):
    if _is_pl_lit_wrapper(obj):
        return _get_unwrapped_obj(obj.args[0])
    return obj


def _build_graph(G, node_meta, obj, node_id, parent_id=None, edge_label=None):
    if hasattr(obj, '__class__') and obj.__class__.__name__ == 'TempFunc':
        if hasattr(obj, 'args') and obj.args:
            return _build_graph(G, node_meta, obj.args[0], node_id, parent_id, edge_label)
        else:
            G.add_node(node_id)
            node_meta[node_id] = {'type': 'tempfunc', 'label': 'TempFunc'}
            if parent_id:
                G.add_edge(parent_id, node_id, label=edge_label or '')
            return

    obj = _get_unwrapped_obj(obj)

    if obj is None:
        return

    if hasattr(obj, '__class__'):
        class_name = obj.__class__.__name__
    else:
        class_name = "Unknown"

    if class_name == "Func":
        func_name = obj.func_ref.val if hasattr(obj.func_ref, 'val') else str(obj.func_ref)

        G.add_node(node_id)
        node_meta[node_id] = {
            'type': 'func',
            'label': f"{func_name}",
            'func_name': func_name
        }

        if parent_id:
            G.add_edge(parent_id, node_id, label=edge_label or '')

        for i, arg in enumerate(obj.args):
            arg_id = str(uuid.uuid4())
            arg_label = f"Arg {i + 1}"
            _build_graph(G, node_meta, arg, arg_id, node_id, arg_label)

    elif class_name == "IfFunc":
        G.add_node(node_id)
        node_meta[node_id] = {'type': 'ifunc', 'label': 'If'}

        if parent_id:
            G.add_edge(parent_id, node_id, label=edge_label or '')

        for i, condition_val in enumerate(obj.conditions):
            cond_id = str(uuid.uuid4())
            G.add_node(cond_id)
            node_meta[cond_id] = {'type': 'condition', 'label': f'Cond {i + 1}'}
            G.add_edge(node_id, cond_id, label=f'Condition: {i + 1}')

            if hasattr(condition_val, 'condition') and condition_val.condition:
                expr_id = str(uuid.uuid4())
                G.add_node(expr_id)
                if hasattr(condition_val.condition, 'get_readable_pl_function'):
                    readable_expr = condition_val.condition.get_readable_pl_function()
                    if len(readable_expr) > 15:
                        readable_expr = readable_expr[:12] + "..."
                    node_meta[expr_id] = {'type': 'expr', 'label': readable_expr}
                else:
                    node_meta[expr_id] = {'type': 'expr', 'label': 'Expr'}
                G.add_edge(cond_id, expr_id, label='When')
                _build_graph(G, node_meta, condition_val.condition, expr_id)

            if hasattr(condition_val, 'val') and condition_val.val:
                then_id = str(uuid.uuid4())
                G.add_node(then_id)
                if hasattr(condition_val.val, 'get_readable_pl_function'):
                    readable_then = condition_val.val.get_readable_pl_function()
                    if len(readable_then) > 15:
                        readable_then = readable_then[:12] + "..."
                    node_meta[then_id] = {'type': 'then', 'label': readable_then}
                else:
                    node_meta[then_id] = {'type': 'then', 'label': 'Then'}
                G.add_edge(cond_id, then_id, label='Then')
                _build_graph(G, node_meta, condition_val.val, then_id)

        if obj.else_val:
            else_id = str(uuid.uuid4())
            G.add_node(else_id)
            if hasattr(obj.else_val, 'get_readable_pl_function'):
                readable_else = obj.else_val.get_readable_pl_function()
                if len(readable_else) > 15:
                    readable_else = readable_else[:12] + "..."
                node_meta[else_id] = {'type': 'else', 'label': readable_else}
            else:
                node_meta[else_id] = {'type': 'else', 'label': 'Else'}
            G.add_edge(node_id, else_id, label='else')
            _build_graph(G, node_meta, obj.else_val, else_id)

    elif class_name == "Classifier":
        val = obj.val if hasattr(obj, 'val') else str(obj)
        val_type = obj.val_type if hasattr(obj, 'val_type') else ""

        G.add_node(node_id)

        if val_type in ["number", "string", "boolean"]:
            display_val = f'"{val}"' if val_type == "string" else val
            node_meta[node_id] = {
                'type': 'value',
                'label': f"{display_val}",
                'value': val,
                'value_type': val_type
            }
        else:
            node_meta[node_id] = {
                'type': 'classifier',
                'label': f"{val}",
                'value': val
            }

        if parent_id:
            G.add_edge(parent_id, node_id, label=edge_label or '')

    else:
        display_val = obj.val if hasattr(obj, 'val') else str(obj)
        G.add_node(node_id)
        node_meta[node_id] = {'type': 'other', 'label': f"{display_val}"}

        if parent_id:
            G.add_edge(parent_id, node_id, label=edge_label or '')


def visualize_expression(expr):
    try:
        func_obj = build_func(expr)
        func_obj.get_pl_func()
        G, node_meta, root_id = build_expression_graph(func_obj)

        # Modern color palette
        node_colors = {
            'func': '#F59E0B',  # Amber-500
            'ifunc': '#EC4899',  # Pink-500
            'condition': '#BE185D',  # Pink-800
            'expr': '#8B5CF6',  # Violet-500
            'then': '#3B82F6',  # Blue-500
            'else': '#06B6D4',  # Cyan-500
            'value': '#10B981',  # Emerald-500
            'classifier': '#6366F1',  # Indigo-500
            'other': '#6B7280',  # Gray-500
            'tempfunc': '#9CA3AF'  # Gray-400
        }

        # Compact node sizes
        node_sizes = {
            'func': 1200,
            'ifunc': 1500,
            'condition': 1300,
            'expr': 1100,
            'then': 1100,
            'else': 1100,
            'value': 1000,
            'classifier': 1100,
            'other': 900,
            'tempfunc': 800
        }

        # Create figure with clean background
        plt.figure(figsize=(10, 6), facecolor='white', dpi=100, tight_layout=True)
        ax = plt.gca()
        ax.set_facecolor('white')

        # Layout
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

        # Prepare node attributes
        colors = [node_colors.get(node_meta[node]['type'], '#6B7280') for node in G.nodes()]
        sizes = [node_sizes.get(node_meta[node]['type'], 1200) for node in G.nodes()]

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, alpha=0.95,
                               edgecolors='white', linewidths=1.0, node_shape='o')

        # Draw edges
        nx.draw_networkx_edges(G, pos, width=0.9, arrowsize=12, alpha=0.8,
                               edge_color='#94A3B8', arrows=True,
                               connectionstyle='arc3,rad=0.08',
                               min_source_margin=12, min_target_margin=12)

        # Node labels
        labels = {node: node_meta[node]['label'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7,
                                font_family='sans-serif', font_weight='normal',
                                font_color='#27272A')

        # Edge labels
        edge_labels = {(u, v): d.get('label', '')
                       for u, v, d in G.edges(data=True) if d.get('label', '')}

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6,
                                     font_color='#475569', font_weight='normal',
                                     bbox=dict(facecolor='white', edgecolor='none',
                                               alpha=0.7, pad=0.5))

        plt.axis('off')
        plt.title(expr, fontsize=10, loc='center', pad=5, color='#64748B')

        # Remove frame border
        for spine in ax.spines.values():
            spine.set_visible(False)

        return plt

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def streamlit_app():
    st.set_page_config(
        page_title="Expression Visualizer",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Minimal CSS
    st.markdown("""
    <style>
    .stApp {
        font-family: sans-serif;
    }
    .main {
        padding: 1rem;
    }
    .diagram-container {
        background-color: white;
        border-radius: 4px;
        padding: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Just a simple input and visualization
    expr = st.text_input("Expression:", value="if 1==2 then True else False")

    if st.button("Visualize", type="primary"):
        with st.spinner(""):
            plt = visualize_expression(expr)
            if plt is not None:
                st.pyplot(plt)


if __name__ == "__main__":
    streamlit_app()