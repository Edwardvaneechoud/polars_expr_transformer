from polars_expr_transformer.function_overview import get_expression_overview, get_all_expressions
import streamlit as st
import polars as pl
from polars_expr_transformer.process.polars_expr_transformer import simple_function_to_expr, preprocess, tokenize

st.set_page_config(
    page_title="Polars Expression Transformer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple CSS for basic styling
st.markdown("""
<style>
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #ddd;
        text-align: center;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


def display_dataframe_sample(df):
    """Display a sample of the dataframe with styling"""
    # Convert to pandas for better display in Streamlit
    if isinstance(df, pl.DataFrame):
        pandas_df = df.to_pandas()
    else:
        pandas_df = df

    # Display the styled dataframe
    st.dataframe(pandas_df.head(10), use_container_width=True)


def create_sample_dataframe():
    """Create a sample dataframe with diverse column types for demonstrations"""
    return pl.DataFrame({
        # Numeric columns
        'a': [1, 2, 3, 4, 5],
        'b': [10, 20, 30, 40, 50],
        'c': [100.5, 200.25, 300.75, 400.125, 500.625],

        # String columns
        'text': ['apple', 'banana', 'cherry', 'date', 'elderberry'],
        'category': ['fruit', 'fruit', 'fruit', 'fruit', 'fruit'],
        'description': ['Red and sweet', 'Yellow and soft', 'Red and tart', 'Brown and chewy', 'Purple and tangy'],

        # Date columns
        'date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'],
        'timestamp': ['2023-01-01 08:30:00', '2023-02-01 09:15:30', '2023-03-01 10:45:15',
                      '2023-04-01 11:20:45', '2023-05-01 12:10:20'],

        # Boolean column
        'in_stock': [True, False, True, True, False],

        # Mixed column with nulls
        'price': [1.99, 0.99, None, 2.49, 3.99]
    })


# Add session state to maintain dataframe state
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = create_sample_dataframe()


def run_expression(df, expr, output_column='output'):
    """Run an expression on a dataframe and return the result"""
    try:
        # Create a copy of the dataframe to avoid modifying the original
        result_df = df.clone()
        # Run the expression and store it in the specified output column
        expr_result = simple_function_to_expr(expr)
        result_df = result_df.with_columns(expr_result.alias(output_column))
        return result_df, None
    except Exception as e:
        return None, str(e)


# App Header
st.title("📊 Polars Expression Transformer")
st.markdown(
    "Transform SQL-like string expressions into powerful Polars operations - simplifying data transformations for SQL and Tableau users.")

# Add function count information
all_expressions = get_all_expressions()
function_counts = {}
for category in get_expression_overview():
    function_counts[category.expression_type] = len(category.expressions)

total_functions = len(all_expressions)

# Simple metrics display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Functions", total_functions)
with col2:
    st.metric("Function Categories", len(function_counts))
with col3:
    # Find the category with the most functions
    most_functions_category = max(function_counts.items(), key=lambda x: x[1])
    category_name = most_functions_category[0].replace('_', ' ').title()
    st.metric(f"Most in {category_name}", most_functions_category[1])

# Sidebar content with more convincing description
st.sidebar.markdown('<h1>Polars Expression Transformer</h1>', unsafe_allow_html=True)
st.sidebar.markdown("""
### Why Use This Tool?

**Bridge the Gap Between SQL and Polars**  
Write familiar SQL-like expressions that automatically convert to optimized Polars code.

**Simplify Data Transformations**  
Perform complex data operations without writing verbose code - ideal for analysts and data scientists.

**Perfect for Applications**  
Embed powerful data transformation capabilities in your apps with a simple string-based interface.

**Enhance Productivity**  
Reduce development time with intuitive syntax that's easy to read, write, and maintain.
""")

st.sidebar.markdown("""
### Key Features

📌 **Column References**  
Access columns with simple `[column_name]` syntax

📌 **Rich Expression Support**  
- Math: `+`, `-`, `*`, `/` 
- Comparisons: `=`, `!=`, `<`, `>`, `<=`, `>=`
- Logic: `and`, `or`
- Conditionals: `if/then/else`

📌 **Function Library**  
Over 50+ functions across categories:
- String manipulation
- Date operations
- Mathematical calculations
- Type conversions

📌 **Documentation Support**  
Easily explore available functions and syntax
""")

st.sidebar.markdown("""
### Get Started

1. Try the examples in the "Basic Usage" tab
2. Explore available functions in "Documentation"
3. Run test cases to see the transformer in action

→ [View Source on GitHub](https://github.com/edwardv1/polars-expr-transformer)
→ [Read Documentation](https://github.com/edwardv1/polars-expr-transformer/docs)
""")

# Create tabs for different demos
tab1, tab2, tab3, tab4 = st.tabs(["💡 Quick Start", "🔍 Expression Explorer", "🧪 Test Examples", "📚 Function Reference"])

# Tab 1: Basic Usage
with tab1:
    st.header("Quick Start Guide")
    st.markdown("Transform SQL-like expressions into powerful Polars operations with just a few simple steps.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Sample Data")

        # Add reset button
        if st.button("Reset Sample Data", key="reset_btn"):
            st.session_state.dataframe = create_sample_dataframe()
            st.success("Sample data has been reset!")

        df = st.session_state.dataframe
        display_dataframe_sample(df)

    with col2:
        st.subheader("Your Expression")
        st.markdown("Enter an expression below using column references like `[column_name]`")

        output_column = st.text_input(
            "Name for output column:",
            "result"
        )

        expression = st.text_area(
            "Enter your expression:",
            "if [a] < 3 then [a] * [b] else [a] + [b] endif // Simple conditional",
            height=100
        )

        col_options = st.radio(
            "Result handling:",
            ["Add as a new column", "Replace existing data"],
            horizontal=True
        )

        if st.button("Run Expression", key="run_expr_btn"):
            result, error = run_expression(df, expression, output_column)
            if error:
                st.error(f"Error: {error}")
            else:
                st.success("✅ Expression executed successfully!")
                st.markdown('<div class="sub-header">Result</div>', unsafe_allow_html=True)
                display_dataframe_sample(result)

                # Update the session state with the new dataframe
                if col_options == "Replace existing data":
                    st.session_state.dataframe = result
                    st.info("DataFrame has been updated with the expression result.")
                    # Rerun to show the updated state
                    st.rerun()

# Tab 2: Expression Exploration
with tab2:
    st.header("Expression Explorer")
    st.markdown("See how your expressions are processed under the hood.")

    expr = st.text_area(
        "Enter an expression to explore:",
        "([a] + [b] * 2) / [c] + sqrt(abs([b] - [c])) // Math operation with function",
        height=100,
        key="explore_expr"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Apply to Sample Data")
        output_column = st.text_input(
            "Output column name:",
            "expression_result",
            key="explore_output_column"
        )

        if st.button("Run on Sample Data", key="apply_expr_btn"):
            df = create_sample_dataframe()
            result, error = run_expression(df, expr, output_column)
            if error:
                st.error(f"Error: {error}")
            else:
                st.success("Expression applied successfully!")
                display_dataframe_sample(result)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Expression Analysis")
        if st.button("Analyze Expression", key="explore_expr_btn"):
            st.markdown("##### Preprocessing")
            preprocessed = preprocess(expr)
            st.code(preprocessed)

            st.markdown("##### Tokenization")
            tokens = tokenize(preprocessed)
            st.json(tokens)
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Test Cases
with tab3:
    st.header("Example Test Cases")
    st.markdown("Explore these pre-built examples to understand the transformer.")

    test_cases = {
        "Basic Arithmetic": "[a] + [b] * 2",
        "String Concatenation": "concat([text], ' is delicious')",
        "Conditional Logic": "if [a] < 3 then 'small' else 'large' endif",
        "Multiple Conditions": "if [a] < 2 then 'tiny' elseif [a] < 4 then 'medium' else 'huge' endif",
        "Date Functions": "year(to_date([date]))",
        "String Functions": "uppercase(trim([text]))",
        "Nested Functions": "sqrt(abs([b] - [c]))",
        "Comments Example": "[a] + [b] // Adding columns a and b"
    }

    test_case = st.selectbox("Select a test case:", list(test_cases.keys()))
    expression = test_cases[test_case]

    descriptions = {
        "Basic Arithmetic": "Simple calculation that adds column 'a' to twice the value of column 'b'.",
        "String Concatenation": "Combines text from the 'text' column with a static string.",
        "Conditional Logic": "Uses if/then/else to categorize values in column 'a' as either 'small' or 'large'.",
        "Multiple Conditions": "Expands conditional logic with multiple cases.",
        "Date Functions": "Extracts the year from a date string.",
        "String Functions": "Combines string operations by trimming whitespace and converting to uppercase.",
        "Nested Functions": "Calculates the square root of the absolute difference between columns.",
        "Comments Example": "Demonstrates how to add comments to expressions."
    }

    st.markdown(f"**{test_case}**: {descriptions[test_case]}")
    st.code(expression, language="python")

    col1, col2 = st.columns([1, 2])

    with col1:
        output_column = st.text_input(
            "Output column name:",
            "result",
            key="test_output_column"
        )

        col_options = st.radio(
            "Result handling:",
            ["Add as a new column", "Replace existing data"],
            horizontal=True,
            key="test_case_radio"
        )

    with col2:
        if st.button("Run Test Case", key="test_run_button"):
            df = st.session_state.dataframe
            result, error = run_expression(df, expression, output_column)
            if error:
                st.error(f"Error: {error}")
            else:
                st.success("Test case executed successfully!")
                st.session_state.test_result = result

                # Update the session state with the new dataframe
                if col_options == "Replace existing data":
                    st.session_state.dataframe = result
                    st.info("DataFrame has been updated with the expression result.")
                    st.rerun()

    # Display result if available
    if "test_result" in st.session_state:
        st.subheader("Result")
        display_dataframe_sample(st.session_state.test_result)

# Tab 4: Documentation
with tab4:
    st.header("Function Reference")
    st.markdown("Explore all available functions by category.")

    # Get all expression documentation from the package
    expression_docs = get_expression_overview()

    # Create categories list for selectbox
    categories = [category.expression_type.replace('_', ' ').title() for category in expression_docs]
    categories.extend(["Operators", "Conditional Expressions"])

    # Let user select a category
    col1, col2 = st.columns([1, 2])

    with col1:
        selected_category = st.selectbox("Select a category:", categories)


    # Function for displaying the function table
    def display_function_table(category_obj=None):
        st.markdown("---")
        st.markdown('<div class="function-table">', unsafe_allow_html=True)
        st.markdown("### All Functions in This Category")

        if category_obj and category_obj.expressions:
            # Create a dataframe for the table
            function_table = []
            for f in category_obj.expressions:
                # Get a short description (first line of docstring)
                short_desc = f.doc.strip().split('\n')[0] if f.doc else "No description"
                function_table.append({"Function": f.name, "Description": short_desc})

            # Display as a table
            st.table(function_table)
        elif selected_category == "Operators":
            # Create operators table
            operators_table = [
                {"Operator": "+", "Description": "Addition (numbers) or concatenation (strings)"},
                {"Operator": "-", "Description": "Subtraction"},
                {"Operator": "*", "Description": "Multiplication"},
                {"Operator": "/", "Description": "Division"},
                {"Operator": "=", "Description": "Equal to comparison (also ==)"},
                {"Operator": "!=", "Description": "Not equal to comparison"},
                {"Operator": "<", "Description": "Less than comparison"},
                {"Operator": ">", "Description": "Greater than comparison"},
                {"Operator": "<=", "Description": "Less than or equal to comparison"},
                {"Operator": ">=", "Description": "Greater than or equal to comparison"},
                {"Operator": "and", "Description": "Logical AND operator"},
                {"Operator": "or", "Description": "Logical OR operator"},
                {"Operator": "in", "Description": "Check if value exists in column/string"}
            ]
            st.table(operators_table)
        elif selected_category == "Conditional Expressions":
            # Create conditional expressions table
            conditionals_table = [
                {"Expression": "if...then...else...endif", "Description": "Basic conditional expression"},
                {"Expression": "if...then...elseif...then...else...endif",
                 "Description": "Multiple conditional branches"}
            ]
            st.table(conditionals_table)
        else:
            st.info("No functions available in this category.")

        st.markdown('</div>', unsafe_allow_html=True)


    # Display the selected category's functions
    if selected_category in ["Operators", "Conditional Expressions"]:
        if selected_category == "Operators":
            st.markdown("""
            ## Operators

            Operators allow you to perform calculations, comparisons, and logical operations on columns and values.
            """)

            # Add practical examples
            st.markdown("### Examples")
            examples = {
                "Addition": "[a] + [b]",
                "Multiplication": "[a] * 2",
                "Division": "[a] / [b]",
                "Comparison": "[a] > [b]",
                "Logical AND": "[a] < 3 and [b] > 20",
                "Containment": '"a" in [text]'
            }

            selected_operator_example = st.selectbox("Select an example:", list(examples.keys()),
                                                     key="operator_example")
            st.code(examples[selected_operator_example], language="python")

            # Add button to run the example
            if st.button("Run Example", key="run_operator"):
                df = st.session_state.dataframe
                result, error = run_expression(df, examples[selected_operator_example], "operator_result")
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.success("Operator applied successfully!")
                    display_dataframe_sample(result)

        else:  # Conditional Expressions
            st.markdown("""
            ## Conditional Expressions

            Conditional expressions allow you to apply different transformations based on conditions.
            """)

            # Add practical examples
            st.markdown("### Examples")
            examples = {
                "Simple If-Else": "if [a] < 3 then [a] * [b] else [a] + [b] endif",
                "If-ElseIf-Else": "if [a] < 2 then 'Low' elseif [a] < 4 then 'Medium' else 'High' endif",
                "Nested Conditions": "if [a] > 3 then if [b] > 30 then 'Both high' else 'A high only' endif else 'A low' endif",
                "With Math": "if [a] + [b] > 30 then sqrt([c]) else [c] / 10 endif"
            }

            selected_condition_example = st.selectbox("Select an example:", list(examples.keys()),
                                                      key="condition_example")
            st.code(examples[selected_condition_example], language="python")

            # Add button to run the example
            if st.button("Run Example", key="run_condition"):
                df = st.session_state.dataframe
                result, error = run_expression(df, examples[selected_condition_example], "condition_result")
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.success("Conditional expression applied successfully!")
                    display_dataframe_sample(result)

        # Display the function table for this category
        display_function_table()
    else:
        # Find the selected category
        category_obj = next((cat for cat in expression_docs
                             if cat.expression_type.replace('_', ' ').title() == selected_category), None)

        if category_obj:
            if not category_obj.expressions:
                st.info(f"No functions available in the {selected_category} category.")
            else:
                # Create a select box for functions in this category
                function_names = [func.name for func in category_obj.expressions]
                selected_function = st.selectbox("Select a function:", function_names)

                # Display the selected function's documentation
                func = next((f for f in category_obj.expressions if f.name == selected_function), None)
                if func:
                    # Display function name and documentation
                    st.markdown(f"## {func.name}()")

                    doc = func.doc.strip() if func.doc else "No documentation available"
                    st.markdown(doc)

                    # Show an example expression using this function with proper column references
                    st.markdown("### Try this example:")

                    # Create appropriate examples based on function and available columns
                    if category_obj.expression_type == 'string':
                        if func.name in ['uppercase', 'lowercase', 'trim', 'length', 'titlecase']:
                            example = f'{func.name}([text])'
                        elif func.name in ['left', 'right', 'pad_left', 'pad_right']:
                            example = f'{func.name}([text], 2)'
                        elif func.name in ['replace']:
                            example = f'{func.name}([text], "a", "X")'
                        elif func.name in ['concat']:
                            example = f'{func.name}([text], " - ", "example")'
                        elif func.name in ['count_match', 'contains']:
                            example = f'{func.name}([text], "a")'
                        elif func.name in ['string_similarity']:
                            example = f'{func.name}([text], "apple", "levenshtein")'
                        else:
                            example = f'{func.name}([text])'

                    elif category_obj.expression_type == 'date':
                        if func.name in ['year', 'month', 'day', 'hour', 'minute', 'second']:
                            example = f'{func.name}(to_date([date]))'
                        elif func.name in ['add_days', 'add_years', 'add_hours', 'add_minutes', 'add_seconds']:
                            example = f'{func.name}(to_date([date]), 2)'
                        elif func.name in ['date_diff_days', 'datetime_diff_seconds']:
                            example = f'{func.name}(to_date([date]), to_date("2023-01-01"))'
                        elif func.name in ['now', 'today']:
                            example = func.name + '()'
                        else:
                            example = f'{func.name}(to_date([date]))'

                    elif category_obj.expression_type == 'math':
                        if func.name in ['abs', 'sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'ceil', 'floor', 'tanh']:
                            example = f'{func.name}([a])'
                        elif func.name in ['round']:
                            example = f'{func.name}([a] / 3, 2)'
                        elif func.name in ['random_int']:
                            example = f'{func.name}(1, 10)'
                        else:
                            example = f'{func.name}([a])'

                    elif category_obj.expression_type == 'logic':
                        if func.name in ['equals', 'does_not_equal', 'contains']:
                            example = f'{func.name}([a], [b])'
                        elif func.name in ['is_empty', 'is_not_empty', 'is_string']:
                            example = f'{func.name}([text])'
                        else:
                            example = f'{func.name}([a], [b])'

                    elif category_obj.expression_type == 'type_conversions':
                        if func.name in ['to_string', 'to_integer', 'to_float', 'to_number', 'to_boolean',
                                         'to_decimal']:
                            example = f'{func.name}([a])'
                        elif func.name in ['to_date']:
                            example = f'{func.name}([date])'
                        elif func.name in ['to_datetime']:
                            example = f'{func.name}([date], "%Y-%m-%d")'
                        else:
                            example = f'{func.name}([a])'
                    else:
                        example = f'{func.name}()'

                    # Display example with ability to edit
                    user_example = st.text_area("Edit expression if needed:", example, key=f"example_{func.name}")

                    # Add column selection
                    output_column = st.text_input("Output column name:", f"{func.name}_result",
                                                  key=f"output_{func.name}")

                    # Add button to run the example
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        run_option = st.radio(
                            "Result handling:",
                            ["Add as a new column", "Replace existing data"],
                            key=f"option_{func.name}",
                            horizontal=True
                        )

                    with col2:
                        if st.button("Run Example", key=f"run_{func.name}"):
                            df = st.session_state.dataframe
                            result, error = run_expression(df, user_example, output_column)
                            if error:
                                st.error(f"Error: {error}")
                            else:
                                st.success(f"Function '{func.name}' applied successfully!")
                                display_dataframe_sample(result)

                                # Update the session state with the new dataframe if requested
                                if run_option == "Replace existing data":
                                    st.session_state.dataframe = result
                                    st.info("DataFrame has been updated with the expression result.")
                                    st.rerun()

                # Display the function table for this category
                display_function_table(category_obj)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2023-2025 Polars Expression Transformer | Created by Edward van Eechoud | <a href="https://github.com/edwardv1/polars-expr-transformer" target="_blank">GitHub</a></p>
</div>
""", unsafe_allow_html=True)