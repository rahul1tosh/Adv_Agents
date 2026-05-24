from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage
from langchain_groq import ChatGroq

from pydantic import BaseModel
from typing import Callable

import inspect
import difflib
from dotenv import load_dotenv
load_dotenv()


class State(BaseModel):
    function: Callable
    function_string: str
    arguments: list
    error: bool
    error_description: str = ""
    new_function_string: str = ""


class Agent:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.llm = ChatGroq(model=model_name)
        self.graph = self.build_graph()

    def code_execution_node(self, state: State):
        try:
            result = state.function(*state.arguments)
        except Exception as e:
            state.error = True
            state.error_description = str(e)
        return state

    def code_fix_node(self, state: State):
        prompt = ChatPromptTemplate.from_template(
            "Fix this Python function that raised an error. "
            "Function: {function_string} "
            "Error: {error_description} "
            "Handle the error gracefully by returning an error message. "
            "Use the exact same name and parameters. "
            "Return only the function definition with no additional text or formatting."
        )
        message = HumanMessage(
            content=prompt.format(
                function_string=state.function_string,
                error_description=state.error_description,
            )
        )
        new_function_string = self.llm.invoke([message]).content.strip()
        state.new_function_string = new_function_string

        namespace = {}
        exec(state.new_function_string, namespace)
        state.function = namespace[state.function.__name__]
        state.function(*state.arguments)
        state.error = False
        return state

    def error_router(self, state: State):
        return "code_fix_node" if state.error else END

    def build_graph(self):
        builder = StateGraph(State)
        builder.add_node("code_execution_node", self.code_execution_node)
        builder.add_node("code_fix_node", self.code_fix_node)

        builder.set_entry_point("code_execution_node")
        builder.add_conditional_edges("code_execution_node", self.error_router)
        builder.add_edge("code_fix_node", "code_execution_node")

        return builder.compile()

    def execute(self, function, arguments):
        state = State(
            error=False,
            function=function,
            function_string=inspect.getsource(function),
            arguments=arguments,
        )
        final_state =  self.graph.invoke(state)
        return final_state["function"](*arguments)


class App:
    def __init__(self):
        self.agent = Agent()
        self.original_code = ""
        self.fixed_code = ""

    def run_agent_and_show_diff(self, code_string, arguments_str):
        """Execute the agent and capture original and fixed code."""
        import json

        try:
            arguments = json.loads(arguments_str)
            if not isinstance(arguments, list):
                arguments = [arguments]
        except json.JSONDecodeError:
            return code_string, "Error parsing arguments", '<pre style="color: red;">Invalid JSON format for arguments</pre>', "Error: Invalid arguments"

        try:
            namespace = {}
            exec(code_string, namespace)
            function = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('__'):
                    function = obj
                    break

            if function is None:
                return code_string, "No function found", '<pre style="color: red;">No function found in the code</pre>', "Error: No function found"
        except Exception as e:
            return code_string, f"Error: {str(e)}", f'<pre style="color: red;">Error executing code: {str(e)}</pre>', f"Error: {str(e)}"

        state = State(
            error=False,
            function=function,
            function_string=code_string,
            arguments=arguments,
        )

        self.original_code = code_string

        final_state = self.agent.graph.invoke(state)

        self.fixed_code = final_state.get("new_function_string", "")

        if self.fixed_code:
            diff = self._generate_colored_diff_html(self.original_code, self.fixed_code)
        else:
            diff = '<pre style="font-family: monospace; background-color: #f6f8fa; padding: 10px; border-radius: 5px; color: #1a7f37;">No errors detected - code executed successfully!</pre>'

        try:
            result = final_state["function"](*arguments)
        except Exception as e:
            result = f"Error: {str(e)}"

        return self.original_code, self.fixed_code or "No fix needed", diff, str(result)

    def _generate_diff(self, original, fixed):
        """Generate a unified diff like git diff."""
        original_lines = original.splitlines()
        fixed_lines = fixed.splitlines()

        diff = difflib.unified_diff(
            original_lines,
            fixed_lines,
            fromfile='original',
            tofile='fixed',
            lineterm=''
        )

        return '\n'.join(diff)

    def _generate_colored_diff_html(self, original, fixed):
        """Generate a colored HTML diff like git diff."""
        import html

        diff_text = self._generate_diff(original, fixed)

        if not diff_text:
            return '<pre style="font-family: monospace; background-color: #f6f8fa; padding: 10px; border-radius: 5px;">No differences found</pre>'

        lines = diff_text.split('\n')
        html_lines = []

        for line in lines:
            escaped_line = html.escape(line)
            if line.startswith('---') or line.startswith('+++'):
                # File headers - bold
                html_lines.append(f'<div style="font-weight: bold;">{escaped_line}</div>')
            elif line.startswith('@@'):
                # Line numbers - cyan
                html_lines.append(f'<div style="color: #0969da;">{escaped_line}</div>')
            elif line.startswith('-'):
                # Removed lines - red background
                html_lines.append(f'<div style="background-color: #ffebe9; color: #cf222e;">{escaped_line}</div>')
            elif line.startswith('+'):
                # Added lines - green background
                html_lines.append(f'<div style="background-color: #dafbe1; color: #1a7f37;">{escaped_line}</div>')
            else:
                # Context lines
                html_lines.append(f'<div>{escaped_line}</div>')

        html_content = ''.join(html_lines)
        return f'<pre style="font-family: monospace; background-color: #f6f8fa; padding: 10px; border-radius: 5px; overflow-x: auto;">{html_content}</pre>'

    def launch(self):
        """Launch the Gradio interface."""
        import gradio as gr

        # Example functions with their test arguments
        examples = {
            "Division by Zero": {
                "code": """def divide_two_numbers(a, b):
    return a / b""",
                "args": "[10, 0]"
            },
            "E-commerce: 100% Discount Price": {
                "code": """def calculate_total_price(items, discount_percent):
    subtotal = sum(item['price'] * item['quantity'] for item in items)
    discount_amount = subtotal * (discount_percent / 100)
    margin = subtotal / (subtotal - discount_amount)
    final_price = subtotal - discount_amount
    return final_price""",
                "args": '[[{"price": 10, "quantity": 2}], 100]'
            },
            "Invalid Coupon Code": {
                "code": """def apply_coupon_code(cart_total, coupon_code):
    coupons = {
        'SAVE10': 0.10,
        'SAVE20': 0.20,
        'BLACKFRIDAY': 0.50
    }
    discount = coupons[coupon_code]
    return cart_total * (1 - discount)""",
                "args": '[100, "INVALID"]'
            },
            "Product Index Out of Range": {
                "code": """def get_product_price(products, product_id):
    return products[product_id]['price']""",
                "args": '[[{"price": 10}, {"price": 20}], 5]'
            },
            "Bulk Discount - Zero Quantity": {
                "code": """def calculate_bulk_discount(quantity, unit_price):
    avg_discount = 100 / quantity
    if quantity >= 100:
        discount = 0.20
    elif quantity >= 50:
        discount = 0.10
    elif quantity >= 10:
        discount = 0.05
    else:
        discount = 0.0
    return quantity * unit_price * (1 - discount)""",
                "args": "[0, 10.99]"
            },
            "Format Price - None Value": {
                "code": """def format_price_display(price):
    return f"${price:.2f}" """,
                "args": "[null]"
            },
            "Inventory - None in Calculation": {
                "code": """def get_inventory_count(inventory, sku):
    return inventory.get(sku)""",
                "args": '[{"ABC123": 10}, "XYZ999"]'
            }
        }

        def load_example(example_name):
            if example_name in examples:
                return examples[example_name]["code"], examples[example_name]["args"]
            return "", "[]"

        with gr.Blocks(title="Agentic Bug Fixing - Code Diff Viewer") as demo:
            gr.Markdown("# Agentic Bug Fixing - Code Diff Viewer")
            gr.Markdown("Enter your Python function code and arguments to see how the agent fixes bugs.")

            example_dropdown = gr.Dropdown(
                choices=list(examples.keys()),
                label="Load Example",
                value="Division by Zero"
            )

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Python Function Code")
                    code_input = gr.Code(
                        value=examples["Division by Zero"]["code"],
                        language="python",
                        label="",
                        lines=12
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### Function Arguments")
                    arguments_input = gr.Textbox(
                        value=examples["Division by Zero"]["args"],
                        label="Arguments (JSON format)",
                        placeholder='[10, 0] or {"key": "value"}',
                        lines=2
                    )
                    run_btn = gr.Button("Run Agent & Show Diff", variant="primary", size="lg")

            gr.Markdown("### Git-Style Diff")
            diff_output = gr.HTML(label="")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Original Code")
                    original_output = gr.Code(language="python", label="")

                with gr.Column():
                    gr.Markdown("### Fixed Code")
                    fixed_output = gr.Code(language="python", label="")

            gr.Markdown("### Execution Result")
            result_output = gr.Textbox(label="", lines=3)

            example_dropdown.change(
                fn=load_example,
                inputs=[example_dropdown],
                outputs=[code_input, arguments_input]
            )

            run_btn.click(
                fn=self.run_agent_and_show_diff,
                inputs=[code_input, arguments_input],
                outputs=[original_output, fixed_output, diff_output, result_output]
            )

        demo.launch()


if __name__ == "__main__":
    a = Agent()
    def divide_two_numbers(a, b):
        return a / b
    answer_1 = a.execute(divide_two_numbers, [10, 0])
    answer_2 = divide_two_numbers(10, 0)


    # app = App()
    # app.launch()

